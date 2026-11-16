// Fibo 2.0 pipeline: a structured caption (or a short description) → one image; with images, an instruction edit, a
// masked edit, or a multi-reference edit. Isomorphic to diffusers 0.43 `BriaFibo2Pipeline.__call__` for one prompt.
//
// Precision follows the reference at `dtype=torch.bfloat16`: Qwen3-VL and the DiT run in bf16; the latents stay fp32
// across the Euler steps, with the guidance combine and the update in fp32; the VAE runs in fp32 (the reference runs
// it in bf16; fp32 is the fleet's FLUX.2 VAE lane), in row bands by default (flux2-vae-mlx-swift).
//
// One saving over the reference, with the same result: the Perceiver's gist tokens depend on the prompt only, so
// they are computed once per prompt, not at every step.

import Flux2VAE
import Foundation
import MLX
import MLXRandom

public final class Fibo2Pipeline {
    public let transformer: Fibo2Transformer
    public private(set) var textEncoder: Fibo2TextEncoder?
    public let vaeEncoder: Flux2VAEEncoder
    public let vae: Flux2VAE
    public let textRenderer: Fibo2TextRenderer
    public let snapshot: URL
    /// Row-band size for the fp32 VAE decode and the edit-image encode; nil runs each untiled.
    public var vaeTile: Flux2VAETile? = .standard

    public init(transformer: Fibo2Transformer, textEncoder: Fibo2TextEncoder?, vaeEncoder: Flux2VAEEncoder,
                vae: Flux2VAE, textRenderer: Fibo2TextRenderer, snapshot: URL) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.vaeEncoder = vaeEncoder
        self.vae = vae
        self.textRenderer = textRenderer
        self.snapshot = snapshot
    }

    /// Load from a snapshot of briaai/Fibo-2.0-dev or briaai/Fibo-2.0-edit-turbo: `transformer/`, `text_encoder/`,
    /// `processor/`, `vae/`, `fonts/`.
    /// - `shared`: where `text_encoder/`, `processor/`, `vae/` and `fonts/` are read from, when it isn't `snapshot`. The
    ///   two checkpoints ship the same bytes for them (sha256 compared 2026-11-15), so an Edit Turbo `transformer/` can
    ///   run on a Dev snapshot's.
    /// - `textEncoderQuant: nil` loads no text encoder: pass one to `encode(_:encoder:)` per request.
    public static func load(snapshot: URL, shared: URL? = nil, dtype: DType = .bfloat16, quant: Fibo2Quant = .bf16,
                            textEncoderQuant: Fibo2TextEncoderQuant? = .bf16) async throws -> Fibo2Pipeline {
        let shared = shared ?? snapshot
        let transformer = try Fibo2Weights.loadTransformer(snapshot: snapshot, dtype: dtype, quant: quant)
        var textEncoder: Fibo2TextEncoder? = nil
        if let textEncoderQuant {
            textEncoder = try await Fibo2TextEncoder.load(snapshot: shared, dtype: dtype, quant: textEncoderQuant)
        }
        let vaeDir = shared.appendingPathComponent("vae")
        return Fibo2Pipeline(
            transformer: transformer, textEncoder: textEncoder,
            vaeEncoder: try Flux2VAEWeights.loadEncoder(directory: vaeDir, dtype: .float32),
            vae: try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .float32),
            textRenderer: try Fibo2TextRenderer(fontsDirectory: shared.appendingPathComponent("fonts")),
            snapshot: snapshot)
    }

    public struct Request: Sendable {
        /// A structured JSON caption (cleaned and serialized as training did), or a short description. With `images`:
        /// the edit instruction, or a JSON object with an "edit_instruction" key and the caption of the result.
        public var prompt: String
        /// nil = the null caption Fibo 2.0 was trained with. Encoded as given.
        public var negativePrompt: String?
        /// The images to edit (1–5; the instruction can call them <image_1>, <image_2>, …). Empty = generate.
        public var images: [Fibo2RGBImage]
        /// A single-band mask the size of the one image to edit: ≥ 128 marks the region to regenerate.
        public var mask: [UInt8]?
        /// Output size, multiples of 16. Default 1024², or when editing the trained size for the first image's
        /// aspect ratio (about one megapixel).
        public var width: Int?
        public var height: Int?
        public var steps: Int
        /// Classifier-free guidance; ≤ 1 turns it off (one DiT pass per step).
        public var guidance: Float
        public var seed: UInt64
        /// Draw the caption's `text_render` texts and show them to Qwen3-VL, as training did.
        public var renderTextImages: Bool

        public init(prompt: String, negativePrompt: String? = nil, images: [Fibo2RGBImage] = [], mask: [UInt8]? = nil,
                    width: Int? = nil, height: Int? = nil, steps: Int = Fibo2Schedule.defaultSteps,
                    guidance: Float = Fibo2Schedule.defaultGuidance, seed: UInt64 = 0, renderTextImages: Bool = true) {
            self.prompt = prompt
            self.negativePrompt = negativePrompt
            self.images = images
            self.mask = mask
            self.width = width
            self.height = height
            self.steps = steps
            self.guidance = guidance
            self.seed = seed
            self.renderTextImages = renderTextImages
        }
    }

    /// A request's conditioning: everything the text encoder and the VAE encoder produce.
    public struct Encoded {
        /// [1, 6, L, 7680] bundles: the prompt (with every image), the base prompt a generation with text images runs
        /// its first steps on, the negative prompt (nil without guidance).
        public var prompt: MLXArray
        public var base: MLXArray?
        public var negative: MLXArray?
        /// Clean tokens of the images to edit, [M, 128] in all, and their grids.
        public var context: MLXArray?
        public var contextGrids: [Fibo2Transformer.Grid]
        public var width: Int
        public var height: Int
        /// The prompt as encoded (before the chat template), for reporting.
        public var shapedPrompt: String
    }

    public struct Result {
        public var image: Fibo2RGBImage
        public var seed: UInt64
        public var width: Int
        public var height: Int
        public var shapedPrompt: String
        public var timings: [String: Double]
    }

    /// Starting noise [1, 32, H/8, W/8], standard normal fp32. MLX's generator, not torch's: pass `noise` to
    /// `generate` to replay a reference run.
    public static func noise(seed: UInt64, width: Int, height: Int) -> MLXArray {
        MLXRandom.normal([1, 32, height / 8, width / 8], key: MLXRandom.key(seed))
    }

    /// Latents [1, 32, 2h, 2w] → tokens [h·w, 128], row-major, features ordered (channel, dy, dx).
    public static func tokens(fromLatents latents: MLXArray) -> MLXArray {
        let (c, h, w) = (latents.dim(1), latents.dim(2) / 2, latents.dim(3) / 2)
        return latents.reshaped(c, h, 2, w, 2).transposed(1, 3, 0, 2, 4).reshaped(h * w, c * 4)
    }

    /// Packed latents [1, 128, h, w] ↔ tokens [h·w, 128].
    public static func tokens(fromPacked packed: MLXArray) -> MLXArray {
        packed[0].transposed(1, 2, 0).reshaped(packed.dim(2) * packed.dim(3), packed.dim(1))
    }

    public static func packed(fromTokens tokens: MLXArray, rows: Int, columns: Int) -> MLXArray {
        tokens.reshaped(rows, columns, tokens.dim(1)).transposed(2, 0, 1)[.newAxis]
    }

    /// Drop the text encoder (~8.9 GB in bf16) after the last prompt has been encoded.
    public func releaseTextEncoder() { textEncoder = nil }

    public static func validate(_ request: Request) throws {
        for side in [request.width, request.height] {
            if let side, side % 16 != 0 || side < 64 {
                throw Fibo2Error.invalidInput("width and height have to be multiples of 16, at least 64 (got \(side))")
            }
        }
        guard request.steps >= 1 else { throw Fibo2Error.invalidInput("steps has to be at least 1") }
        guard request.images.count <= Fibo2ImagePrep.maxImages else {
            throw Fibo2Error.invalidInput("Fibo 2.0 edits 1 to \(Fibo2ImagePrep.maxImages) images (got \(request.images.count))")
        }
        if !request.images.isEmpty, let problem = Fibo2Prompt.editPromptProblem(request.prompt) {
            throw Fibo2Error.invalidInput(problem)
        }
        if let mask = request.mask {
            guard request.images.count == 1 else { throw Fibo2Error.invalidInput("a mask needs exactly one image") }
            guard mask.count == request.images[0].width * request.images[0].height else {
                throw Fibo2Error.invalidInput("the mask has to be the image's size")
            }
        }
    }

    /// Steps 1–4 of `__call__`: shape the prompt, prepare the images, run Qwen3-VL and the VAE encoder.
    /// `encoder`: a text encoder held outside the pipeline (loaded per request and dropped before denoising).
    public func encode(_ request: Request, encoder: Fibo2TextEncoder? = nil) throws -> Encoded {
        try Self.validate(request)
        guard let textEncoder = encoder ?? self.textEncoder else {
            throw Fibo2Error.invalidInput("no text encoder: it was released and none was given")
        }
        let shaped = Fibo2Prompt.shape(request.prompt, imageCount: request.images.count,
                                       renderTextImages: request.renderTextImages)
        let textImages = shaped.renderTexts.map(textRenderer.draw)

        var width = request.width ?? 1024, height = request.height ?? 1024
        var context: MLXArray? = nil
        var grids: [Fibo2Transformer.Grid] = []
        var encoderImages: [Fibo2RGBImage] = []
        if !request.images.isEmpty {
            let prepared = Fibo2ImagePrep.prepare(images: request.images, mask: request.mask, width: request.width,
                                                  height: request.height)
            (width, height) = (prepared.width, prepared.height)
            encoderImages = prepared.textEncoderImages
            var tokens: [MLXArray] = []
            for crop in prepared.crops {
                let pixels = crop.pixelsNCHW()
                let packed = vaeTile.map { vaeEncoder.encodePacked(pixels, tile: $0) } ?? vaeEncoder.encodePacked(pixels)
                let t = Self.tokens(fromPacked: packed).asType(transformer.dtype)
                eval(t)  // one encoder transient at a time
                tokens.append(t)
                grids.append(.init(rows: packed.dim(2), columns: packed.dim(3)))
            }
            context = concatenated(tokens, axis: 0)
        }

        let prompt = try textEncoder.encode(prompt: shaped.prompt, images: encoderImages + textImages)
        eval(prompt)
        var base: MLXArray? = nil
        if let basePrompt = shaped.basePrompt {
            base = try textEncoder.encode(prompt: basePrompt)
            eval(base!)
        }
        var negative: MLXArray? = nil
        if request.guidance > 1 {
            negative = try textEncoder.encode(prompt: request.negativePrompt ?? Fibo2Prompt.defaultNegative)
            eval(negative!)
        }
        return Encoded(prompt: prompt, base: base, negative: negative, context: context, contextGrids: grids,
                       width: width, height: height, shapedPrompt: shaped.prompt)
    }

    /// `noise`: [1, 32, H/8, W/8] to replay a reference run. `progress(step, steps)` runs after each step; throwing
    /// from it cancels.
    public func generate(_ request: Request, encoded: Encoded? = nil, noise providedNoise: MLXArray? = nil,
                         progress: ((Int, Int) throws -> Void)? = nil) throws -> Result {
        var timings: [String: Double] = [:]
        var clock = Date()
        func lap(_ name: String) { timings[name] = Date().timeIntervalSince(clock); clock = Date() }

        let encoded = try encoded ?? encode(request)
        lap("encode")
        let (width, height) = (encoded.width, encoded.height)
        let grid = Fibo2Transformer.Grid(rows: height / 16, columns: width / 16)
        let latents = providedNoise?.asType(.float32) ?? Self.noise(seed: request.seed, width: width, height: height)
        guard latents.shape == [1, 32, height / 8, width / 8] else {
            throw Fibo2Error.invalidInput("noise shape \(latents.shape) ≠ [1, 32, \(height / 8), \(width / 8)]")
        }
        var x = Self.tokens(fromLatents: latents)

        // Gist tokens and token geometry, once per prompt.
        func prepare(_ bundles: MLXArray?) -> (Fibo2Transformer.Conditioning, Fibo2Transformer.Layout)? {
            guard let bundles else { return nil }
            let c = transformer.condition(bundles)
            eval(c.gist)
            return (c, transformer.layout(image: grid, context: encoded.contextGrids, gistCount: c.gistCount))
        }
        let prompt = prepare(encoded.prompt)!
        let base = prepare(encoded.base)
        let negative = request.guidance > 1 ? prepare(encoded.negative) : nil
        lap("gist")

        let sigmas = Fibo2Schedule.sigmas(steps: request.steps, imageTokens: grid.count)
        for step in 0..<request.steps {
            let level = Fibo2Schedule.noiseLevel(sigmas[step])
            let cond = (base != nil && level > Fibo2Schedule.textImagesMaxNoiseLevel) ? base! : prompt
            var velocity = transformer(image: x, timestep: level, conditioning: cond.0, context: encoded.context,
                                       layout: cond.1).asType(.float32)
            if let negative {
                let uncond = transformer(image: x, timestep: level, conditioning: negative.0, context: encoded.context,
                                         layout: negative.1).asType(.float32)
                velocity = uncond + request.guidance * (velocity - uncond)
            }
            x = Fibo2Schedule.step(x, velocity: velocity, sigma: sigmas[step], next: sigmas[step + 1])
            eval(x)
            try progress?(step + 1, request.steps)
        }
        lap("denoise")

        let packed = Self.packed(fromTokens: x, rows: grid.rows, columns: grid.columns)
        let decoded = vaeTile.map { vae.decodePackedLatents(packed, tile: $0) } ?? vae.decodePackedLatents(packed)
        let image = Fibo2RGBImage.fromPixels(decoded[0])
        lap("decode")
        return Result(image: image, seed: request.seed, width: width, height: height,
                      shapedPrompt: encoded.shapedPrompt, timings: timings)
    }
}
