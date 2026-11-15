// MLXEngine package for Bria's Fibo 2.0 Dev: `textToImage` + `imageEdit` on one core (the model is a unified
// generator / editor; `run(_:)` dispatches on the request).
//
// LICENCE (C7): the weights are CC BY-NC 4.0 (`license_name: bria-fibo`), behind Bria's non-commercial access form;
// commercial use needs an agreement with Bria. It is declared as the SPDX id `CC-BY-NC-4.0`, package-locally, and is
// NOT allowlisted, so a `.permissiveOnly` engine refuses or flags this package by construction. It is a Gaea plugin
// model (AB-D-0226), named-only and opt-in, never a shipping route. Bria states the training data is licensed.
// C8: the port code is MIT.

import Fibo2
import Flux2VAE
import Foundation
import MLX
import MLXToolKit

extension SPDXLicense {
    /// Creative Commons Attribution-NonCommercial 4.0. Intentionally absent from `permissiveAllowlist`.
    public static let ccByNC4: SPDXLicense = "CC-BY-NC-4.0"
}

/// `description` is the readable message too: engines that report `"\(error)"` show the sentence, not the enum case.
public enum Fibo2PackageError: Error, LocalizedError, CustomStringConvertible {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    case unsupportedQuant(Quant)
    /// The request would run outside the envelope the declared footprint holds for. Refused rather than clamped:
    /// silently shrinking a size changes the output.
    case outsideMeasuredEnvelope(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p):
            return "Fibo 2.0 Dev snapshot not readable at \(p). The repo is gated: the Hugging Face account behind the "
                + "token needs access to briaai/Fibo-2.0-dev."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        case .unsupportedQuant(let q): return "Fibo 2.0 Dev has no \(q) tier (bf16, int8 and int4 are declared)."
        case .outsideMeasuredEnvelope(let why): return why
        }
    }

    public var description: String { errorDescription ?? "Fibo 2.0 Dev error" }
}

/// The input envelope the manifest's footprint holds for (Fibo2Footprints). `run()` refuses anything outside it, so
/// the declaration stays an upper bound (AB-L-0202).
public enum Fibo2Envelope {
    public static let maxOutputPixels = Fibo2Footprints.envelopeMaxPixels
    public static let minOutputPixels = Fibo2Footprints.envelopeMinPixels
    public static let minSide = 64
    public static let maxImages = Fibo2Footprints.envelopeMaxImages

    /// nil when inside the envelope, else a user-facing reason. nil sizes: 1024², or when editing the trained size for
    /// the first image's aspect ratio (always inside the band).
    public static func violation(width: Int?, height: Int?, imageCount: Int) -> String? {
        if imageCount > maxImages {
            return "Fibo 2.0 Dev edits at most \(maxImages) images at once; got \(imageCount)."
        }
        guard let w = width, let h = height else {
            return (width == nil) == (height == nil) ? nil : "Fibo 2.0 Dev needs both width and height, or neither."
        }
        if w % 16 != 0 || h % 16 != 0 { return "Fibo 2.0 Dev sizes are multiples of 16 (got \(w)×\(h))." }
        if w < minSide || h < minSide { return "Fibo 2.0 Dev sides are at least \(minSide) px (got \(w)×\(h))." }
        if w * h > maxOutputPixels {
            return "Fibo 2.0 Dev: \(w)×\(h) is past the area the model was trained at (about one megapixel, at most "
                + "1.1 × 1024²), e.g. 1024×1024, 1424×800 or 1248×832."
        }
        if w * h < minOutputPixels { return "Fibo 2.0 Dev: \(w)×\(h) is under 256² of area." }
        return nil
    }
}

/// Package-specific `metaData` levers (contract C5).
public enum Fibo2MetaKeys {
    /// String: a base64 PNG or JPEG the size of the single image to edit. White marks the region to regenerate; the
    /// model sees it greyed out, as in training. (`IEditRequest` has no mask field.)
    public static let mask = "mask"
    /// Bool, default true: draw each `text_render` text of the caption and show it to Qwen3-VL, as training did.
    public static let textImages = "textImages"

    static func mask(_ meta: MetaData) throws -> Data? {
        switch meta[mask] {
        case nil, .null?: return nil
        case .string(let s)?:
            guard let data = Data(base64Encoded: s, options: .ignoreUnknownCharacters), !data.isEmpty else {
                throw PackageError.unsupportedRequestFeature("metaData[\"\(mask)\"] is a base64 PNG or JPEG")
            }
            return data
        default: throw PackageError.unsupportedRequestFeature("metaData[\"\(mask)\"] is a base64 PNG or JPEG")
        }
    }

    static func textImages(_ meta: MetaData) throws -> Bool {
        switch meta[textImages] {
        case nil, .null?: return true
        case .bool(let b)?: return b
        default: throw PackageError.unsupportedRequestFeature("metaData[\"\(textImages)\"] must be a boolean")
        }
    }
}

/// The DiT and VAE stay resident. The conditioner loads per request and is released before denoising, unless
/// `keepEncoderResident` is set. When it is resident, the denoise steps run with it in memory: the resident hint
/// covers its weights.
@InferenceActor
final class Fibo2Generator {
    let pipeline: Fibo2Pipeline
    let snapshot: URL
    let keepEncoderResident: Bool
    let textEncoderQuant: Fibo2TextEncoderQuant
    private var residentEncoder: Fibo2TextEncoder?

    init(pipeline: Fibo2Pipeline, snapshot: URL, keepEncoderResident: Bool, textEncoderQuant: Fibo2TextEncoderQuant) {
        self.pipeline = pipeline
        self.snapshot = snapshot
        self.keepEncoderResident = keepEncoderResident
        self.textEncoderQuant = textEncoderQuant
    }

    static func load(configuration c: Fibo2Configuration, snapshot: URL) async throws -> Fibo2Generator {
        let quant: Fibo2Quant
        switch c.quant {
        case .bf16: quant = .bf16
        case .int8: quant = .int8
        case .int4: quant = .int4
        default: throw Fibo2PackageError.unsupportedQuant(c.quant)
        }
        let pipeline = try await Fibo2Pipeline.load(snapshot: snapshot, quant: quant, textEncoderQuant: nil)
        MLX.Memory.clearCache()
        return Fibo2Generator(pipeline: pipeline, snapshot: snapshot, keepEncoderResident: c.keepEncoderResident,
                              textEncoderQuant: c.textEncoderQuant)
    }

    /// The request's conditioning. A per-request encoder is created and dropped inside this scope, so it never lives
    /// into the denoise loop.
    private func encode(_ request: Fibo2Pipeline.Request) async throws -> Fibo2Pipeline.Encoded {
        if residentEncoder == nil && keepEncoderResident {
            residentEncoder = try await Fibo2TextEncoder.load(snapshot: snapshot, quant: textEncoderQuant)
        }
        let encoder: Fibo2TextEncoder
        if let residentEncoder {
            encoder = residentEncoder
        } else {
            encoder = try await Fibo2TextEncoder.load(snapshot: snapshot, quant: textEncoderQuant)
        }
        return try pipeline.encode(request, encoder: encoder)
    }

    func generate(_ request: Fibo2Pipeline.Request) async throws -> Fibo2Pipeline.Result {
        RunProgress.report(.encode)
        let encoded = try await encode(request)
        MLX.Memory.clearCache()
        try Task.checkCancellation()  // post-encode seam
        let result = try pipeline.generate(request, encoded: encoded) { step, total in
            RunProgress.report(.denoise, step: step, totalSteps: total)
            try Task.checkCancellation()  // between denoise steps
        }
        MLX.Memory.clearCache()
        return result
    }
}

@InferenceActor
enum Fibo2Serving {
    static func load(_ c: Fibo2Configuration) async throws -> Fibo2Generator {
        guard let snapshot = c.resolvedSnapshotDirectory(storeRoot: c.modelsRootDirectory), Fibo2Configuration.complete(snapshot)
        else { throw Fibo2PackageError.unreadableSnapshot(c.snapshotPath.isEmpty ? Fibo2Configuration.repo : c.snapshotPath) }
        return try await Fibo2Generator.load(configuration: c, snapshot: snapshot)
    }

    static func run(_ request: any CapabilityRequest, generator: Fibo2Generator,
                    configuration c: Fibo2Configuration) async throws -> any CapabilityResponse {
        var req = Fibo2Pipeline.Request(prompt: "")
        let steps: Int?, guidance: Double?, seed: UInt64?
        let meta: MetaData
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
            req.prompt = t2i.prompt
            // The model card's text-to-image negative, unless the caller sends one. Edits default to the null caption.
            req.negativePrompt = t2i.negativePrompt ?? Fibo2Prompt.cardNegativeT2I
            (req.width, req.height) = (t2i.width ?? 1024, t2i.height ?? 1024)
            (steps, guidance, seed, meta) = (t2i.steps, t2i.guidanceScale, t2i.seed, t2i.metaData)
        case .imageEdit:
            guard let edit = request as? IEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            guard !edit.images.isEmpty else { throw Fibo2PackageError.imageDecode }
            req.prompt = edit.prompt
            req.negativePrompt = edit.negativePrompt
            req.images = try edit.images.map { image in
                do { return try Fibo2RGBImage.read(data: image.data) } catch { throw Fibo2PackageError.imageDecode }
            }
            (req.width, req.height) = (edit.width, edit.height)
            (steps, guidance, seed, meta) = (edit.steps, edit.guidanceScale, edit.seed, edit.metaData)
            if let maskData = try Fibo2MetaKeys.mask(meta) {
                guard let mask = try? Fibo2RGBImage.read(data: maskData) else { throw Fibo2PackageError.imageDecode }
                guard req.images.count == 1, mask.width == req.images[0].width, mask.height == req.images[0].height else {
                    throw PackageError.unsupportedRequestFeature(
                        "metaData[\"mask\"] needs exactly one image, and has to be its size")
                }
                req.mask = Fibo2ImagePrep.luma(mask)
            }
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
        if let why = Fibo2Envelope.violation(width: req.width, height: req.height, imageCount: req.images.count) {
            throw Fibo2PackageError.outsideMeasuredEnvelope(why)
        }
        req.steps = steps ?? c.defaultSteps
        req.guidance = Float(guidance ?? c.defaultGuidance)
        req.seed = seed ?? UInt64.random(in: 0..<(1 << 32))
        req.renderTextImages = try Fibo2MetaKeys.textImages(meta)
        guard (1...200).contains(req.steps) else {
            throw PackageError.unsupportedRequestFeature("steps is 1–200 (got \(req.steps))")
        }
        do { try Fibo2Pipeline.validate(req) } catch let error as Fibo2Error {
            throw PackageError.unsupportedRequestFeature(error.description)
        }
        try Task.checkCancellation()
        let result = try await generator.generate(req)
        try Task.checkCancellation()
        let png: Data
        do { png = try result.image.pngData() } catch { throw Fibo2PackageError.pngEncode }
        let artifact = Image(format: .png, data: png, width: result.width, height: result.height)
        return request.capability == .textToImage ? T2IResponse(image: artifact) : IEditResponse(image: artifact)
    }

    /// What a caller (or an agent writing the prompt) must know: the model reads structured captions.
    nonisolated static let promptGuidance =
        "Fibo 2.0 was trained on STRUCTURED JSON CAPTIONS (Bria's VGL, docs.bria.ai/vgl): a JSON object with "
        + "short_description, objects [{description, location, shape_and_color, texture, …}], background_setting, "
        + "lighting {…}, aesthetics {…}, photographic_characteristics {…}, style_medium, artistic_style, and "
        + "text_render [{text, location, size, color, font}] for text to draw. Send that JSON as the prompt for the "
        + "control the model is known for; a short plain description also works, with less control. "

    nonisolated static let licenceNote =
        "CC BY-NC 4.0 WEIGHTS: non-commercial use only; commercial use needs an agreement with Bria."

    nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .ccByNC4, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: Fibo2Configuration.repo, revision: Fibo2Configuration.revision, tier: 1),
            requirements: RequirementsManifest(
                footprints: Fibo2Footprints.declared,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "fibo-2-dev",
                    summary: "Fibo 2.0 Dev text-to-image (Bria, 8.1B single-stream DiT on Qwen3-VL-4B, FLUX.2 VAE): the "
                        + "full base model, 50 steps with classifier-free guidance 5, about one megapixel (1024×1024, "
                        + "1424×800, 1248×832, …). Two DiT passes per step: about 3.5 s a step on an M5 Max. "
                        + promptGuidance
                        + "negativePrompt defaults to the model card's text-to-image negative. " + licenceNote,
                    modes: []),
                IEditContract.descriptor(
                    name: "fibo-2-dev",
                    summary: "Fibo 2.0 Dev instruction editing: the prompt is the edit instruction (\"replace the jacket "
                        + "with a red leather one\"), or a JSON object with \"edit_instruction\" and the caption of the "
                        + "result under \"structured_caption\". Several images are named in order as <image_1>, "
                        + "<image_2>. The result is drawn from noise at the size trained for the first image's aspect "
                        + "ratio (about one megapixel), which crops it to that ratio. metaData[\"mask\"] (a base64 PNG "
                        + "the image's size, white = the region to regenerate) confines the edit of a single image. "
                        + "Up to 5 images. Slower than text-to-image: every image adds its tokens to every step "
                        + "(about 2× the time with one image, 4× with two, 11× with five). " + licenceNote,
                    modes: []),
            ]
        )
    }
}

@InferenceActor
public final class Fibo2DevPackage: ModelPackage {
    public typealias Configuration = Fibo2Configuration

    public nonisolated static var manifest: PackageManifest { Fibo2Serving.manifest }

    let configuration: Configuration
    private var generator: Fibo2Generator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        generator = try await Fibo2Serving.load(configuration)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()  // CAN-1: first act of run()
        guard let generator else { throw PackageError.notLoaded }
        return try await Fibo2Serving.run(request, generator: generator, configuration: configuration)
    }
}

extension Fibo2DevPackage {
    public nonisolated static var registration: PackageRegistration { .of(Fibo2DevPackage.self) }
}
