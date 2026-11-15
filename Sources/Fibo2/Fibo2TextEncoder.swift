// Qwen3-VL-4B conditioner for Fibo 2.0 — diffusers' `_get_qwen_prompt_embeds` on the fleet's qwen3vl-mlx-swift
// backbone. The snapshot's `text_encoder/` is Qwen3-VL-4B-Instruct with its vision tower; `processor/` holds the
// tokenizer.
//
// Reference facts (pinned by the oracle's prompt and encoder goldens):
//   - the prompt goes in one user turn with no system turn:
//     "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", truncated at 4096 tokens, never padded here
//     (one prompt per forward);
//   - images (the ones to edit, then the images of the caption's text) sit at their `<|image_pad|>` markers, each
//     expanded to one token per 32-px cell; the processor resizes with min 3136 / max 1 605 632 pixels (BICUBIC) and
//     normalises with mean = std = 0.5;
//   - the full Qwen3-VL forward: vision features merged in, deepstack added after the first three layers, spatial
//     M-RoPE positions;
//   - six bundles, each three `hidden_states[k]` concatenated per token (→ 7680): the Perceiver's (9, 20, 31), then
//     the injection blocks' (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35). Each layer is scaled
//     to unit RMS per token in fp32 (eps 1e-6 inside the root) before the concat.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Qwen3VL
import Tokenizers

/// Text-encoder weight tier.
public enum Fibo2TextEncoderQuant: String, Codable, Sendable, CaseIterable {
    case bf16, int8

    public var bits: Int? { self == .int8 ? 8 : nil }
}

/// Fibo 2.0's `Qwen3VLProcessor`: the chat template, the tokenizer, and the image processor that turns the images of
/// a prompt into patches and expands their `<|image_pad|>` markers.
public struct Fibo2Processor {
    public let tokenizer: any Tokenizers.Tokenizer
    public let imageProcessor: Qwen3VLImageProcessor
    public let imagePadId: Int

    public static let maxLength = 4096
    /// processor_config.json `size`: shortest_edge / longest_edge are the pixel bounds.
    public static let minPixels = 3136, maxPixels = 1_605_632

    public init(tokenizer: any Tokenizers.Tokenizer) throws {
        self.tokenizer = tokenizer
        self.imageProcessor = Qwen3VLImageProcessor(minPixels: Self.minPixels, maxPixels: Self.maxPixels)
        guard let pad = tokenizer.convertTokenToId("<|image_pad|>") else {
            throw Fibo2Error.loading("the tokenizer lacks <|image_pad|>")
        }
        self.imagePadId = pad
    }

    /// The tokenizer in a snapshot's `processor/`.
    public static func load(snapshot: URL) async throws -> Fibo2Processor {
        try Fibo2Processor(tokenizer: try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("processor")))
    }

    public struct Inputs {
        public var ids: [Int32]
        public var pixelValues: MLXArray?
        public var grids: [THW]

        public init(ids: [Int32], pixelValues: MLXArray?, grids: [THW]) {
            self.ids = ids
            self.pixelValues = pixelValues
            self.grids = grids
        }
    }

    /// Template, tokenize, expand the k-th `<|image_pad|>` to the k-th image's token count (one per 32-px cell).
    public func callAsFunction(prompt: String, images: [Fibo2RGBImage] = []) throws -> Inputs {
        var pixels: [MLXArray] = []
        var grids: [THW] = []
        for image in images {
            let (pv, grid) = imageProcessor.preprocess(rgb: image.rgb, width: image.width, height: image.height)
            pixels.append(pv)
            grids.append(grid)
        }
        let merge = imageProcessor.mergeSize * imageProcessor.mergeSize
        let raw = tokenizer.encode(text: Fibo2Prompt.template(prompt), addSpecialTokens: false)
        var ids: [Int32] = []
        var k = 0
        for id in raw {
            guard id == imagePadId else { ids.append(Int32(id)); continue }
            guard k < grids.count else {
                throw Fibo2Error.invalidInput("the prompt holds more <|image_pad|> markers than images (\(grids.count))")
            }
            ids.append(contentsOf: [Int32](repeating: Int32(imagePadId), count: grids[k].product / merge))
            k += 1
        }
        guard k == grids.count else {
            throw Fibo2Error.invalidInput("the prompt holds \(k) <|image_pad|> markers for \(grids.count) images")
        }
        if ids.count > Self.maxLength {
            guard images.isEmpty else {
                throw Fibo2Error.invalidInput(
                    "the prompt and its images take \(ids.count) tokens; Fibo 2.0 reads at most \(Self.maxLength)")
            }
            ids = Array(ids.prefix(Self.maxLength))
        }
        return Inputs(ids: ids, pixelValues: pixels.isEmpty ? nil : concatenated(pixels, axis: 0), grids: grids)
    }
}

public final class Fibo2TextEncoder {
    public let model: Qwen3VL
    public let processor: Fibo2Processor

    public static let outputLayers = [[9, 20, 31], [5, 16, 27], [7, 18, 29], [9, 20, 31], [11, 22, 33], [13, 24, 35]]

    public init(model: Qwen3VL, processor: Fibo2Processor) {
        self.model = model
        self.processor = processor
    }

    /// Load `text_encoder/` and the tokenizer in `processor/` from a Fibo 2.0 snapshot. `quant: .int8` quantizes the
    /// language model's and the vision tower's Linears after load (affine, group 64); the embedding table stays at
    /// `dtype`.
    public static func load(snapshot: URL, dtype: DType = .bfloat16, quant: Fibo2TextEncoderQuant = .bf16,
                            groupSize: Int = 64) async throws -> Fibo2TextEncoder {
        let model = try Qwen3VLLoader.load(directory: snapshot.appendingPathComponent("text_encoder"), dtype: dtype)
        if let bits = quant.bits {
            quantize(model: model, filter: { _, module in module is Linear ? (groupSize, bits, .affine) : nil })
            eval(model)
        }
        return Fibo2TextEncoder(model: model, processor: try await Fibo2Processor.load(snapshot: snapshot))
    }

    /// The six bundles [1, 6, L, 7680] in the encoder's dtype.
    public func bundles(_ inputs: Fibo2Processor.Inputs) throws -> MLXArray {
        let layers = Array(Set(Self.outputLayers.flatMap { $0 })).sorted()
        let ids = MLXArray(inputs.ids).reshaped(1, inputs.ids.count)
        let states = try model.hiddenStates(inputIds: ids, layers: layers, pixelValues: inputs.pixelValues,
                                            imageGridTHW: inputs.grids.isEmpty ? nil : inputs.grids)
        var normalized: [Int: MLXArray] = [:]
        for (layer, state) in zip(layers, states) {
            let s = state.asType(.float32)
            normalized[layer] = (s * rsqrt(mean(s * s, axis: -1, keepDims: true) + 1e-6)).asType(state.dtype)
        }
        let bundles = Self.outputLayers.map { taps in concatenated(taps.map { normalized[$0]! }, axis: -1) }
        return stacked(bundles, axis: 1)
    }

    public func encode(prompt: String, images: [Fibo2RGBImage] = []) throws -> MLXArray {
        try bundles(processor(prompt: prompt, images: images))
    }
}
