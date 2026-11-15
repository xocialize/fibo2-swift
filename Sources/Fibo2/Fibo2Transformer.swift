// BriaFibo2Transformer2DModel — isomorphic port of diffusers 0.43 `transformer_bria_fibo2.py`, for one sample
// (batch 1, no padding, so no attention masks): an 8.1B single-stream DiT of Z-Image blocks.
//
//   - tokens: 2×2 latent patches, i.e. the FLUX.2 VAE's packed 128-channel latents, one token per /16 cell
//     (feature order (channel, dy, dx) is the packed channel order, so no re-patchify is needed);
//   - 2 noise-refiner blocks over the image tokens (and the clean context tokens of an edit);
//   - a 2-layer Perceiver resampler squeezes the first text bundle into 256–512 "gist" tokens (more for a longer
//     prompt), which are PREPENDED: [gist | image | context…];
//   - 44 main blocks; blocks 6/14/22/30/38 also cross-attend to their own text bundle through a tanh-gated branch;
//   - per-token adaLN: image tokens get the real timestep, clean tokens (gist, context) get t = 0;
//   - 3-axis RoPE (32, 48, 48), θ 256: gist token i at (1 + i, 0, 0), the image on plane 513, context image k on
//     plane 514 + k with its own rows and columns.
//
// Deviations forced by the frameworks (mirrored in the loader's key remap):
//   - `t_embedder.mlp` is Sequential(Linear, SiLU, Linear) → `mlp_in` / `mlp_out`;
//   - `final_layer.adaLN_modulation` is Sequential(SiLU, Linear) → `adaLN_linear`;
//   - the five `layers.N.cross_attn_gate` scalars are stacked into one `cross_attn_gates` array on the model, so a
//     plain block carries no empty parameter;
//   - complex `freqs_cis` → real (cos, sin) tables in fp32, built with Double math on the host.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum Fibo2Error: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)
    case cancelled

    public var description: String {
        switch self {
        case .loading(let m): return "Fibo2 loading error: \(m)"
        case .invalidInput(let m): return "Fibo2 input error: \(m)"
        case .cancelled: return "Fibo2: cancelled"
        }
    }
}

/// `transformer/config.json`.
public struct Fibo2DiTConfig: Sendable, Codable {
    public var inChannels = 32
    public var patchSize = 2
    public var dim = 3584
    public var nLayers = 44
    public var nRefinerLayers = 2
    public var nHeads = 28
    public var normEps: Float = 1e-5
    public var capFeatDim = 7680
    public var injectionLayerIds = [6, 14, 22, 30, 38]
    public var perceiverNumLayers = 2
    public var minNumGistTokens = 256
    public var maxNumGistTokens = 512
    public var gistStep = 64
    public var gistMinTextLen = 256
    public var gistMaxTextLen = 2048
    public var ropeTheta: Double = 256
    public var tScale: Float = 1000
    public var axesDims = [32, 48, 48]
    public var axesLens = [2048, 512, 512]

    public init() {}

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels", patchSize = "patch_size", dim, nLayers = "n_layers"
        case nRefinerLayers = "n_refiner_layers", nHeads = "n_heads", normEps = "norm_eps"
        case capFeatDim = "cap_feat_dim", injectionLayerIds = "injection_layer_ids"
        case perceiverNumLayers = "perceiver_num_layers", minNumGistTokens = "min_num_gist_tokens"
        case maxNumGistTokens = "max_num_gist_tokens", gistStep = "gist_step"
        case gistMinTextLen = "gist_min_text_len", gistMaxTextLen = "gist_max_text_len"
        case ropeTheta = "rope_theta", tScale = "t_scale", axesDims = "axes_dims", axesLens = "axes_lens"
    }

    public var tokenChannels: Int { inChannels * patchSize * patchSize }
    public var ffnHidden: Int { (dim * 8 / 3) / 64 * 64 }
    static let adalnEmbedDim = 256

    /// `compute_gist_count`: text lengths between `gistMinTextLen` and `gistMaxTextLen` map linearly onto
    /// [`minNumGistTokens`, `maxNumGistTokens`], rounded up to a multiple of `gistStep`.
    public func gistCount(textLength: Int) -> Int {
        let span = Double(gistMaxTextLen - gistMinTextLen)
        let ratio = min(1.0, max(0.0, (Double(textLength) - Double(gistMinTextLen)) / span))
        let raw = Double(minNumGistTokens) + ratio * Double(maxNumGistTokens - minNumGistTokens)
        let quantized = Int((raw / Double(gistStep)).rounded(.up)) * gistStep
        return max(minNumGistTokens, min(maxNumGistTokens, quantized))
    }
}

final class Fibo2TimestepEmbedder: Module {
    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear
    let frequencyEmbeddingSize: Int

    init(outSize: Int, midSize: Int, frequencyEmbeddingSize: Int = 256) {
        self._mlpIn.wrappedValue = Linear(frequencyEmbeddingSize, midSize, bias: true)
        self._mlpOut.wrappedValue = Linear(midSize, outSize, bias: true)
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
    }

    /// t: [B] fp32 → [B, dim], fp32 sinusoid (cos | sin), max period 10000.
    static func timestepEmbedding(_ t: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = exp(-Float(Foundation.log(10000.0)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let args = t.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
        return concatenated([cos(args), sin(args)], axis: -1)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize).asType(mlpIn.weight.dtype)
        return mlpOut(silu(mlpIn(tFreq)))
    }
}

/// diffusers `RMSNorm` (elementwise affine, no bias): x · rsqrt(mean(x²) + eps) · weight, in the input's dtype.
final class Fibo2RMSNorm: Module {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(_ dim: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: weight, eps: eps) }
}

final class Fibo2FeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

/// RoPE tables for a token sequence: cos / sin, each [1, N, headDim / 2], fp32.
public struct Fibo2Rope {
    public var cos: MLXArray
    public var sin: MLXArray

    static func concatenated(_ parts: [Fibo2Rope]) -> Fibo2Rope {
        Fibo2Rope(cos: MLX.concatenated(parts.map(\.cos), axis: 1), sin: MLX.concatenated(parts.map(\.sin), axis: 1))
    }

    /// Rotate pairs in fp32, cast back: (x0 + i·x1)·(cos + i·sin). x: [1, N, heads, headDim].
    func apply(_ xIn: MLXArray) -> MLXArray {
        let shape = xIn.shape
        let x = xIn.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let x0 = x[0..., 0..., 0..., 0..., 0], x1 = x[0..., 0..., 0..., 0..., 1]
        let c = cos[0..., 0..., .newAxis, 0...], s = sin[0..., 0..., .newAxis, 0...]
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped(shape).asType(xIn.dtype)
    }
}

/// Plain class, not a Module: the tables must never become parameters.
final class Fibo2RopeEmbedder {
    let theta: Double
    let axesDims: [Int]
    let axesLens: [Int]
    private var tables: (cos: [MLXArray], sin: [MLXArray])?

    init(theta: Double, axesDims: [Int], axesLens: [Int]) {
        self.theta = theta
        self.axesDims = axesDims
        self.axesLens = axesLens
    }

    /// The reference computes the angles in float64, casts them to float32, then takes cos / sin (`torch.polar`).
    private func build() -> (cos: [MLXArray], sin: [MLXArray]) {
        var cosT: [MLXArray] = [], sinT: [MLXArray] = []
        for (d, e) in zip(axesDims, axesLens) {
            let half = d / 2
            var c = [Float](repeating: 0, count: e * half), s = [Float](repeating: 0, count: e * half)
            for pos in 0..<e {
                for j in 0..<half {
                    let angle = Float(Double(pos) * (1.0 / pow(theta, Double(2 * j) / Double(d))))
                    c[pos * half + j] = Foundation.cos(angle)
                    s[pos * half + j] = Foundation.sin(angle)
                }
            }
            cosT.append(MLXArray(c, [e, half]))
            sinT.append(MLXArray(s, [e, half]))
        }
        return (cosT, sinT)
    }

    /// ids: row-major (axis0, row, column) per token.
    func callAsFunction(_ ids: [Int32]) -> Fibo2Rope {
        if tables == nil { tables = build() }
        let n = ids.count / 3
        let all = MLXArray(ids, [n, 3])
        var cosParts: [MLXArray] = [], sinParts: [MLXArray] = []
        for i in 0..<axesDims.count {
            let index = all[0..., i]
            cosParts.append(tables!.cos[i][index])
            sinParts.append(tables!.sin[i][index])
        }
        return Fibo2Rope(cos: concatenated(cosParts, axis: -1)[.newAxis], sin: concatenated(sinParts, axis: -1)[.newAxis])
    }

    static func grid(size: (Int, Int, Int), start: (Int, Int, Int)) -> [Int32] {
        var ids = [Int32]()
        ids.reserveCapacity(size.0 * size.1 * size.2 * 3)
        for a in 0..<size.0 { for b in 0..<size.1 { for c in 0..<size.2 {
            ids.append(Int32(start.0 + a)); ids.append(Int32(start.1 + b)); ids.append(Int32(start.2 + c))
        } } }
        return ids
    }
}

/// Self-attention (with RoPE when given) or cross-attention to `context` (kv from `kvDim` features).
final class Fibo2Attention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: Fibo2RMSNorm
    @ModuleInfo(key: "norm_k") var normK: Fibo2RMSNorm

    init(dim: Int, heads: Int, kvDim: Int? = nil, eps: Float) {
        self.heads = heads
        self.headDim = dim / heads
        self._toQ.wrappedValue = Linear(dim, dim, bias: false)
        self._toK.wrappedValue = Linear(kvDim ?? dim, dim, bias: false)
        self._toV.wrappedValue = Linear(kvDim ?? dim, dim, bias: false)
        self._toOut.wrappedValue = [Linear(dim, dim, bias: false)]
        self._normQ.wrappedValue = Fibo2RMSNorm(headDim, eps: eps)
        self._normK.wrappedValue = Fibo2RMSNorm(headDim, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil, rope: Fibo2Rope? = nil) -> MLXArray {
        let source = context ?? x
        let (b, n, m) = (x.dim(0), x.dim(1), source.dim(1))
        var q = normQ(toQ(x).reshaped(b, n, heads, headDim))
        var k = normK(toK(source).reshaped(b, m, heads, headDim))
        let v = toV(source).reshaped(b, m, heads, headDim)
        if let rope {
            q = rope.apply(q)
            k = rope.apply(k)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1.0 / Foundation.sqrt(Float(headDim)), mask: nil)
        return toOut[0](out.transposed(0, 2, 1, 3).reshaped(b, n, heads * headDim))
    }
}

/// The four adaLN factors of a block, each broadcastable to [1, N, dim].
struct Fibo2Modulation {
    var scaleMSA: MLXArray
    var gateMSA: MLXArray
    var scaleMLP: MLXArray
    var gateMLP: MLXArray
}

/// Z-Image's block. With `crossAttentionDim` it is an injection block that also reads the text encoder.
final class Fibo2TransformerBlock: Module {
    @ModuleInfo var attention: Fibo2Attention
    @ModuleInfo(key: "feed_forward") var feedForward: Fibo2FeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: Fibo2RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: Fibo2RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: Fibo2RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: Fibo2RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Linear]
    @ModuleInfo(key: "cross_attn_norm1") var crossAttnNorm1: Fibo2RMSNorm?
    @ModuleInfo(key: "cross_attention") var crossAttention: Fibo2Attention?
    @ModuleInfo(key: "cross_attn_norm2") var crossAttnNorm2: Fibo2RMSNorm?

    init(_ c: Fibo2DiTConfig, crossAttentionDim: Int? = nil) {
        self._attention.wrappedValue = Fibo2Attention(dim: c.dim, heads: c.nHeads, eps: c.normEps)
        self._feedForward.wrappedValue = Fibo2FeedForward(dim: c.dim, hiddenDim: c.ffnHidden)
        self._attentionNorm1.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._ffnNorm1.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._attentionNorm2.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._ffnNorm2.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._adaLNModulation.wrappedValue = [Linear(min(c.dim, Fibo2DiTConfig.adalnEmbedDim), 4 * c.dim, bias: true)]
        if let crossAttentionDim {
            self._crossAttnNorm1.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
            self._crossAttention.wrappedValue = Fibo2Attention(dim: c.dim, heads: c.nHeads, kvDim: crossAttentionDim,
                                                               eps: c.normEps)
            self._crossAttnNorm2.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        } else {
            self._crossAttnNorm1.wrappedValue = nil
            self._crossAttention.wrappedValue = nil
            self._crossAttnNorm2.wrappedValue = nil
        }
    }

    /// The block's modulation of one adaLN input ([1, 256]): chunks (scale_msa, gate_msa, scale_mlp, gate_mlp), gates
    /// through tanh, scales 1 +. Each [1, 1, dim].
    func modulation(_ adaln: MLXArray) -> Fibo2Modulation {
        let parts = split(adaLNModulation[0](adaln)[0..., .newAxis, 0...], parts: 4, axis: 2)
        return Fibo2Modulation(scaleMSA: 1.0 + parts[0], gateMSA: tanh(parts[1]), scaleMLP: 1.0 + parts[2],
                               gateMLP: tanh(parts[3]))
    }

    /// `noisyMask`: [1, N, 1] bool, true on the noisy (image) tokens; nil = every token takes `adalnNoisy`.
    /// `text` and `gate` (the block's `cross_attn_gate`, before tanh): an injection block's text bundle.
    func callAsFunction(_ xIn: MLXArray, rope: Fibo2Rope, adalnNoisy: MLXArray, adalnClean: MLXArray,
                        noisyMask: MLXArray?, text: MLXArray? = nil, gate: MLXArray? = nil) -> MLXArray {
        var mod = modulation(adalnNoisy)
        if let noisyMask {
            let clean = modulation(adalnClean)
            mod = Fibo2Modulation(scaleMSA: MLX.where(noisyMask, mod.scaleMSA, clean.scaleMSA),
                                  gateMSA: MLX.where(noisyMask, mod.gateMSA, clean.gateMSA),
                                  scaleMLP: MLX.where(noisyMask, mod.scaleMLP, clean.scaleMLP),
                                  gateMLP: MLX.where(noisyMask, mod.gateMLP, clean.gateMLP))
        }
        var x = xIn
        let attnOut = attention(attentionNorm1(x) * mod.scaleMSA, rope: rope)
        x = x + mod.gateMSA * attentionNorm2(attnOut)
        if let crossAttention, let crossAttnNorm1, let crossAttnNorm2, let text, let gate {
            let crossOut = crossAttention(crossAttnNorm1(x), context: text)
            x = x + tanh(gate) * crossAttnNorm2(crossOut)
        }
        x = x + mod.gateMLP * ffnNorm2(feedForward(ffnNorm1(x) * mod.scaleMLP))
        return x
    }
}

/// Gist tokens read the text encoder (cross-attention), then each other (self-attention), then an FFN.
final class Fibo2PerceiverLayer: Module {
    @ModuleInfo(key: "cross_attn_norm") var crossAttnNorm: Fibo2RMSNorm
    @ModuleInfo(key: "cross_attention") var crossAttention: Fibo2Attention
    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: Fibo2RMSNorm
    @ModuleInfo(key: "self_attention") var selfAttention: Fibo2Attention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Fibo2RMSNorm
    @ModuleInfo var ffn: Fibo2FeedForward

    init(_ c: Fibo2DiTConfig) {
        self._crossAttnNorm.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._crossAttention.wrappedValue = Fibo2Attention(dim: c.dim, heads: c.nHeads, kvDim: c.capFeatDim, eps: c.normEps)
        self._selfAttnNorm.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._selfAttention.wrappedValue = Fibo2Attention(dim: c.dim, heads: c.nHeads, eps: c.normEps)
        self._ffnNorm.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
        self._ffn.wrappedValue = Fibo2FeedForward(dim: c.dim, hiddenDim: c.ffnHidden)
    }

    func callAsFunction(_ latentsIn: MLXArray, text: MLXArray) -> MLXArray {
        var latents = latentsIn + crossAttention(crossAttnNorm(latentsIn), context: text)
        latents = latents + selfAttention(selfAttnNorm(latents))
        return latents + ffn(ffnNorm(latents))
    }
}

/// Squeezes the text encoder's hidden states into `count` gist tokens.
final class Fibo2PerceiverResampler: Module {
    @ParameterInfo(key: "latent_queries") var latentQueries: MLXArray
    @ModuleInfo var layers: [Fibo2PerceiverLayer]
    @ModuleInfo(key: "norm_out") var normOut: Fibo2RMSNorm

    init(_ c: Fibo2DiTConfig) {
        self._latentQueries.wrappedValue = zeros([c.maxNumGistTokens, c.dim])
        self._layers.wrappedValue = (0..<c.perceiverNumLayers).map { _ in Fibo2PerceiverLayer(c) }
        self._normOut.wrappedValue = Fibo2RMSNorm(c.dim, eps: c.normEps)
    }

    /// text: [1, L, capFeatDim] → [1, count, dim].
    func callAsFunction(_ text: MLXArray, count: Int) -> MLXArray {
        var latents = latentQueries[..<count][.newAxis].asType(text.dtype)
        for layer in layers { latents = layer(latents, text: text) }
        return normOut(latents)
    }
}

final class Fibo2FinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "adaLN_linear") var adaLNLinear: Linear

    init(_ c: Fibo2DiTConfig) {
        self._normFinal.wrappedValue = LayerNorm(dimensions: c.dim, eps: 1e-6, affine: false)
        self._linear.wrappedValue = Linear(c.dim, c.tokenChannels, bias: true)
        self._adaLNLinear.wrappedValue = Linear(min(c.dim, Fibo2DiTConfig.adalnEmbedDim), c.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let scale = 1.0 + adaLNLinear(silu(c))
        return linear(normFinal(x) * scale[0..., .newAxis, 0...])
    }
}

public final class Fibo2Transformer: Module {
    public let config: Fibo2DiTConfig
    /// The compute dtype of the token stream (the loader sets it to the weights' dtype).
    public var dtype: DType = .bfloat16

    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "t_embedder") var tEmbedder: Fibo2TimestepEmbedder
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [Fibo2TransformerBlock]
    @ModuleInfo var perceiver: Fibo2PerceiverResampler
    @ModuleInfo var layers: [Fibo2TransformerBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: Fibo2FinalLayer
    /// The injection blocks' `cross_attn_gate` scalars, in `injectionLayerIds` order.
    @ParameterInfo(key: "cross_attn_gates") var crossAttnGates: MLXArray

    let ropeEmbedder: Fibo2RopeEmbedder

    public init(_ config: Fibo2DiTConfig = Fibo2DiTConfig()) {
        self.config = config
        self._xEmbedder.wrappedValue = Linear(config.tokenChannels, config.dim, bias: true)
        self._tEmbedder.wrappedValue = Fibo2TimestepEmbedder(
            outSize: min(config.dim, Fibo2DiTConfig.adalnEmbedDim), midSize: 1024)
        self._noiseRefiner.wrappedValue = (0..<config.nRefinerLayers).map { _ in Fibo2TransformerBlock(config) }
        self._perceiver.wrappedValue = Fibo2PerceiverResampler(config)
        self._layers.wrappedValue = (0..<config.nLayers).map { i in
            Fibo2TransformerBlock(config, crossAttentionDim: config.injectionLayerIds.contains(i) ? config.capFeatDim : nil)
        }
        self._finalLayer.wrappedValue = Fibo2FinalLayer(config)
        self._crossAttnGates.wrappedValue = zeros([config.injectionLayerIds.count])
        self.ropeEmbedder = Fibo2RopeEmbedder(theta: config.ropeTheta, axesDims: config.axesDims, axesLens: config.axesLens)
    }

    /// Token grid of one image: (rows, columns) of /16 cells.
    public struct Grid: Sendable, Equatable {
        public var rows: Int
        public var columns: Int
        public init(rows: Int, columns: Int) { self.rows = rows; self.columns = columns }
        public var count: Int { rows * columns }
    }

    /// What stays fixed across the denoising steps of one conditioning: the gist tokens, RoPE and the noisy mask.
    public struct Conditioning {
        public var gist: MLXArray            // [1, G, dim]
        public var bundles: MLXArray         // [1, 6, L, capFeatDim] in the DiT dtype
        public var gistCount: Int
    }

    /// The Perceiver's gist tokens for a prompt's bundles ([1, 6, L, capFeatDim]). They do not depend on the
    /// timestep, so a sampler computes them once per prompt.
    public func condition(_ bundlesIn: MLXArray) -> Conditioning {
        let bundles = bundlesIn.asType(dtype)
        let count = config.gistCount(textLength: bundles.dim(2))
        return Conditioning(gist: perceiver(bundles[0..., 0], count: count), bundles: bundles, gistCount: count)
    }

    /// Fixed token geometry of a run: RoPE for the refiner (image | context) and the full stream, and the masks.
    public struct Layout {
        var imageRope: Fibo2Rope
        var streamRope: Fibo2Rope
        var refinerMask: MLXArray?
        var streamMask: MLXArray
        var imageTokens: Int
        var gistCount: Int
    }

    public func layout(image: Grid, context: [Grid], gistCount: Int) -> Layout {
        let plane = config.maxNumGistTokens + 1
        var ids = Fibo2RopeEmbedder.grid(size: (1, image.rows, image.columns), start: (plane, 0, 0))
        for (k, grid) in context.enumerated() {
            ids += Fibo2RopeEmbedder.grid(size: (1, grid.rows, grid.columns), start: (plane + k + 1, 0, 0))
        }
        let imageRope = ropeEmbedder(ids)
        let gistRope = ropeEmbedder(Fibo2RopeEmbedder.grid(size: (gistCount, 1, 1), start: (1, 0, 0)))
        let contextTokens = context.reduce(0) { $0 + $1.count }
        func mask(_ flags: [Bool]) -> MLXArray { MLXArray(flags).reshaped(1, flags.count, 1) }
        let imageFlags = [Bool](repeating: true, count: image.count) + [Bool](repeating: false, count: contextTokens)
        return Layout(
            imageRope: imageRope, streamRope: Fibo2Rope.concatenated([gistRope, imageRope]),
            refinerMask: context.isEmpty ? nil : mask(imageFlags),
            streamMask: mask([Bool](repeating: false, count: gistCount) + imageFlags),
            imageTokens: image.count, gistCount: gistCount)
    }

    /// One velocity prediction.
    /// - image: [N, 128] image tokens (the packed latent, row-major); fp32 or the DiT dtype.
    /// - timestep: the noise level in [0, 1].
    /// - context: clean tokens of the images to edit, [M, 128] in all (already concatenated), or nil.
    /// Returns the velocity [N, 128] in the DiT dtype.
    public func callAsFunction(image: MLXArray, timestep: Float, conditioning: Conditioning, context: MLXArray?,
                               layout: Layout, capture: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let t = MLXArray([timestep * config.tScale])
        let adalnNoisy = tEmbedder(t).asType(dtype)
        let adalnClean = tEmbedder(MLXArray([Float(0)])).asType(dtype)

        var tokens = image.asType(dtype)
        if let context { tokens = concatenated([tokens, context.asType(dtype)], axis: 0) }
        var hidden = xEmbedder(tokens)[.newAxis]
        capture?("x_embed", hidden)

        for (i, block) in noiseRefiner.enumerated() {
            hidden = block(hidden, rope: layout.imageRope, adalnNoisy: adalnNoisy, adalnClean: adalnClean,
                           noisyMask: layout.refinerMask)
            capture?("refiner.\(i)", hidden)
        }

        capture?("gist", conditioning.gist)
        hidden = concatenated([conditioning.gist, hidden], axis: 1)
        for (i, block) in layers.enumerated() {
            var text: MLXArray? = nil, gate: MLXArray? = nil
            if let k = config.injectionLayerIds.firstIndex(of: i) {
                text = conditioning.bundles[0..., k + 1]
                gate = crossAttnGates[k].asType(dtype)
            }
            hidden = block(hidden, rope: layout.streamRope, adalnNoisy: adalnNoisy, adalnClean: adalnClean,
                           noisyMask: layout.streamMask, text: text, gate: gate)
            capture?("layer.\(i)", hidden)
        }

        let start = layout.gistCount
        return finalLayer(hidden[0..., start ..< start + layout.imageTokens], adalnNoisy)[0]
    }
}
