// Weight loading for Fibo 2.0: the DiT loads from the upstream diffusers shards (`transformer/*.safetensors`, 937
// tensors) under its own key names, two-way strict, after three renames (see Fibo2Transformer.swift). Quantized tiers
// are made on load from the bf16 files (affine, group 64), block by block from the lazy loads, so the bf16 copy is
// never resident in full.

import Foundation
import MLX
import MLXNN

/// DiT weight tier.
public enum Fibo2Quant: String, Codable, Sendable, CaseIterable {
    case bf16, int8, int4

    public var bits: Int? {
        switch self {
        case .bf16: nil
        case .int8: 8
        case .int4: 4
        }
    }
}

public enum Fibo2Weights {

    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.mlp_in.")
            .replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp_out.")
            .replacingOccurrences(of: "final_layer.adaLN_modulation.1.", with: "final_layer.adaLN_linear.")
    }

    public static func config(snapshot: URL) throws -> Fibo2DiTConfig {
        let url = snapshot.appendingPathComponent("transformer/config.json")
        guard let data = try? Data(contentsOf: url) else { return Fibo2DiTConfig() }
        return try JSONDecoder().decode(Fibo2DiTConfig.self, from: data)
    }

    /// The DiT from a snapshot holding `transformer/`, cast to `dtype`, quantized on load when `quant` asks for it.
    public static func loadTransformer(snapshot: URL, dtype: DType = .bfloat16, quant: Fibo2Quant = .bf16,
                                       groupSize: Int = 64) throws -> Fibo2Transformer {
        let dir = snapshot.appendingPathComponent("transformer")
        let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Fibo2Error.loading("no transformer/*.safetensors under \(snapshot.path)") }
        let config = try config(snapshot: snapshot)
        let model = Fibo2Transformer(config)
        // The checkpoint dictionary lives only inside this scope. A quantized tier evaluates and drops the bf16
        // weights block by block; a dictionary still holding them would keep all 16 GB alive beside the quantized
        // copy (int8: a 24.9 GB process peak at load, measured, against 9.8 GB once it is released first).
        try loadWeights(into: model, files: files, config: config, dtype: dtype, evaluate: quant.bits == nil)
        model.dtype = dtype
        if let bits = quant.bits {
            quantizeTransformer(model, bits: bits, groupSize: groupSize, eightBit: quant == .int4 ? int4EightBitLayers : [])
        }
        return model
    }

    /// The int4 tier's 8-bit layers: every attention projection and the FFN down-projection (`w2`); the FFN's two
    /// up-projections (`w1`, `w3`), 39 % of a block, are the 4-bit part. Uniform 4-bit (group 64) puts the velocity at
    /// cos 0.979 (text-to-image) / 0.985 (edit) against fp32; this recipe reads 0.9925 / 0.9952, the best of six
    /// probed (attention alone 0.984 / 0.994, `w2` alone 0.982 / 0.986, group 32 0.980 / 0.991, attention at group 32
    /// 0.983 / 0.994). int8 reads 0.9989 / 0.9996.
    public static let int4EightBitLayers = ["to_q", "to_k", "to_v", "to_out", "feed_forward.w2", "ffn.w2"]

    static func loadWeights(into model: Fibo2Transformer, files: [URL], config: Fibo2DiTConfig, dtype: DType,
                            evaluate: Bool) throws {
        var weights: [String: MLXArray] = [:]
        var gates: [Int: MLXArray] = [:]
        for file in files {
            for (k, v) in try MLX.loadArrays(url: file) {
                if k.hasPrefix("layers."), k.hasSuffix(".cross_attn_gate"), let id = Int(k.split(separator: ".")[1]) {
                    gates[id] = v.asType(dtype).reshaped(1)
                } else {
                    weights[sanitizeDiTKey(k)] = v.asType(dtype)
                }
            }
        }
        guard Set(gates.keys) == Set(config.injectionLayerIds) else {
            throw Fibo2Error.loading("cross_attn_gate keys \(gates.keys.sorted()) ≠ injection layers \(config.injectionLayerIds)")
        }
        weights["cross_attn_gates"] = concatenated(config.injectionLayerIds.map { gates[$0]! }, axis: 0)
        try verifyAndLoad(model: model, weights: weights, label: "Fibo 2.0 DiT", evaluate: evaluate)
    }

    /// Layers kept at the load dtype in a quantized tier: the patch embedding, the timestep MLP, the final layer and
    /// every block's adaLN modulation producer (a 256-wide input each). About 0.17B of the 8.1B parameters.
    static func keepHiPrecision(path: String) -> Bool {
        path.hasPrefix("x_embedder") || path.hasPrefix("t_embedder") || path.hasPrefix("final_layer")
            || path.contains("adaLN_modulation")
    }

    /// Quantize every block Linear in place (affine). `FIBO2_QUANT_GROUP` / `FIBO2_QUANT_EIGHT_BIT` (comma-separated
    /// path substrings kept at 8 bits inside a 4-bit tier) override the recipe for recipe work. Evaluated one block at
    /// a time, so a model still holding lazy loads is read, quantized and dropped block by block.
    public static func quantizeTransformer(_ model: Fibo2Transformer, bits: Int, groupSize: Int = 64,
                                           eightBit: [String] = []) {
        let env = ProcessInfo.processInfo.environment
        let group = env["FIBO2_QUANT_GROUP"].flatMap(Int.init) ?? groupSize
        let keep8 = env["FIBO2_QUANT_EIGHT_BIT"].map { $0.split(separator: ",").map(String.init) } ?? eightBit
        quantize(model: model, filter: { path, module in
            guard module is Linear, !keepHiPrecision(path: path) else { return nil }
            let b = keep8.contains { path.contains($0) } ? 8 : bits
            return (group, b, .affine)
        })
        for block in model.noiseRefiner { eval(block) }
        for layer in model.perceiver.layers { eval(layer) }
        for block in model.layers { eval(block) }
        eval(model)
    }

    /// Two-way strict load: every module key filled, every checkpoint key consumed, shapes equal. `evaluate: false`
    /// leaves the lazy loads in place for a quantizer to consume.
    public static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String,
                                     evaluate: Bool = true) throws {
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        let missing = Set(shapes.keys).subtracting(weights.keys).sorted()
        guard missing.isEmpty else {
            throw Fibo2Error.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. " + missing.prefix(4).joined(separator: ", "))
        }
        let unused = Set(weights.keys).subtracting(shapes.keys).sorted()
        guard unused.isEmpty else {
            throw Fibo2Error.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. " + unused.prefix(4).joined(separator: ", "))
        }
        for (k, v) in weights where shapes[k] != v.shape {
            throw Fibo2Error.loading("\(label): shape mismatch at \(k): checkpoint \(v.shape) vs module \(shapes[k] ?? [])")
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        if evaluate { eval(model) }
    }
}
