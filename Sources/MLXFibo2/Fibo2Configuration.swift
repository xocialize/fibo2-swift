// Init-time configuration (C9) for the Fibo 2.0 Dev package.
//
// Weights materialize from Bria's own repo, pinned by commit: `briaai/Fibo-2.0-dev` is GATED (a non-commercial access
// form), so nothing is mirrored and a first download needs a Hugging Face token whose account has been granted
// access; the engine resolves HF_TOKEN, then the Keychain, then the CLI token file. Quant tiers are made at load from
// the bf16 files.

import Fibo2
import Foundation
import MLXToolKit

public struct Fibo2Configuration: PackageConfiguration, ModelStorable, QuantConfigured, FootprintConfigured,
    WeightSourcing {
    /// DiT tier: `.bf16`, `.int8` or `.int4`, all made at load.
    public var quant: Quant
    /// The Qwen3-VL-4B conditioner tier, orthogonal to `quant`.
    public var textEncoderQuant: Fibo2TextEncoderQuant
    /// Explicit snapshot root (`transformer/`, `text_encoder/`, `processor/`, `vae/`, `fonts/`). Empty = the store.
    public var snapshotPath: String
    /// The model card's 50; diffusers' own default is 30.
    public var defaultSteps: Int
    public var defaultGuidance: Double
    /// Keep the conditioner (8.9 GB bf16) resident between requests. Off: it loads per request and is released before
    /// denoising, which is what the declared footprint measures.
    public var keepEncoderResident: Bool
    public var modelsRootDirectory: URL?

    public static let repo = "briaai/Fibo-2.0-dev"
    /// The upstream revision the port was gated against (oracle goldens, AB-T-0493).
    public static let revision = "b01e5a34a0a9540c65f585c2c281535deabac16d"
    /// What the package reads from the snapshot (assets, the fine-tuning kit and the style prompts are not fetched).
    public static let files = ["model_index.json", "README.md", "transformer/*", "text_encoder/*", "processor/*",
                               "vae/*", "fonts/*", "scheduler/*"]

    public init(quant: Quant = .bf16, textEncoderQuant: Fibo2TextEncoderQuant = .bf16, snapshotPath: String = "",
                defaultSteps: Int = Fibo2Schedule.defaultSteps, defaultGuidance: Double = Double(Fibo2Schedule.defaultGuidance),
                keepEncoderResident: Bool = false, modelsRootDirectory: URL? = nil) {
        self.quant = quant
        self.textEncoderQuant = textEncoderQuant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.keepEncoderResident = keepEncoderResident
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case quant, textEncoderQuant, snapshotPath, defaultSteps, defaultGuidance, keepEncoderResident
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decode(Quant.self, forKey: .quant)
        textEncoderQuant = try c.decodeIfPresent(Fibo2TextEncoderQuant.self, forKey: .textEncoderQuant) ?? .bf16
        snapshotPath = try c.decodeIfPresent(String.self, forKey: .snapshotPath) ?? ""
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? Fibo2Schedule.defaultSteps
        defaultGuidance = try c.decodeIfPresent(Double.self, forKey: .defaultGuidance) ?? Double(Fibo2Schedule.defaultGuidance)
        keepEncoderResident = try c.decodeIfPresent(Bool.self, forKey: .keepEncoderResident) ?? false
        modelsRootDirectory = nil
    }

    /// A resident encoder adds its weights to the DiT tier's resident bytes.
    public var residentBytesHint: UInt64? {
        guard keepEncoderResident else { return nil }
        let dit = Fibo2Footprints.declared.first { $0.quant == quant }?.residentBytes ?? 0
        return dit + Fibo2Footprints.encoderBytes(textEncoderQuant)
    }

    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: Self.repo, revision: Self.revision, matching: Self.files)]
    }

    static func complete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ["transformer/config.json", "text_encoder/config.json", "processor/tokenizer.json", "vae/config.json"]
            .allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        weightSources.filter { _ in
            if !snapshotPath.isEmpty, Self.complete(URL(fileURLWithPath: snapshotPath)) { return false }
            guard let dir = resolvedSnapshotDirectory(storeRoot: storeRoot) else { return true }
            return !Self.complete(dir)
        }
    }

    /// The snapshot root: the explicit path, else the store's flat or revision directory.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        if let flat = store.directory(for: Self.repo), Self.complete(flat) { return flat }
        if let snap = store.snapshotDirectory(for: Self.repo, revision: Self.revision), Self.complete(snap) { return snap }
        return store.directory(for: Self.repo)
    }
}
