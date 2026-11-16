// Init-time configuration (C9) for the Fibo 2.0 packages, parameterized by checkpoint. The wrapping package
// (Fibo2DevPackage / Fibo2EditTurboPackage) fixes the variant.
//
// Weights materialize from Bria's own repos, pinned by commit. Both are GATED (a non-commercial access form each), so
// nothing is mirrored and a first download needs a Hugging Face token whose account has been granted access; the
// engine resolves HF_TOKEN, then the Keychain, then the CLI token file. Quant tiers are made at load from the bf16
// files.
//
// The two repos ship the same text encoder, tokenizer, VAE and fonts, byte for byte (LFS sha256 compared 2026-11-15);
// only `transformer/` differs. Edit Turbo therefore declares two sources in its own repo, its transformer and the
// shared components, and takes the shared ones from a Dev snapshot when one is already installed: 16 GB to fetch, not
// 25.

import Fibo2
import Foundation
import MLXToolKit

public enum Fibo2Variant: String, Codable, Sendable, CaseIterable {
    /// The full base model: text-to-image and editing, 50 steps with classifier-free guidance 5.
    case dev
    /// Distilled from Dev for editing: 4 steps, no guidance. Editing only.
    case editTurbo

    public var repo: String {
        switch self {
        case .dev: "briaai/Fibo-2.0-dev"
        case .editTurbo: "briaai/Fibo-2.0-edit-turbo"
        }
    }

    /// The upstream revision the port was gated against (AB-T-0493).
    public var revision: String {
        switch self {
        case .dev: "b01e5a34a0a9540c65f585c2c281535deabac16d"
        case .editTurbo: "7b133d62d92e897f1468af0427a73cc41a03cb5d"
        }
    }

    public var defaultSteps: Int { self == .dev ? Fibo2Schedule.defaultSteps : Fibo2Schedule.turboSteps }
    public var defaultGuidance: Double { Double(self == .dev ? Fibo2Schedule.defaultGuidance : Fibo2Schedule.turboGuidance) }
    public var packageName: String { self == .dev ? "fibo-2-dev" : "fibo-2-edit-turbo" }
    public var title: String { self == .dev ? "Fibo 2.0 Dev" : "Fibo 2.0 Edit Turbo" }
}

public struct Fibo2Configuration: PackageConfiguration, ModelStorable, QuantConfigured, FootprintConfigured,
    WeightSourcing {
    public var variant: Fibo2Variant
    /// DiT tier: `.bf16`, `.int8` or `.int4`, all made at load.
    public var quant: Quant
    /// The Qwen3-VL-4B conditioner tier, orthogonal to `quant`.
    public var textEncoderQuant: Fibo2TextEncoderQuant
    /// Explicit snapshot root holding this variant's `transformer/`. Empty = the store.
    public var snapshotPath: String
    /// Explicit root of `text_encoder/`, `processor/`, `vae/` and `fonts/`. Empty = `snapshotPath` when it holds them,
    /// else the store (this variant's snapshot, else a Dev snapshot).
    public var sharedSnapshotPath: String
    /// nil = the variant's: 50 for Dev (the model card's; diffusers' own default is 30), 4 for Edit Turbo.
    public var steps: Int?
    /// nil = the variant's: 5 for Dev, 1 (no guidance) for Edit Turbo.
    public var guidance: Double?
    /// Keep the conditioner (8.9 GB bf16) resident between requests. Off: it loads per request and is released before
    /// denoising, which is what the declared footprint measures.
    public var keepEncoderResident: Bool
    public var modelsRootDirectory: URL?

    public var defaultSteps: Int { steps ?? variant.defaultSteps }
    public var defaultGuidance: Double { guidance ?? variant.defaultGuidance }

    /// A snapshot's transformer files, and the components both checkpoints share.
    public static let transformerFiles = ["model_index.json", "README.md", "transformer/*", "scheduler/*"]
    public static let sharedFiles = ["text_encoder/*", "processor/*", "vae/*", "fonts/*"]

    public init(variant: Fibo2Variant = .dev, quant: Quant = .bf16, textEncoderQuant: Fibo2TextEncoderQuant = .bf16,
                snapshotPath: String = "", sharedSnapshotPath: String = "", steps: Int? = nil, guidance: Double? = nil,
                keepEncoderResident: Bool = false, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.quant = quant
        self.textEncoderQuant = textEncoderQuant
        self.snapshotPath = snapshotPath
        self.sharedSnapshotPath = sharedSnapshotPath
        self.steps = steps
        self.guidance = guidance
        self.keepEncoderResident = keepEncoderResident
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant, quant, textEncoderQuant, snapshotPath, sharedSnapshotPath, steps, guidance, keepEncoderResident
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decodeIfPresent(Fibo2Variant.self, forKey: .variant) ?? .dev
        quant = try c.decode(Quant.self, forKey: .quant)
        textEncoderQuant = try c.decodeIfPresent(Fibo2TextEncoderQuant.self, forKey: .textEncoderQuant) ?? .bf16
        snapshotPath = try c.decodeIfPresent(String.self, forKey: .snapshotPath) ?? ""
        sharedSnapshotPath = try c.decodeIfPresent(String.self, forKey: .sharedSnapshotPath) ?? ""
        steps = try c.decodeIfPresent(Int.self, forKey: .steps)
        guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
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
        switch variant {
        case .dev:
            return [WeightSource(role: "snapshot", repo: variant.repo, revision: variant.revision,
                                 matching: Self.transformerFiles + Self.sharedFiles)]
        case .editTurbo:
            return [WeightSource(role: "transformer", repo: variant.repo, revision: variant.revision,
                                 matching: Self.transformerFiles),
                    WeightSource(role: "shared", repo: variant.repo, revision: variant.revision,
                                 matching: Self.sharedFiles)]
        }
    }

    static func hasTransformer(_ dir: URL) -> Bool {
        FileManager.default.fileExists(atPath: dir.appendingPathComponent("transformer/config.json").path)
    }

    static func hasShared(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ["text_encoder/config.json", "processor/tokenizer.json", "vae/config.json"]
            .allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    static func complete(_ dir: URL) -> Bool { hasTransformer(dir) && hasShared(dir) }

    static func storeDirectory(_ store: ModelStore, _ variant: Fibo2Variant, where ok: (URL) -> Bool) -> URL? {
        if let flat = store.directory(for: variant.repo), ok(flat) { return flat }
        if let snap = store.snapshotDirectory(for: variant.repo, revision: variant.revision), ok(snap) { return snap }
        return nil
    }

    /// The root of this variant's `transformer/`: the explicit path, else the store.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        return Self.storeDirectory(store, variant, where: Self.hasTransformer) ?? store.directory(for: variant.repo)
    }

    /// The root of the shared components: the explicit path; the transformer's own root when it holds them; else this
    /// variant's snapshot in the store, else a Dev snapshot there (the same bytes).
    public func resolvedSharedDirectory(storeRoot: URL?) -> URL? {
        if !sharedSnapshotPath.isEmpty { return URL(fileURLWithPath: sharedSnapshotPath) }
        if !snapshotPath.isEmpty, Self.hasShared(URL(fileURLWithPath: snapshotPath)) { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        if let own = Self.storeDirectory(store, variant, where: Self.hasShared) { return own }
        if variant != .dev, let dev = Self.storeDirectory(store, .dev, where: Self.hasShared) { return dev }
        return nil
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let transformer = resolvedSnapshotDirectory(storeRoot: storeRoot).map(Self.hasTransformer) ?? false
        let shared = resolvedSharedDirectory(storeRoot: storeRoot).map(Self.hasShared) ?? false
        return weightSources.filter { source in
            switch source.role {
            case "transformer": return !transformer
            case "shared": return !shared
            default: return !(transformer && shared)
            }
        }
    }
}
