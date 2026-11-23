// MLXEngine packages for Bria's Fibo 2.0 on one core:
//   - Fibo2DevPackage (`fibo-2-dev`): `textToImage` + `imageEdit`, the full base model;
//   - Fibo2EditTurboPackage (`fibo-2-edit-turbo`): `imageEdit` only, distilled to 4 steps without guidance;
//   - Fibo2RMBGPackage (`rmbg-3`): `matting` only. Bria's RMBG 3.0, Edit Turbo fine-tuned to render an alpha matte.
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
            return "Fibo 2.0 snapshot not readable at \(p). The repo is gated: the Hugging Face account behind the "
                + "token needs access to it."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        case .unsupportedQuant(let q): return "Fibo 2.0 has no \(q) tier (bf16, int8 and int4 are declared)."
        case .outsideMeasuredEnvelope(let why): return why
        }
    }

    public var description: String { errorDescription ?? "Fibo 2.0 error" }
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

    /// Int ≥ 0, RMBG 3.0 only: the noise seed of the matte. Default 0, the model card's. (`MattingRequest` has no
    /// seed field.) The card reports rare broken mattes; another seed is the way to retry one.
    public static let seed = "seed"

    static func seed(_ meta: MetaData) throws -> UInt64? {
        switch meta[seed] {
        case nil, .null?: return nil
        case .int(let n)? where n >= 0: return UInt64(n)
        default: throw PackageError.unsupportedRequestFeature("metaData[\"\(seed)\"] is a whole number, 0 or more")
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
    /// The root of `text_encoder/` and `processor/`: where the per-request encoder loads from.
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

    static func load(configuration c: Fibo2Configuration, snapshot: URL, shared: URL) async throws -> Fibo2Generator {
        let quant: Fibo2Quant
        switch c.quant {
        case .bf16: quant = .bf16
        case .int8: quant = .int8
        case .int4: quant = .int4
        default: throw Fibo2PackageError.unsupportedQuant(c.quant)
        }
        let pipeline = try await Fibo2Pipeline.load(snapshot: snapshot, shared: shared, quant: quant, textEncoderQuant: nil)
        MLX.Memory.clearCache()
        return Fibo2Generator(pipeline: pipeline, snapshot: shared, keepEncoderResident: c.keepEncoderResident,
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
        guard let snapshot = c.resolvedSnapshotDirectory(storeRoot: c.modelsRootDirectory), Fibo2Configuration.hasTransformer(snapshot)
        else { throw Fibo2PackageError.unreadableSnapshot(c.snapshotPath.isEmpty ? c.variant.repo : c.snapshotPath) }
        guard let shared = c.resolvedSharedDirectory(storeRoot: c.modelsRootDirectory), Fibo2Configuration.hasShared(shared)
        else { throw Fibo2PackageError.unreadableSnapshot(c.sharedSnapshotPath.isEmpty ? c.variant.repo : c.sharedSnapshotPath) }
        return try await Fibo2Generator.load(configuration: c, snapshot: snapshot, shared: shared)
    }

    static func run(_ request: any CapabilityRequest, generator: Fibo2Generator,
                    configuration c: Fibo2Configuration) async throws -> any CapabilityResponse {
        if c.variant == .rmbg { return try await matte(request, generator: generator) }
        var req = Fibo2Pipeline.Request(prompt: "")
        let steps: Int?, guidance: Double?, seed: UInt64?
        let meta: MetaData
        switch request.capability {
        case .textToImage:
            // Edit Turbo is distilled for editing only.
            guard c.variant == .dev, let t2i = request as? T2IRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
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

    /// RMBG 3.0: one edit with the instruction the model was trained on, 4 steps without guidance. The picture goes in
    /// whole, stretched to the trained size for its aspect ratio; the matte comes back at the picture's size.
    static func matte(_ request: any CapabilityRequest, generator: Fibo2Generator) async throws -> any CapabilityResponse {
        guard request.capability == .matting, let matting = request as? MattingRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let source: Fibo2RGBImage
        do { source = try Fibo2RGBImage.read(data: matting.image.data) } catch { throw Fibo2PackageError.imageDecode }
        let req = Fibo2Matte.request(for: source, seed: try Fibo2MetaKeys.seed(matting.metaData) ?? Fibo2Matte.defaultSeed)
        if let why = Fibo2Envelope.violation(width: req.width, height: req.height, imageCount: 1) {
            throw Fibo2PackageError.outsideMeasuredEnvelope(why)
        }
        try Task.checkCancellation()
        let result = try await generator.generate(req)
        try Task.checkCancellation()
        let binary = matting.preferredKind == .binary
        let matte = Fibo2Matte.matte(from: result.image, width: source.width, height: source.height, binary: binary)
        let png: Data
        do { png = try matte.pngData() } catch { throw Fibo2PackageError.pngEncode }
        return MattingResponse(matte: Matte(data: png, width: matte.width, height: matte.height,
                                            kind: binary ? .binary : .softAlpha))
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

    nonisolated static func manifest(_ variant: Fibo2Variant) -> PackageManifest {
        let edit: String
        let surfaces: [ToolDescriptor]
        let editCommon = "The prompt is the edit instruction (\"replace the jacket with a red leather one\"), or a JSON "
            + "object with \"edit_instruction\" and the caption of the result under \"structured_caption\". Several "
            + "images are named in order as <image_1>, <image_2>. The result is drawn from noise at the size trained for "
            + "the first image's aspect ratio (about one megapixel), which crops it to that ratio. metaData[\"mask\"] "
            + "(a base64 PNG the image's size, white = the region to regenerate) confines the edit of a single image. "
            + "Up to 5 images; every image adds its tokens to every step. "
        switch variant {
        case .dev:
            edit = "Fibo 2.0 Dev instruction editing, 50 steps with guidance 5 (about 6 minutes with one image on an M5 "
                + "Max; Fibo 2.0 Edit Turbo does it in 4 steps). " + editCommon + licenceNote
            surfaces = [
                T2IContract.descriptor(
                    name: variant.packageName,
                    summary: "Fibo 2.0 Dev text-to-image (Bria, 8.1B single-stream DiT on Qwen3-VL-4B, FLUX.2 VAE): the "
                        + "full base model, 50 steps with classifier-free guidance 5, about one megapixel (1024×1024, "
                        + "1424×800, 1248×832, …). Two DiT passes per step: about 3.5 s a step on an M5 Max. "
                        + promptGuidance
                        + "negativePrompt defaults to the model card's text-to-image negative. " + licenceNote,
                    modes: []),
                IEditContract.descriptor(name: variant.packageName, summary: edit, modes: []),
            ]
        case .editTurbo:
            edit = "Fibo 2.0 Edit Turbo instruction editing (Bria; distilled from Fibo 2.0 Dev): 4 steps without "
                + "guidance, one DiT pass a step, so a negative prompt does nothing. Editing only. " + editCommon + licenceNote
            surfaces = [IEditContract.descriptor(name: variant.packageName, summary: edit, modes: [])]
        case .rmbg:
            surfaces = [MattingContract.descriptor(
                name: variant.packageName,
                summary: "RMBG 3.0 (Bria): background removal as an image edit. Fibo 2.0 Edit Turbo fine-tuned and "
                    + "distilled to render a soft alpha matte in 4 steps without guidance: white is the subject, grey is "
                    + "partial transparency at hair, fur and motion blur. It knows what the subject is: group shots, "
                    + "holes in mesh and spokes, cast shadows left out. The matte is rendered at about one megapixel "
                    + "and resized to the picture, so it is softer than the pixels of a large picture. Glass, veils and "
                    + "smoke tend to come out opaque. An 8.1B model: seconds per picture, not a fast matte. "
                    + "preferredKind \"binary\" thresholds the matte at 128. metaData[\"seed\"] (default 0) changes "
                    + "the noise, to retry a broken matte. " + licenceNote,
                modes: [])]
        }
        return PackageManifest(
            license: LicenseDeclaration(weightLicense: .ccByNC4, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: variant.repo, revision: variant.revision, tier: 1),
            requirements: RequirementsManifest(
                footprints: Fibo2Footprints.declared,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates
            ),
            specialties: [],
            surfaces: surfaces
        )
    }
}

@InferenceActor
public final class Fibo2DevPackage: ModelPackage {
    public typealias Configuration = Fibo2Configuration

    public nonisolated static var manifest: PackageManifest { Fibo2Serving.manifest(.dev) }

    let configuration: Configuration
    private var generator: Fibo2Generator?

    /// The variant is this package's, whatever the configuration carries.
    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.variant = .dev
        self.configuration = c
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

/// Fibo 2.0 Edit Turbo: the same architecture, distilled from Dev to 4 steps without guidance, for editing only.
@InferenceActor
public final class Fibo2EditTurboPackage: ModelPackage {
    public typealias Configuration = Fibo2Configuration

    public nonisolated static var manifest: PackageManifest { Fibo2Serving.manifest(.editTurbo) }

    let configuration: Configuration
    private var generator: Fibo2Generator?

    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.variant = .editTurbo
        self.configuration = c
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

extension Fibo2EditTurboPackage {
    public nonisolated static var registration: PackageRegistration { .of(Fibo2EditTurboPackage.self) }
}

/// RMBG 3.0: the same architecture, Edit Turbo fine-tuned and distilled to render an alpha matte. Matting only.
@InferenceActor
public final class Fibo2RMBGPackage: ModelPackage {
    public typealias Configuration = Fibo2Configuration

    public nonisolated static var manifest: PackageManifest { Fibo2Serving.manifest(.rmbg) }

    let configuration: Configuration
    private var generator: Fibo2Generator?

    /// The variant and the model's 4 unguided steps are this package's, whatever the configuration carries.
    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.variant = .rmbg
        c.steps = nil
        c.guidance = nil
        self.configuration = c
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

extension Fibo2RMBGPackage {
    public nonisolated static var registration: PackageRegistration { .of(Fibo2RMBGPackage.self) }
}
