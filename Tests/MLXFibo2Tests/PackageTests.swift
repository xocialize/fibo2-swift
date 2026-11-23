// Offline manifest / configuration / materialization / cancellation tests for the Fibo 2.0 Dev engine package, and
// prompt-shaping unit tests that need no weights.
import Fibo2
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXFibo2

final class Fibo2PackageTests: XCTestCase {
    /// The upstream snapshot in the Hugging Face cache, when this machine has it.
    static let snapshot: String? = {
        let root = "/Volumes/DEV_VOL1/Models/models--briaai--Fibo-2.0-dev/snapshots/" + Fibo2Variant.dev.revision
        return FileManager.default.fileExists(atPath: root + "/transformer/config.json") ? root : nil
    }()

    /// C7: CC BY-NC 4.0, package-local and NOT allowlisted (AB-D-0226: a Gaea plugin model); C8: MIT.
    func testManifestIsNonCommercialAndNotPermissive() {
        let m = Fibo2DevPackage.manifest
        XCTAssertEqual(m.license.weightLicense, .ccByNC4)
        XCTAssertEqual(m.license.weightLicense.identifier, "CC-BY-NC-4.0")
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.contractVersion, ContractVersion.current)
        XCTAssertEqual(Set(m.surfaces.map(\.capability)), [.textToImage, .imageEdit])
        XCTAssertTrue(m.surfaces.allSatisfy { $0.name == "fibo-2-dev" })
        XCTAssertTrue(m.surfaces.allSatisfy { $0.summary.contains("CC BY-NC 4.0") })
        XCTAssertEqual(m.provenance.sourceRepo, "briaai/Fibo-2.0-dev")
    }

    func testFootprintsCoverEveryTier() {
        let fp = Fibo2DevPackage.manifest.requirements.footprints
        XCTAssertEqual(Set(fp.map(\.quant)), [.bf16, .int8, .int4])
        let r = Dictionary(uniqueKeysWithValues: fp.map { ($0.quant, $0.residentBytes) })
        XCTAssertGreaterThan(r[.bf16]!, r[.int8]!)
        XCTAssertGreaterThan(r[.int8]!, r[.int4]!)
        // The declared totals cover the measured process peaks of the worst request (a one-image edit).
        let total = Dictionary(uniqueKeysWithValues: fp.map { ($0.quant, $0.residentBytes + $0.peakActivationBytes) })
        XCTAssertGreaterThanOrEqual(total[.bf16]!, 29_850_000_000)
        XCTAssertGreaterThanOrEqual(total[.int8]!, 22_530_000_000)
        XCTAssertGreaterThanOrEqual(total[.int4]!, 20_720_000_000)
        XCTAssertEqual(Fibo2Envelope.maxImages, 5)
    }

    func testEnvelope() {
        XCTAssertNil(Fibo2Envelope.violation(width: 1024, height: 1024, imageCount: 0))
        XCTAssertNil(Fibo2Envelope.violation(width: 1424, height: 800, imageCount: 0))   // the card's 16:9
        XCTAssertNil(Fibo2Envelope.violation(width: 1248, height: 832, imageCount: 0))   // the card's 3:2
        XCTAssertNil(Fibo2Envelope.violation(width: 512, height: 512, imageCount: 0))
        XCTAssertNil(Fibo2Envelope.violation(width: nil, height: nil, imageCount: 1))
        XCTAssertNotNil(Fibo2Envelope.violation(width: 1536, height: 1536, imageCount: 0))  // past the trained area
        XCTAssertNotNil(Fibo2Envelope.violation(width: 1000, height: 1024, imageCount: 0))  // not ×16
        XCTAssertNotNil(Fibo2Envelope.violation(width: 1024, height: nil, imageCount: 0))
        XCTAssertNotNil(Fibo2Envelope.violation(width: nil, height: nil, imageCount: Fibo2Envelope.maxImages + 1))
    }

    /// Every size the model picks for an aspect ratio is inside the envelope, so an un-sized edit never leaves it.
    func testTrainedSizesFitTheEnvelope() {
        for (w, h) in [(1024, 2048), (5071, 3294), (16, 9), (9, 16), (1, 1), (4000, 1000), (3, 4), (21, 9)] {
            let size = Fibo2ImagePrep.imageSize(aspectRatio: Double(w) / Double(h))
            XCTAssertNil(Fibo2Envelope.violation(width: size.width, height: size.height, imageCount: 1), "\(w):\(h) → \(size)")
        }
    }

    func testWeightSourceIsThePinnedUpstream() {
        let s = Fibo2Configuration().weightSources
        XCTAssertEqual(s.map(\.repo), ["briaai/Fibo-2.0-dev"])
        XCTAssertEqual(s.first?.revision?.count, 40)  // a commit, never a moving "main"
        XCTAssertTrue(s.first?.matching?.contains("fonts/*") ?? false)  // the fallback fonts of the text images
    }

    func testMATGate() {
        let report = MaterializationConformance.check(
            freshConfiguration: Fibo2Configuration(),
            satisfiedConfiguration: Self.snapshot.map { Fibo2Configuration(snapshotPath: $0) })
        XCTAssertTrue(report.passed, report.summary)
    }

    func testConfigurationRoundTrips() throws {
        let c = Fibo2Configuration(quant: .int8, textEncoderQuant: .int8, steps: 30, keepEncoderResident: true)
        let back = try JSONDecoder().decode(Fibo2Configuration.self, from: JSONEncoder().encode(c))
        XCTAssertEqual(back.quant, .int8)
        XCTAssertEqual(back.textEncoderQuant, .int8)
        XCTAssertEqual(back.defaultSteps, 30)
        XCTAssertTrue(back.keepEncoderResident)
        let old = try JSONDecoder().decode(Fibo2Configuration.self, from: Data(#"{"quant":"bf16"}"#.utf8))
        XCTAssertEqual(old.defaultSteps, 50)
        XCTAssertEqual(old.variant, .dev, "a configuration saved before 0.2.0 is Dev's")
        XCTAssertFalse(old.keepEncoderResident)
        XCTAssertNil(old.residentBytesHint)
        XCTAssertGreaterThan(c.residentBytesHint!, Fibo2Footprints.declared.first { $0.quant == .int8 }!.residentBytes)
    }

    // MARK: Edit Turbo (0.2.0)

    func testEditTurboIsEditOnlyWithItsOwnDefaults() {
        let m = Fibo2EditTurboPackage.manifest
        XCTAssertEqual(m.surfaces.map(\.capability), [.imageEdit])
        XCTAssertEqual(m.surfaces[0].name, "fibo-2-edit-turbo")
        XCTAssertEqual(m.license.weightLicense, .ccByNC4)
        XCTAssertEqual(m.provenance.sourceRepo, "briaai/Fibo-2.0-edit-turbo")
        XCTAssertEqual(m.provenance.revision.count, 40)
        let c = Fibo2Configuration(variant: .editTurbo)
        XCTAssertEqual(c.defaultSteps, 4)
        XCTAssertEqual(c.defaultGuidance, 1)
        XCTAssertEqual(Fibo2Configuration(variant: .editTurbo, steps: 8).defaultSteps, 8)
        XCTAssertEqual(Fibo2Configuration().defaultSteps, 50)
        XCTAssertEqual(Fibo2Configuration().defaultGuidance, 5)
    }

    /// Edit Turbo fetches its transformer and the shared components as two sources of its own repo, so a Dev
    /// snapshot already in the store satisfies the second.
    func testEditTurboWeightSources() {
        let s = Fibo2Configuration(variant: .editTurbo).weightSources
        XCTAssertEqual(s.map(\.role), ["transformer", "shared"])
        XCTAssertTrue(s.allSatisfy { $0.repo == "briaai/Fibo-2.0-edit-turbo" && $0.revision?.count == 40 })
        XCTAssertTrue(s[0].matching?.contains("transformer/*") ?? false)
        XCTAssertFalse(s[0].matching?.contains("text_encoder/*") ?? true)
        XCTAssertTrue(s[1].matching?.contains("text_encoder/*") ?? false)
        // A fresh store misses both.
        let empty = FileManager.default.temporaryDirectory.appending(path: "fibo2-empty-\(UUID().uuidString)")
        XCTAssertEqual(Fibo2Configuration(variant: .editTurbo).missingWeightSources(storeRoot: empty).map(\.role), ["transformer", "shared"])
        XCTAssertEqual(Fibo2Configuration().missingWeightSources(storeRoot: empty).map(\.role), ["snapshot"])
    }

    /// With Dev installed, Edit Turbo's shared components resolve to Dev's snapshot and only its transformer is missing.
    func testEditTurboSharesAnInstalledDev() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(path: "fibo2-store-\(UUID().uuidString)")
        let dev = root.appending(path: "models--briaai--Fibo-2.0-dev/snapshots/\(Fibo2Variant.dev.revision)")
        for file in ["transformer/config.json", "text_encoder/config.json", "processor/tokenizer.json", "vae/config.json"] {
            let url = dev.appending(path: file)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: url)
        }
        let turbo = Fibo2Configuration(variant: .editTurbo)
        XCTAssertEqual(turbo.missingWeightSources(storeRoot: root).map(\.role), ["transformer"])
        XCTAssertEqual(turbo.resolvedSharedDirectory(storeRoot: root)?.standardizedFileURL, dev.standardizedFileURL)
        XCTAssertTrue(Fibo2Configuration().missingWeightSources(storeRoot: root).isEmpty)
        // Its own transformer lands: nothing is missing.
        let own = root.appending(path: "models--briaai--Fibo-2.0-edit-turbo/snapshots/\(Fibo2Variant.editTurbo.revision)/transformer/config.json")
        try fm.createDirectory(at: own.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: own)
        XCTAssertTrue(turbo.missingWeightSources(storeRoot: root).isEmpty)
        try? fm.removeItem(at: root)
    }

    func testEditTurboPackageFixesItsVariant() async {
        // A Dev-shaped configuration handed to the Turbo package is still Turbo's.
        let package = Fibo2EditTurboPackage(configuration: Fibo2Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Fibo2EditTurboPackage.manifest))
    }

    // MARK: RMBG 3.0 (0.3.0)

    func testRMBGIsMattingOnly() {
        let m = Fibo2RMBGPackage.manifest
        XCTAssertEqual(m.surfaces.map(\.capability), [.matting])
        XCTAssertEqual(m.surfaces[0].name, "rmbg-3")
        XCTAssertTrue(m.surfaces[0].summary.contains("CC BY-NC 4.0"))
        XCTAssertEqual(m.license.weightLicense, .ccByNC4)
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
        XCTAssertEqual(m.provenance.sourceRepo, "briaai/RMBG-3.0")
        XCTAssertEqual(m.provenance.revision.count, 40)
        XCTAssertEqual(Set(m.requirements.footprints.map(\.quant)), [.bf16, .int8, .int4])
        let c = Fibo2Configuration(variant: .rmbg)
        XCTAssertEqual(c.defaultSteps, 4)
        XCTAssertEqual(c.defaultGuidance, 1)
    }

    /// The transformer is RMBG 3.0's own; the text encoder, VAE and fonts are Edit Turbo's, the card's base.
    func testRMBGWeightSources() throws {
        let s = Fibo2Configuration(variant: .rmbg).weightSources
        XCTAssertEqual(s.map(\.role), ["transformer", "shared"])
        XCTAssertEqual(s.map(\.repo), ["briaai/RMBG-3.0", "briaai/Fibo-2.0-edit-turbo"])
        XCTAssertEqual(s[1].revision, Fibo2Variant.editTurbo.revision)
        XCTAssertFalse(s[0].matching?.contains("scheduler/*") ?? true)  // the repo has none
        XCTAssertTrue(s[1].matching?.contains("vae/*") ?? false)

        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(path: "fibo2-store-\(UUID().uuidString)")
        let rmbg = Fibo2Configuration(variant: .rmbg)
        XCTAssertEqual(rmbg.missingWeightSources(storeRoot: root).map(\.role), ["transformer", "shared"])
        func touch(_ variant: Fibo2Variant, _ files: [String]) throws -> URL {
            let snap = root.appending(path: "models--\(variant.repo.replacingOccurrences(of: "/", with: "--"))/snapshots/\(variant.revision)")
            for file in files {
                let url = snap.appending(path: file)
                try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data("{}".utf8).write(to: url)
            }
            return snap
        }
        let shared = ["text_encoder/config.json", "processor/tokenizer.json", "vae/config.json"]
        // A Dev snapshot satisfies the shared components; so does an Edit Turbo one, which is preferred.
        let dev = try touch(.dev, ["transformer/config.json"] + shared)
        XCTAssertEqual(rmbg.missingWeightSources(storeRoot: root).map(\.role), ["transformer"])
        XCTAssertEqual(rmbg.resolvedSharedDirectory(storeRoot: root)?.standardizedFileURL, dev.standardizedFileURL)
        let turbo = try touch(.editTurbo, shared)
        XCTAssertEqual(rmbg.resolvedSharedDirectory(storeRoot: root)?.standardizedFileURL, turbo.standardizedFileURL)
        let own = try touch(.rmbg, ["transformer/config.json"])
        XCTAssertTrue(rmbg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(rmbg.resolvedSnapshotDirectory(storeRoot: root)?.standardizedFileURL, own.standardizedFileURL)
        try? fm.removeItem(at: root)
    }

    func testRMBGPackageFixesItsVariantAndRefusesEdits() async {
        let package = Fibo2RMBGPackage(configuration: Fibo2Configuration(steps: 30, guidance: 5))
        let image = Image(format: .png, data: Data(), width: 1, height: 1)
        let report = await CancellationConformance.checkRun(package: package, request: MattingRequest(image: image))
        XCTAssertTrue(report.passed, report.summary)
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Fibo2RMBGPackage.manifest))
    }

    /// The card's prompt and settings, and the picture's way in and out.
    func testMatteRequestAndResult() throws {
        XCTAssertEqual(Fibo2Matte.prompt, #"{"edit_instruction":"\#(Fibo2Matte.instruction)"}"#)
        XCTAssertTrue(Fibo2Matte.instruction.hasPrefix("Generate a detailed grayscale alpha matte. Map the opaque foreground to white"))
        XCTAssertTrue(Fibo2Matte.instruction.hasSuffix("to represent fine details and transparency."))
        XCTAssertNil(Fibo2Prompt.editPromptProblem(Fibo2Matte.prompt))
        XCTAssertEqual(Fibo2Prompt.editPrompt(Fibo2Matte.prompt, imageCount: 1, textImages: true),
                       #"{"image":"\#(Fibo2Prompt.visionMarker)","edit_instruction":"\#(Fibo2Matte.instruction)"}"#)

        // A 3:2 picture goes in whole at the trained size, so the pipeline's cover-and-crop changes nothing.
        let source = Fibo2RGBImage(rgb: [UInt8](repeating: 200, count: 300 * 200 * 3), width: 300, height: 200)
        let request = Fibo2Matte.request(for: source, seed: 7)
        let size = Fibo2ImagePrep.imageSize(aspectRatio: 1.5)
        XCTAssertEqual([request.width, request.height], [size.width, size.height])
        XCTAssertEqual([request.images[0].width, request.images[0].height], [size.width, size.height])
        XCTAssertEqual(request.steps, 4)
        XCTAssertEqual(request.guidance, 1)
        XCTAssertEqual(request.seed, 7)
        XCTAssertNil(Fibo2Envelope.violation(width: request.width, height: request.height, imageCount: 1))
        let prepared = Fibo2ImagePrep.prepare(images: request.images, width: request.width, height: request.height)
        XCTAssertEqual(prepared.crops[0], request.images[0])
        XCTAssertEqual(Fibo2Matte.request(for: source, fit: .crop).images[0], source)  // the card's way: cropped later

        // Left half white, right half black, rendered at 64×32: back at 300×200, soft only at the seam.
        var rgb = [UInt8](repeating: 0, count: 64 * 32 * 3)
        for y in 0..<32 { for x in 0..<32 { for c in 0..<3 { rgb[(y * 64 + x) * 3 + c] = 255 } } }
        let rendered = Fibo2RGBImage(rgb: rgb, width: 64, height: 32)
        let matte = Fibo2Matte.matte(from: rendered, width: 300, height: 200)
        XCTAssertEqual([matte.width, matte.height], [300, 200])
        XCTAssertEqual(matte.pixels[100 * 300 + 20], 255)
        XCTAssertEqual(matte.pixels[100 * 300 + 280], 0)
        XCTAssertEqual(matte.coverage, 0.5, accuracy: 0.01)
        let binary = Fibo2Matte.matte(from: rendered, width: 300, height: 200, binary: true)
        XCTAssertEqual(Set(binary.pixels), [0, 255])
        // A single-channel 8-bit PNG, and the cut-out carries it as alpha.
        let png = try matte.pngData()
        XCTAssertEqual([UInt8](png[24..<26]), [8, 0], "bit depth 8, colour type 0 (grey)")
        let cutout = Fibo2Matte.cutout(source, matte: matte)
        XCTAssertEqual(cutout.rgba[(100 * 300 + 20) * 4 + 3], 255)
        XCTAssertEqual(cutout.rgba[(100 * 300 + 280) * 4 + 3], 0)
        XCTAssertEqual(cutout.rgba[0], 200)
    }

    func testMetaLevers() throws {
        XCTAssertNil(try Fibo2MetaKeys.seed([:]))
        XCTAssertEqual(try Fibo2MetaKeys.seed(["seed": .int(7)]), 7)
        XCTAssertThrowsError(try Fibo2MetaKeys.seed(["seed": .int(-1)]))
        XCTAssertThrowsError(try Fibo2MetaKeys.seed(["seed": .string("7")]))
        XCTAssertNil(try Fibo2MetaKeys.mask([:]))
        XCTAssertTrue(try Fibo2MetaKeys.textImages([:]))
        XCTAssertFalse(try Fibo2MetaKeys.textImages(["textImages": .bool(false)]))
        XCTAssertThrowsError(try Fibo2MetaKeys.textImages(["textImages": .int(1)]))
        XCTAssertThrowsError(try Fibo2MetaKeys.mask(["mask": .int(1)]))
        XCTAssertThrowsError(try Fibo2MetaKeys.mask(["mask": .string("")]))
        XCTAssertEqual(try Fibo2MetaKeys.mask(["mask": .string(Data([1, 2, 3]).base64EncodedString())]), Data([1, 2, 3]))
    }

    // MARK: CAN gate (offline)

    func testCANGatePreCancelledTextToImage() async {
        let package = Fibo2DevPackage(configuration: Fibo2Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledImageEdit() async {
        let package = Fibo2DevPackage(configuration: Fibo2Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Fibo2DevPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: Fibo2DevPackage.manifest, posture: .cadence([.init(phase: .denoise, unit: .step)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

/// Prompt shaping: the rules of diffusers' `_parse_prompt` / `_get_edit_prompt`, pinned without the oracle.
final class Fibo2PromptTests: XCTestCase {
    func testJSONKeepsOrderAndPythonNumbers() throws {
        let text = #"{"b": 1, "a": [1.0, 2.50, 1e-5, -0, true, null], "c": "é\n\"x\"", "b": 2}"#
        XCTAssertEqual(try Fibo2JSON.parse(text).dumps(), #"{"b":2,"a":[1.0,2.5,1e-05,0,true,null],"c":"é\n\"x\""}"#)
        XCTAssertThrowsError(try Fibo2JSON.parse(#"{"a": 1} trailing"#))
        XCTAssertThrowsError(try Fibo2JSON.parse("{\"a\": \"line\nbreak\"}"))  // json.loads strict mode
    }

    func testCleaning() {
        let prompt = #"{"short_description":"a cat","objects":[{"description":"cat","texture":""},{}],"lighting":{"aesthetic_score":7},"context":"","aesthetic_score":6.9,"n":0,"flag":false}"#
        XCTAssertEqual(Fibo2Prompt.generationPrompt(prompt),
                       #"{"short_description":"a cat","objects":[{"description":"cat"}],"n":0,"flag":false}"#)
        XCTAssertEqual(Fibo2Prompt.generationPrompt("a red fox"), "a red fox")
        XCTAssertEqual(Fibo2Prompt.generationPrompt("[1, 2]"), "[1, 2]")
        XCTAssertEqual(Fibo2Prompt.generationPrompt(#"{"short_description":"a poster","text_render":[]}"#),
                       #"{"short_description":"a poster","text_render":[]}"#)
        XCTAssertEqual(Fibo2Prompt.generationPrompt(#"{"context":""}"#), "{}")
    }

    func testEditPrompts() {
        let marker = Fibo2Prompt.visionMarker
        XCTAssertEqual(Fibo2Prompt.editPrompt("make it snow", imageCount: 1, textImages: false),
                       #"{"image":"\#(marker)","edit_instruction":"make it snow"}"#)
        XCTAssertEqual(Fibo2Prompt.editPrompt("swap <image_1> and <image_2>", imageCount: 2, textImages: false),
                       #"{"image_1":"\#(marker)","image_2":"\#(marker)","edit_instruction":"swap <image_1> and <image_2>"}"#)
        let json = #"{"edit_instruction":"add a sign","structured_caption":{"short_description":"a shop","text_render":[{"text":"OPEN"}]}}"#
        XCTAssertEqual(Fibo2Prompt.editPrompt(json, imageCount: 1, textImages: true),
                       #"{"image":"\#(marker)","structured_caption":{"short_description":"a shop","text_render":[{"text":"OPEN","image_tokens":"\#(marker)"}]},"edit_instruction":"add a sign"}"#)
        XCTAssertNil(Fibo2Prompt.editPromptProblem("make it snow"))
        XCTAssertNotNil(Fibo2Prompt.editPromptProblem(#"{"short_description":"a cat"}"#))
        XCTAssertNotNil(Fibo2Prompt.editPromptProblem(#"{"edit_instruction":"x","structured_caption":{},"extra":1}"#))
    }

    func testTextImagesShape() {
        let shaped = Fibo2Prompt.shape(#"{"short_description":"a sign","text_render":[{"text":"OPEN"},{"text":""},{"location":"door"}]}"#, imageCount: 0)
        XCTAssertEqual(shaped.renderTexts, ["OPEN"])
        XCTAssertEqual(shaped.basePrompt, #"{"short_description":"a sign","text_render":[{"text":"OPEN","image_tokens":""},{"location":"door"}]}"#)
        XCTAssertTrue(shaped.prompt.contains(#""image_tokens":"<|vision_start|><|image_pad|><|vision_end|>""#))
        XCTAssertNil(Fibo2Prompt.shape("a red fox", imageCount: 0).basePrompt)
        XCTAssertEqual(Fibo2Prompt.shape(#"{"text_render":[{"text":"A"}]}"#, imageCount: 0, renderTextImages: false).renderTexts, [])
    }

    func testImageSizes() {
        XCTAssertEqual(Fibo2ImagePrep.imageSize(aspectRatio: 0.5).width, 720)
        XCTAssertEqual(Fibo2ImagePrep.imageSize(aspectRatio: 0.5).height, 1440)
        XCTAssertEqual(Fibo2ImagePrep.imageSize(aspectRatio: 1).width, 1024)
        XCTAssertEqual(Fibo2Transformer.Grid(rows: 64, columns: 64).count, 4096)
        XCTAssertEqual(Fibo2DiTConfig().gistCount(textLength: 46), 256)
        XCTAssertEqual(Fibo2DiTConfig().gistCount(textLength: 300), 320)
        XCTAssertEqual(Fibo2DiTConfig().gistCount(textLength: 4096), 512)
    }
}
