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
        let root = "/Volumes/DEV_VOL1/Models/models--briaai--Fibo-2.0-dev/snapshots/" + Fibo2Configuration.revision
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
        // The measured worst request (an edit's encode phase, 10.89 GB) is under the declared activation.
        XCTAssertTrue(fp.allSatisfy { $0.peakActivationBytes >= 10_890_000_000 })
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
        let c = Fibo2Configuration(quant: .int8, textEncoderQuant: .int8, defaultSteps: 30, keepEncoderResident: true)
        let back = try JSONDecoder().decode(Fibo2Configuration.self, from: JSONEncoder().encode(c))
        XCTAssertEqual(back.quant, .int8)
        XCTAssertEqual(back.textEncoderQuant, .int8)
        XCTAssertEqual(back.defaultSteps, 30)
        XCTAssertTrue(back.keepEncoderResident)
        let old = try JSONDecoder().decode(Fibo2Configuration.self, from: Data(#"{"quant":"bf16"}"#.utf8))
        XCTAssertEqual(old.defaultSteps, 50)
        XCTAssertFalse(old.keepEncoderResident)
        XCTAssertNil(old.residentBytesHint)
        XCTAssertGreaterThan(c.residentBytesHint!, Fibo2Footprints.declared.first { $0.quant == .int8 }!.residentBytes)
    }

    func testMetaLevers() throws {
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
