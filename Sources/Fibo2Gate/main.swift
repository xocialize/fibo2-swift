// Fibo2Gate — parity gates against mlxengine-image/WIP/fibo2-oracle goldens (diffusers 0.43 BriaFibo2Pipeline, fp32
// CPU), and an end-to-end renderer.
//
//   Fibo2Gate prompts  <goldens> <snapshot>            prompt shaping, image prep, token ids, pixel values (no weights)
//   Fibo2Gate textimg  <goldens> <snapshot>            text-image sizes and pixels, fallback-font advances
//   Fibo2Gate sizes    <goldens>                       trained size per aspect ratio
//   Fibo2Gate schedule <goldens>                       sigmas
//   Fibo2Gate encoder  <goldens> <snapshot> [fp32|bf16|int8]   Qwen3-VL bundles
//   Fibo2Gate dit      <goldens> <snapshot> [fp32|bf16|int8|int4]   one DiT forward, staged
//   Fibo2Gate vae      <goldens> <snapshot>            encode (packed) and decode
//   Fibo2Gate vision   <goldens> <snapshot>            bisect: Qwen3-VL vision tower stages (fp32)
//   Fibo2Gate gemm-probe                               fp32 GEMM accuracy on the GPU (TF32)
//
// The fp32 gates need MLX_ENABLE_TF32=0: on M5-class GPUs MLX runs fp32 matmuls in TF32 by default, which costs about
// 1e-3 per GEMM (encoder bundles with images 0.9993 instead of 1.000000).
//   Fibo2Gate render   <snapshot> --prompt <text|@file> [--negative <text|@file|card>] [--image p]… [--mask p]
//                      [--size WxH] [--steps n] [--guidance g] [--seed s] [--quant bf16|int8|int4] [--te bf16|int8]
//                      [--noise file.safetensors] [--dtype fp32|bf16] [--drop-te] [--shared <snapshot>] [--compare ref.png] [--no-text-images] --out out.png
//   Fibo2Gate matte    <RMBG-3.0 snapshot> --shared <edit-turbo or dev snapshot> --image p --out matte.png
//                      [--cutout cutout.png] [--rendered render.png] [--seed s] [--quant bf16|int8|int4] [--fit stretch|crop]

import Fibo2
import Flux2VAE
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Qwen3VL

setvbuf(stdout, nil, _IOLBF, 0)

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func die<T>(_ message: String) -> T { fail(message) }

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
    return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()) + 1e-20)).item(Float.self)
}

func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func jsonObject(_ url: URL) -> Any {
    guard let data = try? Data(contentsOf: url), let o = try? JSONSerialization.jsonObject(with: data) else {
        fail("cannot read \(url.path)")
    }
    return o
}

func imageDiff(_ a: Fibo2RGBImage, _ b: Fibo2RGBImage) -> (max: Int, mean: Double)? {
    guard a.width == b.width, a.height == b.height else { return nil }
    var m = 0, sum = 0
    for i in 0..<a.rgb.count {
        let d = abs(Int(a.rgb[i]) - Int(b.rgb[i]))
        m = max(m, d); sum += d
    }
    return (m, Double(sum) / Double(max(a.rgb.count, 1)))
}

func psnr(_ a: Fibo2RGBImage, _ b: Fibo2RGBImage) -> Double? {
    guard a.width == b.width, a.height == b.height else { return nil }
    var sum = 0.0
    for i in 0..<a.rgb.count { let d = Double(a.rgb[i]) - Double(b.rgb[i]); sum += d * d }
    let mse = sum / Double(a.rgb.count)
    return mse == 0 ? .infinity : 10 * log10(255 * 255 / mse)
}

/// The process's phys_footprint now and its lifetime peak, in GB (what a host's memory governor sees; MLX's own
/// counters miss load-time transients outside its pool).
func physFootprint() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    guard kr == KERN_SUCCESS else { return (0, 0) }
    return (Double(info.phys_footprint) / 1e9, Double(info.ledger_phys_footprint_peak) / 1e9)
}

func dtype(_ name: String) -> DType { name == "fp32" ? .float32 : .bfloat16 }

nonisolated(unsafe) var passed = true
func check(_ ok: Bool, _ line: String) {
    print((ok ? "PASS " : "FAIL ") + line)
    if !ok { passed = false }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail("usage: Fibo2Gate <prompts|textimg|sizes|schedule|encoder|dit|vae|render|matte> …") }

func goldensURL() -> URL { URL(fileURLWithPath: args.count > 1 ? args[1] : die("missing <goldens>")) }
func snapshotURL(_ index: Int = 2) -> URL { URL(fileURLWithPath: args.count > index ? args[index] : die("missing <snapshot>")) }

struct GoldenCase {
    var name: String
    var prompt: String
    var negative: String?
    var images: [String]
    var mask: String?
    var width: Int
    var height: Int
    var requestWidth: Int?
    var requestHeight: Int?
    var crops: [String]
    var calls: [(role: String, text: String, images: [String], tokens: Int)]
}

func goldenCases(_ goldens: URL) -> [GoldenCase] {
    let index = jsonObject(goldens.appendingPathComponent("prompts/index.json")) as! [String: [String: Any]]
    return index.keys.sorted().map { name in
        let entry = index[name]!
        let c = entry["case"] as! [String: Any]
        let calls = (entry["calls"] as! [[String: Any]]).map { call in
            (role: call["role"] as! String, text: call["text"] as! String,
             images: (call["images"] as! [[String: Any]]).map { $0["file"] as! String }, tokens: call["tokens"] as! Int)
        }
        return GoldenCase(
            name: name, prompt: c["prompt"] as! String, negative: c["negative_prompt"] as? String,
            images: c["image"] as? [String] ?? [], mask: c["mask"] as? String, width: entry["width"] as! Int,
            height: entry["height"] as! Int, requestWidth: c["width"] as? Int, requestHeight: c["height"] as? Int,
            crops: (entry["crops"] as? [[String: Any]] ?? []).map { $0["file"] as! String }, calls: calls)
    }
}

func readImage(_ url: URL) -> Fibo2RGBImage {
    guard let image = try? Fibo2RGBImage.read(url: url) else { fail("cannot read image \(url.path)") }
    return image
}

func goldenInputs(_ tensors: [String: MLXArray], role: String) -> Fibo2Processor.Inputs {
    let ids = tensors["\(role).input_ids"]!.asArray(Int32.self)
    var grids: [THW] = []
    if let thw = tensors["\(role).image_grid_thw"] {
        let v = thw.asArray(Int32.self)
        for k in 0..<(v.count / 3) { grids.append(THW(Int(v[k * 3]), Int(v[k * 3 + 1]), Int(v[k * 3 + 2]))) }
    }
    return Fibo2Processor.Inputs(ids: ids, pixelValues: tensors["\(role).pixel_values"], grids: grids)
}

switch command {

case "prompts":
    let goldens = goldensURL(), snapshot = snapshotURL()
    let inputsDir = goldens.deletingLastPathComponent().appendingPathComponent("inputs")
    let processor = try await Fibo2Processor.load(snapshot: snapshot)
    let renderer = try Fibo2TextRenderer(fontsDirectory: snapshot.appendingPathComponent("fonts"))
    for c in goldenCases(goldens) {
        let dir = goldens.appendingPathComponent("prompts")
        let tensors = try MLX.loadArrays(url: dir.appendingPathComponent("\(c.name).safetensors"))
        let images = c.images.map { readImage(inputsDir.appendingPathComponent($0)) }
        let shaped = Fibo2Prompt.shape(c.prompt, imageCount: images.count)
        var texts: [String: String] = ["prompt": Fibo2Prompt.template(shaped.prompt),
                                       "negative": Fibo2Prompt.template(c.negative ?? Fibo2Prompt.defaultNegative)]
        if let base = shaped.basePrompt { texts["base"] = Fibo2Prompt.template(base) }
        check(Set(texts.keys) == Set(c.calls.map(\.role)), "\(c.name): calls \(texts.keys.sorted()) vs \(c.calls.map(\.role))")
        for call in c.calls {
            check(texts[call.role] == call.text, "\(c.name).\(call.role): shaped text")
            if texts[call.role] != call.text { print("   swift: \(texts[call.role] ?? "-")\n   ref:   \(call.text)") }
        }
        // Image prep: the VAE crops and Qwen3-VL's copies, byte for byte.
        var encoderImages: [Fibo2RGBImage] = []
        if !images.isEmpty {
            var mask: [UInt8]? = nil
            if let m = c.mask { mask = Fibo2ImagePrep.luma(readImage(inputsDir.appendingPathComponent(m))) }
            let prepared = Fibo2ImagePrep.prepare(images: images, mask: mask, width: c.requestWidth, height: c.requestHeight)
            check(prepared.width == c.width && prepared.height == c.height,
                  "\(c.name): output size \(prepared.width)×\(prepared.height) vs \(c.width)×\(c.height)")
            for (k, file) in c.crops.enumerated() {
                let d = imageDiff(prepared.crops[k], readImage(dir.appendingPathComponent(file)))
                check(d?.max == 0, "\(c.name): VAE crop \(k) max diff \(d.map { "\($0.max)" } ?? "size mismatch")")
            }
            encoderImages = prepared.textEncoderImages
        }
        let promptCall = c.calls.first { $0.role == "prompt" }!
        let goldenImages = promptCall.images.map { readImage(dir.appendingPathComponent($0)) }
        for (k, image) in encoderImages.enumerated() {
            let d = imageDiff(image, goldenImages[k])
            check(d?.max == 0, "\(c.name): Qwen3-VL image \(k) max diff \(d.map { "\($0.max)" } ?? "size mismatch")")
        }
        let textImages = shaped.renderTexts.map(renderer.draw)
        check(textImages.count == goldenImages.count - encoderImages.count, "\(c.name): \(textImages.count) text images")
        for (k, image) in textImages.enumerated() where encoderImages.count + k < goldenImages.count {
            let g = goldenImages[encoderImages.count + k]
            let d = imageDiff(image, g)
            check(d != nil, "\(c.name): text image \(k) size \(image.width)×\(image.height) vs \(g.width)×\(g.height)"
                  + (d.map { String(format: ", mean abs diff %.2f", $0.mean) } ?? ""))
        }
        // The processor on the reference's own images: ids and pixel values.
        for call in c.calls {
            let refImages = call.images.map { readImage(dir.appendingPathComponent($0)) }
            let body = String(call.text.dropFirst("<|im_start|>user\n".count).dropLast("<|im_end|>\n<|im_start|>assistant\n".count))
            let inputs = try processor(prompt: body, images: refImages)
            let golden = goldenInputs(tensors, role: call.role)
            check(inputs.ids == golden.ids, "\(c.name).\(call.role): \(inputs.ids.count) token ids vs \(golden.ids.count)")
            if let pv = inputs.pixelValues, let g = golden.pixelValues {
                let same = pv.shape == g.shape
                check(same && maxAbs(pv, g) < 2e-2, "\(c.name).\(call.role): pixel values \(pv.shape) vs \(g.shape)"
                      + (same ? String(format: ", max abs diff %.4f, cos %.6f", maxAbs(pv, g), cosine(pv, g)) : ""))
            }
            // Swift-drawn text images through the processor: the token count has to match the reference's.
            if call.role == "prompt", !textImages.isEmpty {
                let own = try processor(prompt: body, images: Array(refImages.prefix(encoderImages.count)) + textImages)
                check(own.ids == golden.ids, "\(c.name).prompt: token ids with Swift-drawn text images")
            }
        }
    }

case "textimg":
    let goldens = goldensURL(), snapshot = snapshotURL()
    let dir = goldens.appendingPathComponent("textimg")
    let renderer = try Fibo2TextRenderer(fontsDirectory: snapshot.appendingPathComponent("fonts"))
    let facts = jsonObject(dir.appendingPathComponent("fontfacts.json")) as! [String: Any]
    var cases = (jsonObject(dir.appendingPathComponent("index.json")) as! [[String: Any]])
    cases += facts["random"] as! [[String: Any]]
    var sizeMismatch = 0, total = 0.0
    for c in cases {
        let text = c["text"] as! String, size = c["size"] as! [Int]
        let image = renderer.draw(text)
        let golden = readImage(dir.appendingPathComponent(c["file"] as! String))
        if let d = imageDiff(image, golden) {
            total += d.mean
        } else {
            sizeMismatch += 1
            print("   size \(image.width)×\(image.height) vs \(size[0])×\(size[1]): \(text.prefix(60))")
        }
    }
    check(sizeMismatch == 0, "text images: \(cases.count - sizeMismatch)/\(cases.count) sizes equal, mean abs pixel diff "
          + String(format: "%.2f / 255", total / Double(max(cases.count - sizeMismatch, 1))))
    let report = renderer.advanceReport(facts["advances"] as! [String: [String: Double]])
    for (font, r) in report.sorted(by: { $0.key < $1.key }) {
        check(r.mismatched == 0, "\(font): \(r.checked - r.mismatched)/\(r.checked) hinted advances equal"
              + (r.examples.isEmpty ? "" : " — e.g. \(r.examples.prefix(6).joined(separator: " "))"))
    }
    if args.count > 3 {
        let out = URL(fileURLWithPath: args[3])
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        for c in cases { try renderer.draw(c["text"] as! String).write(to: out.appendingPathComponent(c["file"] as! String)) }
    }

case "sizes":
    let cases = jsonObject(goldensURL().appendingPathComponent("sizes.json")) as! [[String: Any]]
    var bad = 0
    for c in cases {
        let (w, h, area, size) = (c["w"] as! Int, c["h"] as! Int, c["area"] as! Int, c["size"] as! [Int])
        let got = Fibo2ImagePrep.imageSize(aspectRatio: Double(w) / Double(h), area: area)
        if got.width != size[0] || got.height != size[1] {
            bad += 1
            print("   \(w)×\(h) area \(area): \(got.width)×\(got.height) vs \(size[0])×\(size[1])")
        }
    }
    check(bad == 0, "image sizes: \(cases.count - bad)/\(cases.count) equal")

case "schedule":
    let goldens = goldensURL()
    let tensors = try MLX.loadArrays(url: goldens.appendingPathComponent("schedule.safetensors"))
    let meta = (jsonObject(goldens.appendingPathComponent("schedule.json")) as! [String: Any])["cases"] as! [[String: Any]]
    for c in meta {
        let key = c["key"] as! String
        let (steps, h, w) = (c["steps"] as! Int, c["height"] as! Int, c["width"] as! Int)
        let sigmas = Fibo2Schedule.sigmas(steps: steps, imageTokens: (h / 16) * (w / 16))
        let d = maxAbs(MLXArray(sigmas), tensors["\(key).sigmas"]!)
        check(d <= 1.2e-7, "\(key): sigmas max abs diff \(d)")
    }

case "encoder":
    let goldens = goldensURL(), snapshot = snapshotURL()
    let mode = args.count > 3 ? args[3] : "fp32"
    let encoder = try await Fibo2TextEncoder.load(snapshot: snapshot, dtype: dtype(mode), quant: mode == "int8" ? .int8 : .bf16)
    // bf16: the reference's own bf16 run (torch, bf16_floor.py) is 0.987–0.997 from fp32 on prompts with images.
    let floor: Float = mode == "fp32" ? 0.9999 : (mode == "bf16" ? 0.99 : 0.98)
    let dir = goldens.appendingPathComponent("encoder")
    for file in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        .filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        let name = file.deletingPathExtension().lastPathComponent
        let golden = try MLX.loadArrays(url: file)
        let inputs = try MLX.loadArrays(url: goldens.appendingPathComponent("prompts/\(name).safetensors"))
        for role in ["prompt", "base", "negative"] where golden["\(role).bundles"] != nil {
            let out = try encoder.bundles(goldenInputs(inputs, role: role))
            let ref = golden["\(role).bundles"]!
            guard out.shape == ref.shape else { check(false, "\(name).\(role): shape \(out.shape) vs \(ref.shape)"); continue }
            let perBundle = (0..<6).map { cosine(out[0..., $0], ref[0..., $0]) }
            if ProcessInfo.processInfo.environment["FIBO2_GATE_TOKENS"] != nil {
                // Raw hidden states of the reference's dumped layers: where does a deviation start?
                let dumped = golden.keys.compactMap { k -> Int? in k.hasPrefix("\(role).hidden.") ? Int(k.split(separator: ".").last!) : nil }.sorted()
                if !dumped.isEmpty {
                    let gi = goldenInputs(inputs, role: role)
                    let states = try encoder.model.hiddenStates(
                        inputIds: MLXArray(gi.ids).reshaped(1, gi.ids.count), layers: dumped, pixelValues: gi.pixelValues,
                        imageGridTHW: gi.grids.isEmpty ? nil : gi.grids)
                    for (layer, state) in zip(dumped, states) {
                        let a = state[0].asType(.float32), b = golden["\(role).hidden.\(layer)"]![0].asType(.float32)
                        let c = ((a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))).asArray(Float.self)
                        let w = c.enumerated().min { $0.element < $1.element }!
                        print(String(format: "   hidden[%d]: cos %.6f, tokens < 0.999: %d, worst %d:%.4f, max abs diff %.4f (ref max %.1f)",
                                     layer, cosine(a, b), c.filter { $0 < 0.999 }.count, w.offset, w.element, maxAbs(a, b), abs(b).max().item(Float.self)))
                    }
                }
                // Per-token cosine of the Perceiver's bundle, split into image-pad tokens and the rest.
                let ids = inputs["\(role).input_ids"]!.asArray(Int32.self)
                let a = out[0, 0].asType(.float32), b = ref[0, 0].asType(.float32)
                let cosTok = ((a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))).asArray(Float.self)
                let img = zip(ids, cosTok).filter { $0.0 == 151655 }.map(\.1), txt = zip(ids, cosTok).filter { $0.0 != 151655 }.map(\.1)
                func stats(_ v: [Float]) -> String { v.isEmpty ? "-" : String(format: "n %d mean %.6f min %.6f", v.count, v.reduce(0, +) / Float(v.count), v.min()!) }
                print("   per-token cos (bundle 0): image tokens \(stats(img)); other tokens \(stats(txt))")
                let worst = cosTok.enumerated().sorted { $0.element < $1.element }.prefix(12)
                print("   worst tokens (index:cos): " + worst.map { "\($0.offset):" + String(format: "%.4f", $0.element) }.joined(separator: " "))
                let below = cosTok.filter { $0 < 0.999 }.count
                print("   tokens below 0.999: \(below) of \(cosTok.count)")
                let firstImg = ids.firstIndex(of: 151655) ?? 0
                print("   first image token at \(firstImg); cos of tokens before it: \(stats(Array(cosTok.prefix(firstImg))))")
            }
            check(perBundle.min()! >= floor, "\(name).\(role): bundles \(out.shape) cos min "
                  + String(format: "%.6f", perBundle.min()!) + " [" + perBundle.map { String(format: "%.5f", $0) }.joined(separator: " ") + "]")
        }
    }

case "vision":
    // Bisect: the vision tower's stages against the oracle's (vision_stages.py), fp32.
    let goldens = goldensURL(), snapshot = snapshotURL()
    let model = try Qwen3VLLoader.load(directory: snapshot.appendingPathComponent("text_encoder"), dtype: .float32)
    for name in ["t2i_text", "edit_instr"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("vision/\(name).safetensors"))
        let gi = goldenInputs(try MLX.loadArrays(url: goldens.appendingPathComponent("prompts/\(name).safetensors")), role: "prompt")
        func report(_ label: String, _ a: MLXArray, _ b: MLXArray) {
            let x = a.asType(.float32), y = b.asType(.float32)
            let c = ((x * y).sum(axis: -1) / (sqrt((x * x).sum(axis: -1)) * sqrt((y * y).sum(axis: -1)))).asArray(Float.self)
            let w = c.enumerated().min { $0.element < $1.element }!
            print(String(format: "%@.%@ %@: cos %.6f, rows < 0.999: %d, worst %d:%.4f, max abs diff %.4f (ref max %.1f)", name, label,
                         "\(a.shape)", cosine(a, b), c.filter { $0 < 0.999 }.count, w.offset, w.element, maxAbs(a, b), abs(y).max().item(Float.self)))
        }
        let pre = model.debugVisionPre(pixelValues: gi.pixelValues!, imageGridTHW: gi.grids)
        report("patch", pre.patch, g["patch"]!)
        if let pos = g["pos"] {
            report("pos", pre.pos, pos)
            // The rotary table: Swift returns angles [N, 32] (h | w); the reference cos/sin are [N, 64] (h, w, h, w).
            report("rope.cos", cos(concatenated([pre.rot, pre.rot], axis: -1)), g["rope.cos"]!)
            report("rope.sin", sin(concatenated([pre.rot, pre.rot], axis: -1)), g["rope.sin"]!)
        }
        let blocks = try model.debugVisionBlocks(pixelValues: gi.pixelValues!, imageGridTHW: gi.grids)
        for k in [0, 1, 5, 11, 17, 23] { report("block.\(k)", blocks[k], g["block.\(k)"]!) }
        let d = try model.debugEdit(inputIds: MLXArray(gi.ids).reshaped(1, gi.ids.count), pixelValues: gi.pixelValues!, imageGridTHW: gi.grids)
        report("merged", d.visHidden, g["merged"]!)
        for (k, deep) in d.deepstack.enumerated() { report("deepstack.\(k)", deep, g["deepstack.\(k)"]!) }
    }

case "gemm-probe":
    // How exact is one fp32 GEMM on this machine's GPU? MLX runs fp32 matmuls in TF32 on NAX (M5-class) GPUs unless
    // MLX_ENABLE_TF32=0 (MLXEngine R-NUM-1): products with a 10-bit mantissa, about 3e-4 of the largest value. The fp32
    // parity gates here need MLX_ENABLE_TF32=0; production runs bf16, where it changes nothing.
    let (x, w) = (MLXRandom.normal([1152, 1024], key: MLXRandom.key(1)), MLXRandom.normal([3072, 1024], key: MLXRandom.key(2)) * 0.05)
    eval(x, w)
    let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
        let r = matmul(x.asType(.float64), w.asType(.float64).T)
        eval(r)
        return r
    }
    func diff(_ a: MLXArray) -> Double { Device.withDefaultDevice(.cpu) { abs(a.asType(.float64) - ref).max().item(Double.self) } }
    let gpu = matmul(x, w.T)
    eval(gpu)
    let cpu = Device.withDefaultDevice(.cpu) { () -> MLXArray in let r = matmul(x, w.T); eval(r); return r }
    let refMax = Device.withDefaultDevice(.cpu) { abs(ref).max().item(Double.self) }
    print(String(format: "x %@ · Wᵀ %@ against float64, ref max %.2f (MLX_ENABLE_TF32=%@)", "\(x.shape)", "\(w.shape)", refMax,
                 ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset"))
    print(String(format: "GPU fp32 matmul: max abs diff %.6f", diff(gpu)))
    print(String(format: "CPU fp32 matmul: max abs diff %.6f", diff(cpu)))

case "dit":
    let goldens = goldensURL(), snapshot = snapshotURL()
    let mode = args.count > 3 ? args[3] : "fp32"
    let quant = Fibo2Quant(rawValue: mode) ?? .bf16
    let model = try Fibo2Weights.loadTransformer(snapshot: snapshot, dtype: dtype(mode), quant: quant)
    let floor: Float = mode == "fp32" ? 0.9999 : (mode == "bf16" ? 0.998 : (mode == "int8" ? 0.99 : 0.9))
    // FIBO2_DIT_GOLDENS: the golden folder under <goldens> (default "dit"; "dit-turbo" for Edit Turbo's weights).
    let ditFolder = ProcessInfo.processInfo.environment["FIBO2_DIT_GOLDENS"] ?? "dit"
    for name in ["t2i", "edit"] {
        let url = goldens.appendingPathComponent("\(ditFolder)/\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { continue }
        let g = try MLX.loadArrays(url: url)
        let x = g["in.x"]!
        let grid = Fibo2Transformer.Grid(rows: x.dim(2) / 2, columns: x.dim(3) / 2)
        var context: MLXArray? = nil
        var grids: [Fibo2Transformer.Grid] = []
        if let c = g["in.context0"] {
            context = Fibo2Pipeline.tokens(fromLatents: c)
            grids = [.init(rows: c.dim(2) / 2, columns: c.dim(3) / 2)]
        }
        let conditioning = model.condition(g["in.enc"]!)
        let layout = model.layout(image: grid, context: grids, gistCount: conditioning.gistCount)
        var stages: [String: MLXArray] = [:]
        let v = model(image: Fibo2Pipeline.tokens(fromLatents: x), timestep: g["in.t"]!.item(Float.self),
                      conditioning: conditioning, context: context, layout: layout) { stages[$0] = $1 }
        for key in g.keys.filter({ $0.hasPrefix("stage.") }).sorted(by: { a, b in
            func order(_ k: String) -> (Int, Int) {
                let p = k.split(separator: ".")
                let rank = ["x_embed": 0, "refiner": 1, "gist": 2, "layer": 3][String(p[1])] ?? 9
                return (rank, p.count > 2 ? Int(p[2]) ?? 0 : 0)
            }
            return order(a) < order(b)
        }) {
            let stage = String(key.dropFirst("stage.".count))
            guard let out = stages[stage] else { check(false, "\(name).\(stage): not captured"); continue }
            let c = cosine(out, g[key]!)
            check(out.shape == g[key]!.shape && c >= floor, "\(name).\(stage) \(out.shape): cos " + String(format: "%.6f", c))
        }
        let ref = Fibo2Pipeline.tokens(fromLatents: g["out.v"]!)
        let c = cosine(v, ref)
        check(c >= floor, "\(name).velocity \(v.shape): cos " + String(format: "%.6f", c)
              + String(format: ", max abs diff %.4f (ref max %.2f)", maxAbs(v, ref), abs(ref).max().item(Float.self)))
    }

case "vae":
    let goldens = goldensURL(), snapshot = snapshotURL()
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("vae.safetensors"))
    let dir = snapshot.appendingPathComponent("vae")
    let encoder = try Flux2VAEWeights.loadEncoder(directory: dir, dtype: .float32)
    let vae = try Flux2VAEWeights.loadVAE(directory: dir, dtype: .float32)
    let image = readImage(goldens.appendingPathComponent("vae_input.png"))
    let pixels = image.pixelsNCHW()
    check(maxAbs(pixels, g["in.pixels"]!) < 1e-6, "pixels \(pixels.shape): max abs diff \(maxAbs(pixels, g["in.pixels"]!))")
    for (label, tile) in [("untiled", Flux2VAETile?.none), ("row bands", .standard)] {
        let packed = tile.map { encoder.encodePacked(pixels, tile: $0) } ?? encoder.encodePacked(pixels)
        let c = cosine(packed, g["out.packed"]!)
        check(c >= 0.9999, "encode (\(label)) \(packed.shape): cos " + String(format: "%.6f", c)
              + String(format: ", max abs diff %.4f", maxAbs(packed, g["out.packed"]!)))
        let decoded = tile.map { vae.decodePackedLatents(g["out.packed"]!, tile: $0) } ?? vae.decodePackedLatents(g["out.packed"]!)
        let dc = cosine(decoded, g["out.decoded"]!)
        check(dc >= 0.9999, "decode (\(label)) \(decoded.shape): cos " + String(format: "%.6f", dc)
              + String(format: ", max abs diff %.4f", maxAbs(decoded, g["out.decoded"]!)))
        if let p = psnr(Fibo2RGBImage.fromPixels(decoded[0]), readImage(goldens.appendingPathComponent("vae_decoded.png"))) {
            print(String(format: "   decoded PNG vs reference: %.1f dB", p))
        }
    }
    let tokens = Fibo2Pipeline.tokens(fromPacked: g["out.packed"]!)
    let viaLatents = Fibo2Pipeline.tokens(fromLatents: g["out.latents"]!)
    check(maxAbs(tokens, viaLatents) < 1e-6, "packed tokens == patchified 32-channel latents (max abs diff \(maxAbs(tokens, viaLatents)))")

case "render":
    let snapshot = snapshotURL(1)
    var request = Fibo2Pipeline.Request(prompt: "")
    var quant = Fibo2Quant.bf16, te = Fibo2TextEncoderQuant.bf16
    var out: String? = nil, noiseFile: String? = nil, compare: String? = nil
    var renderDType = DType.bfloat16
    // The engine package's memory path: no resident text encoder; it loads per request and is released before denoising.
    var dropEncoder = false
    // Where text_encoder/, processor/, vae/ and fonts/ are read from, when the snapshot holds a transformer only.
    var shared: URL? = nil
    func text(_ v: String) -> String {
        guard v.hasPrefix("@") else { return v }
        guard let s = try? String(contentsOfFile: String(v.dropFirst()), encoding: .utf8) else { fail("cannot read \(v)") }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var i = 2
    while i < args.count {
        let flag = args[i]
        func value() -> String { i += 1; return i < args.count ? args[i] : die("\(flag) needs a value") }
        switch flag {
        case "--prompt": request.prompt = text(value())
        case "--negative": let v = value(); request.negativePrompt = v == "card" ? Fibo2Prompt.cardNegativeT2I : text(v)
        case "--image": request.images.append(readImage(URL(fileURLWithPath: value())))
        case "--mask": request.mask = Fibo2ImagePrep.luma(readImage(URL(fileURLWithPath: value())))
        case "--size":
            let p = value().split(separator: "x").compactMap { Int($0) }
            guard p.count == 2 else { fail("--size WxH") }
            (request.width, request.height) = (p[0], p[1])
        case "--steps": request.steps = Int(value()) ?? die("--steps n")
        case "--guidance": request.guidance = Float(value()) ?? die("--guidance g")
        case "--seed": request.seed = UInt64(value()) ?? die("--seed s")
        case "--quant": quant = Fibo2Quant(rawValue: value()) ?? die("--quant bf16|int8|int4")
        case "--te": te = Fibo2TextEncoderQuant(rawValue: value()) ?? die("--te bf16|int8")
        case "--shared": shared = URL(fileURLWithPath: value())
        case "--drop-te": dropEncoder = true
        case "--dtype": renderDType = dtype(value())
        case "--compare": compare = value()
        case "--noise": noiseFile = value()
        case "--no-text-images": request.renderTextImages = false
        case "--out": out = value()
        default: fail("unknown option \(flag)")
        }
        i += 1
    }
    guard let out, !request.prompt.isEmpty else { fail("render needs --prompt and --out") }
    var clock = Date()
    if dropEncoder { MLX.Memory.cacheLimit = 2 << 30 }  // the engine's pool cap
    let pipeline = try await Fibo2Pipeline.load(snapshot: snapshot, shared: shared, dtype: renderDType, quant: quant,
                                                textEncoderQuant: dropEncoder ? nil : te)
    MLX.Memory.clearCache()
    let resident = MLX.Memory.activeMemory
    print(String(format: "loaded in %.1f s (DiT %@, encoder %@%@), resident %.2f GB", Date().timeIntervalSince(clock),
                 quant.rawValue, te.rawValue, dropEncoder ? ", per request" : "", Double(resident) / 1e9))
    print(String(format: "  phys_footprint after load: %.2f GB now, %.2f GB peak (the load transient)", physFootprint().now, physFootprint().peak))
    var noise: MLXArray? = nil
    if let noiseFile { noise = try MLX.loadArrays(url: URL(fileURLWithPath: noiseFile))["noise"] }
    clock = Date()
    MLX.Memory.peakMemory = 0
    var encoded: Fibo2Pipeline.Encoded? = nil
    if dropEncoder {
        // The encoder lives only inside this scope.
        encoded = try await { () async throws -> Fibo2Pipeline.Encoded in
            let encoder = try await Fibo2TextEncoder.load(snapshot: shared ?? snapshot, dtype: renderDType, quant: te)
            return try pipeline.encode(request, encoder: encoder)
        }()
        MLX.Memory.clearCache()
        print(String(format: "  encode phase: %.1f s, peak %.2f GB (+%.2f over resident)", Date().timeIntervalSince(clock),
                     Double(MLX.Memory.peakMemory) / 1e9, Double(MLX.Memory.peakMemory - resident) / 1e9))
        print(String(format: "  phys_footprint after encode: %.2f GB now, %.2f GB peak", physFootprint().now, physFootprint().peak))
        MLX.Memory.peakMemory = 0
    }
    let result = try pipeline.generate(request, encoded: encoded, noise: noise) { step, steps in
        if step == 1 || step % 10 == 0 || step == steps {
            print(String(format: "  step %d/%d  %.1f s", step, steps, Date().timeIntervalSince(clock)))
        }
    }
    if dropEncoder {
        print(String(format: "  denoise + decode: peak %.2f GB (+%.2f over resident)", Double(MLX.Memory.peakMemory) / 1e9,
                     Double(MLX.Memory.peakMemory - resident) / 1e9))
    }
    print(String(format: "  phys_footprint at the end: %.2f GB now, %.2f GB peak", physFootprint().now, physFootprint().peak))
    try result.image.write(to: URL(fileURLWithPath: out))
    let t = result.timings
    print(String(format: "%d×%d, %d steps, guidance %.1f: encode %.1f s, gist %.1f s, denoise %.1f s, decode %.1f s; peak %.2f GB",
                 result.width, result.height, request.steps, request.guidance, t["encode"] ?? 0, t["gist"] ?? 0,
                 t["denoise"] ?? 0, t["decode"] ?? 0, Double(MLX.Memory.peakMemory) / 1e9))
    print("prompt as encoded: \(result.shapedPrompt.prefix(300))")
    print("wrote \(out)")
    if let compare, let p = psnr(result.image, readImage(URL(fileURLWithPath: compare))) {
        print(String(format: "PSNR against %@: %.2f dB", compare, p))
    }

case "matte":
    // RMBG 3.0 end to end, on the engine package's memory path (the encoder loads per request).
    let snapshot = snapshotURL(1)
    var shared: URL? = nil, image: String? = nil, out: String? = nil, cutout: String? = nil, rendered: String? = nil
    var quant = Fibo2Quant.bf16, seed = Fibo2Matte.defaultSeed, fit = Fibo2Matte.Fit.stretch
    var i = 2
    while i < args.count {
        let flag = args[i]
        func value() -> String { i += 1; return i < args.count ? args[i] : die("\(flag) needs a value") }
        switch flag {
        case "--shared": shared = URL(fileURLWithPath: value())
        case "--image": image = value()
        case "--out": out = value()
        case "--cutout": cutout = value()
        case "--rendered": rendered = value()
        case "--seed": seed = UInt64(value()) ?? die("--seed s")
        case "--quant": quant = Fibo2Quant(rawValue: value()) ?? die("--quant bf16|int8|int4")
        case "--fit": fit = Fibo2Matte.Fit(rawValue: value()) ?? die("--fit stretch|crop")
        default: fail("unknown option \(flag)")
        }
        i += 1
    }
    guard let image, let out else { fail("matte needs --image and --out") }
    let source = readImage(URL(fileURLWithPath: image))
    let request = Fibo2Matte.request(for: source, seed: seed, fit: fit)
    var clock = Date()
    MLX.Memory.cacheLimit = 2 << 30  // the engine's pool cap
    let pipeline = try await Fibo2Pipeline.load(snapshot: snapshot, shared: shared, quant: quant, textEncoderQuant: nil)
    MLX.Memory.clearCache()
    let resident = MLX.Memory.activeMemory
    print(String(format: "loaded in %.1f s (DiT %@), resident %.2f GB; phys_footprint %.2f GB now, %.2f GB peak",
                 Date().timeIntervalSince(clock), quant.rawValue, Double(resident) / 1e9, physFootprint().now, physFootprint().peak))
    clock = Date()
    MLX.Memory.peakMemory = 0
    let encoded = try await { () async throws -> Fibo2Pipeline.Encoded in
        let encoder = try await Fibo2TextEncoder.load(snapshot: shared ?? snapshot, quant: .bf16)
        return try pipeline.encode(request, encoder: encoder)
    }()
    MLX.Memory.clearCache()
    print(String(format: "  encode phase: %.1f s, peak %.2f GB (+%.2f over resident); phys_footprint %.2f GB now, %.2f GB peak",
                 Date().timeIntervalSince(clock), Double(MLX.Memory.peakMemory) / 1e9,
                 Double(MLX.Memory.peakMemory - resident) / 1e9, physFootprint().now, physFootprint().peak))
    MLX.Memory.peakMemory = 0
    let result = try pipeline.generate(request, encoded: encoded)
    print(String(format: "  denoise + decode: peak %.2f GB (+%.2f over resident); phys_footprint %.2f GB now, %.2f GB peak",
                 Double(MLX.Memory.peakMemory) / 1e9, Double(MLX.Memory.peakMemory - resident) / 1e9,
                 physFootprint().now, physFootprint().peak))
    let resize = Date()
    let matte = Fibo2Matte.matte(from: result.image, width: source.width, height: source.height)
    try matte.write(to: URL(fileURLWithPath: out))
    if let rendered { try result.image.write(to: URL(fileURLWithPath: rendered)) }
    if let cutout { try Fibo2PNG.write(Fibo2Matte.cutout(source, matte: matte), to: URL(fileURLWithPath: cutout)) }
    // A matte is grey: a coloured render means the instruction or the weights are wrong.
    var chroma = 0
    for p in 0..<(result.width * result.height) {
        let (r, g, b) = (Int(result.image.rgb[p * 3]), Int(result.image.rgb[p * 3 + 1]), Int(result.image.rgb[p * 3 + 2]))
        chroma = max(chroma, max(r, g, b) - min(r, g, b))
    }
    let t = result.timings
    print(String(format: "%d×%d picture, rendered at %d×%d (%@), seed %llu: denoise %.1f s, decode %.1f s, resize %.1f s, %.1f s in all",
                 source.width, source.height, result.width, result.height, fit.rawValue, seed, t["denoise"] ?? 0,
                 t["decode"] ?? 0, Date().timeIntervalSince(resize), Date().timeIntervalSince(clock)))
    print(String(format: "matte: %.1f %% of the frame at 128 or above; largest R/G/B spread in the render %d of 255", matte.coverage * 100, chroma))
    print("wrote \(out)")

default:
    fail("unknown command \(command)")
}

if !passed { exit(1) }
