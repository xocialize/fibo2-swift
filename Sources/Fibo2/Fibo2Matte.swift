// RMBG 3.0 (briaai/RMBG-3.0): Fibo 2.0's editor fine-tuned and distilled to render an alpha matte in 4 steps. The
// repo holds a transformer only; everything else is Fibo 2.0 Edit Turbo's. Background removal is one edit with a
// fixed instruction, so this file is the whole port: the instruction, the settings, and the picture's way in and out.
//
// One difference from the model card's snippet, on purpose. The card calls the pipeline without a size, which
// centre-crops the picture to the trained size for its aspect ratio, then stretches the matte over the whole
// picture: the strips the crop cut off (up to 8 px of the model's ~1024) get no matte, and the rest lands slightly
// off. Here the picture is stretched to that same trained size instead (at most the same few pixels of distortion),
// so the matte covers the whole frame and stretches back exactly. `Fibo2Matte.Fit.crop` is the card's way, for
// comparison.

import Foundation

public enum Fibo2Matte {
    /// The instruction RMBG 3.0 was trained on, and nothing else: a changed or missing one gives a coloured picture.
    public static let instruction = "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the "
        + "background to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent "
        + "fine details and transparency."

    /// The card's `json.dumps({"edit_instruction": INSTRUCTION})`.
    public static var prompt: String { Fibo2JSON.object(Fibo2JSONObject([("edit_instruction", .string(instruction))])).dumps() }

    /// The card's settings: 4 steps, no guidance, seed 0.
    public static let steps = Fibo2Schedule.turboSteps
    public static let guidance = Fibo2Schedule.turboGuidance
    public static let defaultSeed: UInt64 = 0

    public enum Fit: String, Sendable {
        /// The whole picture, stretched to the trained size for its aspect ratio.
        case stretch
        /// The model card's snippet: centre-cropped to that size.
        case crop
    }

    /// The edit that renders `image`'s matte, at the size Fibo 2.0 was trained at for its aspect ratio.
    public static func request(for image: Fibo2RGBImage, seed: UInt64 = defaultSeed, fit: Fit = .stretch) -> Fibo2Pipeline.Request {
        let size = Fibo2ImagePrep.imageSize(aspectRatio: Double(image.width) / Double(image.height))
        let input = fit == .stretch ? image.resized(width: size.width, height: size.height, filter: .lanczos) : image
        return Fibo2Pipeline.Request(prompt: prompt, images: [input], width: size.width, height: size.height, steps: steps,
                                     guidance: guidance, seed: seed)
    }

    /// The rendered picture as a matte the size of the source: `convert("L")`, then LANCZOS. White is the subject.
    /// `binary` thresholds it at 128.
    public static func matte(from rendered: Fibo2RGBImage, width: Int, height: Int, binary: Bool = false) -> Fibo2GrayImage {
        var grey = Fibo2ImagePrep.luma(rendered)
        if rendered.width != width || rendered.height != height {
            grey = Fibo2PILResize.resampleBands(grey, bands: 1, width: rendered.width, height: rendered.height,
                                                outWidth: width, outHeight: height, filter: .lanczos)
        }
        if binary { grey = grey.map { $0 >= 128 ? 255 : 0 } }
        return Fibo2GrayImage(pixels: grey, width: width, height: height)
    }

    /// The card's `putalpha`: the source with the matte as its alpha.
    public static func cutout(_ image: Fibo2RGBImage, matte: Fibo2GrayImage) -> Fibo2RGBAImage {
        precondition(matte.width == image.width && matte.height == image.height, "the matte has to be the image's size")
        var out = image.rgba()
        for i in 0..<matte.pixels.count { out.rgba[i * 4 + 3] = matte.pixels[i] }
        return out
    }
}

/// 8-bit grey.
public struct Fibo2GrayImage: Sendable, Equatable {
    public var pixels: [UInt8]
    public var width: Int
    public var height: Int

    public init(pixels: [UInt8], width: Int, height: Int) {
        precondition(pixels.count == width * height)
        self.pixels = pixels
        self.width = width
        self.height = height
    }

    /// Share of the frame at or above 128.
    public var coverage: Double {
        pixels.isEmpty ? 0 : Double(pixels.reduce(0) { $0 + ($1 >= 128 ? 1 : 0) }) / Double(pixels.count)
    }

    /// A single-channel 8-bit PNG.
    public func pngData() throws -> Data { try Fibo2PNG.pngData(self) }
    public func write(to url: URL) throws { try pngData().write(to: url) }
}
