// Image preparation for Fibo 2.0, isomorphic to diffusers 0.43 `BriaFibo2ImageProcessor`:
//   - `imageSize`: the size Fibo 2.0 was trained at for an aspect ratio (multiples of 16, within 10 % of the area,
//     the one that crops the least);
//   - an image to edit: `convert("RGB")`, scaled to cover the size (LANCZOS), centre-cropped;
//   - a masked edit: the region greyed out (128) through the mask made binary at 128, at the image's own size, first;
//   - Qwen3-VL's copy of each crop: `thumbnail((768, 768))` (BICUBIC);
//   - pixels to [-1, 1] as 2·(x / 255) − 1, back as round(clamp(x / 2 + 0.5) · 255).

import CoreGraphics
import Foundation
import ImageIO
import MLX

/// RGB8, interleaved.
public struct Fibo2RGBImage: Sendable, Equatable {
    public var rgb: [UInt8]
    public var width: Int
    public var height: Int

    public init(rgb: [UInt8], width: Int, height: Int) {
        precondition(rgb.count == width * height * 3)
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// `convert("RGB")` of straight RGBA: alpha dropped.
    public init(_ rgba: Fibo2RGBAImage) {
        var out = [UInt8](repeating: 0, count: rgba.width * rgba.height * 3)
        for i in 0..<(rgba.width * rgba.height) {
            out[i * 3] = rgba.rgba[i * 4]; out[i * 3 + 1] = rgba.rgba[i * 4 + 1]; out[i * 3 + 2] = rgba.rgba[i * 4 + 2]
        }
        self.init(rgb: out, width: rgba.width, height: rgba.height)
    }

    static func orientation(_ src: CGImageSource?) -> Int {
        guard let src, let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let o = props[kCGImagePropertyOrientation] as? Int else { return 1 }
        return o
    }

    /// Decode a file, apply its EXIF orientation and drop alpha.
    public static func read(url: URL) throws -> Fibo2RGBImage {
        Fibo2RGBImage(try Fibo2PNG.read(url: url)).oriented(orientation(CGImageSourceCreateWithURL(url as CFURL, nil)))
    }

    /// Decode in-memory PNG/JPEG data, apply its EXIF orientation and drop alpha (an engine request's image).
    public static func read(data: Data) throws -> Fibo2RGBImage {
        Fibo2RGBImage(try Fibo2PNG.read(data: data))
            .oriented(orientation(CGImageSourceCreateWithData(data as CFData, nil)))
    }

    /// PIL `exif_transpose` for EXIF orientation 1–8.
    public func oriented(_ orientation: Int) -> Fibo2RGBImage {
        guard (2...8).contains(orientation) else { return self }
        let (w, h) = (width, height)
        let swap = orientation >= 5
        let (ow, oh) = swap ? (h, w) : (w, h)
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        for y in 0..<oh {
            for x in 0..<ow {
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (w - 1 - x, y)
                case 3: (sx, sy) = (w - 1 - x, h - 1 - y)
                case 4: (sx, sy) = (x, h - 1 - y)
                case 5: (sx, sy) = (y, x)
                case 6: (sx, sy) = (y, h - 1 - x)
                case 7: (sx, sy) = (w - 1 - y, h - 1 - x)
                default: (sx, sy) = (w - 1 - y, x)
                }
                let s = (sy * w + sx) * 3, d = (y * ow + x) * 3
                out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2]
            }
        }
        return Fibo2RGBImage(rgb: out, width: ow, height: oh)
    }

    /// PIL `resize((w, h), filter)`.
    public func resized(width w: Int, height h: Int, filter: Fibo2PILResize.Filter = .lanczos) -> Fibo2RGBImage {
        if w == width && h == height { return self }
        let out = Fibo2PILResize.resampleBands(rgb, bands: 3, width: width, height: height, outWidth: w, outHeight: h,
                                               filter: filter)
        return Fibo2RGBImage(rgb: out, width: w, height: h)
    }

    public func cropped(left: Int, top: Int, width w: Int, height h: Int) -> Fibo2RGBImage {
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let s = ((top + y) * width + left) * 3
            out.replaceSubrange((y * w * 3)..<((y + 1) * w * 3), with: rgb[s..<(s + w * 3)])
        }
        return Fibo2RGBImage(rgb: out, width: w, height: h)
    }

    /// [1, 3, H, W] float32 in [-1, 1]: 2·(x / 255) − 1 (`VaeImageProcessor.preprocess`).
    public func pixelsNCHW() -> MLXArray {
        let plane = width * height
        var out = [Float](repeating: 0, count: 3 * plane)
        for i in 0..<plane { for c in 0..<3 { out[c * plane + i] = 2 * (Float(rgb[i * 3 + c]) / 255) - 1 } }
        return MLXArray(out, [1, 3, height, width])
    }

    /// From [3, H, W] in [-1, 1]: round(clamp(x / 2 + 0.5, 0, 1) · 255) (`VaeImageProcessor.postprocess`).
    public static func fromPixels(_ chw: MLXArray) -> Fibo2RGBImage {
        let (h, w) = (chw.dim(1), chw.dim(2))
        let v = MLX.round(clip(chw.asType(.float32) / 2 + 0.5, min: 0, max: 1) * 255).transposed(1, 2, 0)
            .asType(.uint8)
        return Fibo2RGBImage(rgb: v.asArray(UInt8.self), width: w, height: h)
    }

    public func rgba() -> Fibo2RGBAImage {
        var out = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            out[i * 4] = rgb[i * 3]; out[i * 4 + 1] = rgb[i * 3 + 1]; out[i * 4 + 2] = rgb[i * 3 + 2]
        }
        return Fibo2RGBAImage(rgba: out, width: width, height: height)
    }

    public func write(to url: URL) throws { try Fibo2PNG.write(rgba(), to: url) }
    public func pngData() throws -> Data { try Fibo2PNG.pngData(rgba()) }
}

public enum Fibo2ImagePrep {
    public static let multiple = 16
    public static let defaultArea = 1024 * 1024
    public static let textEncoderImageSize = 768
    /// Fibo 2.0 edits up to 5 images at once.
    public static let maxImages = 5

    /// `get_image_size`: the (width, height) Fibo 2.0 was trained at for an aspect ratio (width / height): of the
    /// sizes divisible by 16, at least 64 a side and within 10 % of `area`, the one that crops the least from a picture
    /// with that aspect ratio, then the one closest to `area`, then the widest.
    public static func imageSize(aspectRatio: Double, area: Int = defaultArea) -> (width: Int, height: Int) {
        let minSide = 4 * multiple
        var best: (crop: Double, delta: Int, width: Int, height: Int)? = nil
        let lo = 0.9 * Double(area), hi = 1.1 * Double(area)
        var width = minSide
        while width < 2 * area / minSide {
            var height = minSide
            while height < 2 * area / width {
                let pixels = width * height
                if Double(pixels) >= lo, Double(pixels) <= hi {
                    let ratio = Double(width) / Double(height)
                    let crop = ((1 - min(ratio / aspectRatio, aspectRatio / ratio)) * 1e9).rounded(.toNearestOrEven) / 1e9
                    let delta = abs(pixels - area)
                    // key: (round(crop, 9), |pixels − area|, −width); the first minimum in iteration order wins.
                    let better: Bool
                    if let b = best {
                        better = crop < b.crop || (crop == b.crop && (delta < b.delta || (delta == b.delta && width > b.width)))
                    } else {
                        better = true
                    }
                    if better { best = (crop, delta, width, height) }
                }
                height += multiple
            }
            width += multiple
        }
        return (best!.width, best!.height)
    }

    /// `_resize_and_crop`: scale to cover `width` × `height` (LANCZOS), keep the centre.
    public static func resizeAndCrop(_ image: Fibo2RGBImage, width: Int, height: Int) -> Fibo2RGBImage {
        let scale = max(Double(height) / Double(image.height), Double(width) / Double(image.width))
        let rw = Int((Double(image.width) * scale).rounded(.up)), rh = Int((Double(image.height) * scale).rounded(.up))
        let resized = image.resized(width: rw, height: rh, filter: .lanczos)
        return resized.cropped(left: (rw - width) / 2, top: (rh - height) / 2, width: width, height: height)
    }

    /// `grey_out`: flat (128, 128, 128) through the mask (single-band, the image's size) made binary at 128.
    public static func greyOut(_ image: Fibo2RGBImage, mask: [UInt8]) -> Fibo2RGBImage {
        precondition(mask.count == image.width * image.height, "the mask has to be the image's size")
        var out = image
        for i in 0..<mask.count where mask[i] >= 128 {
            out.rgb[i * 3] = 128; out.rgb[i * 3 + 1] = 128; out.rgb[i * 3 + 2] = 128
        }
        return out
    }

    /// PIL `convert("L")` of RGB: ITU-R 601-2 luma, (19595·R + 38470·G + 7471·B + 0x8000) >> 16.
    public static func luma(_ image: Fibo2RGBImage) -> [UInt8] {
        (0..<(image.width * image.height)).map { i in
            let r = Int(image.rgb[i * 3]), g = Int(image.rgb[i * 3 + 1]), b = Int(image.rgb[i * 3 + 2])
            return UInt8((19595 * r + 38470 * g + 7471 * b + 0x8000) >> 16)
        }
    }

    /// PIL `thumbnail((size, size))` on a copy: BICUBIC to the largest size that fits, aspect kept; unchanged when it
    /// already fits. (Pillow's `reducing_gap` pre-reduction starts at a 4× shrink, which a ~1 MP crop never reaches.)
    public static func thumbnail(_ image: Fibo2RGBImage, size: Int = textEncoderImageSize) -> Fibo2RGBImage {
        if size >= image.width && size >= image.height { return image }
        var (x, y) = (Double(size), Double(size))
        let aspect = Double(image.width) / Double(image.height)
        func roundAspect(_ number: Double, _ key: (Double) -> Double) -> Double {
            let lo = number.rounded(.down), hi = number.rounded(.up)
            return max(key(lo) <= key(hi) ? lo : hi, 1)
        }
        if x / y >= aspect {
            x = roundAspect(y * aspect) { n in abs(aspect - n / y) }
        } else {
            y = roundAspect(x / aspect) { n in n == 0 ? 0 : abs(aspect - x / n) }
        }
        return image.resized(width: Int(x), height: Int(y), filter: .bicubic)
    }

    public struct Prepared: Sendable {
        /// The crops the VAE encodes, one per image (their sizes can differ).
        public var crops: [Fibo2RGBImage]
        /// Qwen3-VL's copies, at most 768 px a side.
        public var textEncoderImages: [Fibo2RGBImage]
        /// The output size.
        public var width: Int
        public var height: Int
    }

    /// `preprocess` plus the size defaults of `__call__`: a single image is cropped to the output size (by default the
    /// size for its own aspect ratio); several images each to the size for their own aspect ratio, and the output
    /// takes the first image's unless given.
    public static func prepare(images: [Fibo2RGBImage], mask: [UInt8]? = nil, width: Int? = nil, height: Int? = nil,
                               area: Int = defaultArea) -> Prepared {
        var images = images
        if let mask { images = [greyOut(images[0], mask: mask)] }
        var (w, h) = (width, height)
        if w == nil && h == nil {
            let size = imageSize(aspectRatio: Double(images[0].width) / Double(images[0].height), area: area)
            (w, h) = (size.width, size.height)
        }
        let side = Int(Double(area).squareRoot())
        let (outW, outH) = (w ?? side, h ?? side)
        let sizes: [(width: Int, height: Int)] = images.count == 1
            ? [(outW, outH)]
            : images.map { imageSize(aspectRatio: Double($0.width) / Double($0.height), area: area) }
        let crops = zip(images, sizes).map { resizeAndCrop($0, width: $1.width, height: $1.height) }
        return Prepared(crops: crops, textEncoderImages: crops.map { thumbnail($0) }, width: outW, height: outH)
    }
}
