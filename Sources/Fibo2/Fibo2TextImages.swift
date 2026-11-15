// Images of the text a Fibo 2.0 caption asks for, drawn the way diffusers' `text_images.py` draws them (and Fibo 2.0's
// training did): each `text_render` text on its own, whitespace collapsed, in lines of 256 characters, black on white
// with 2 px of padding, at size 20. Qwen3-VL sees them with the prompt.
//
// The reference draws with Pillow + FreeType. This port lays the text out with the SAME metrics and draws the glyphs
// with CoreText, so the image SIZE (which sets Qwen3-VL's token count) follows Pillow's rules and the pixels differ
// only by rasterizer (FreeType hints, CoreGraphics doesn't):
//   - "one font" path (every character is ASCII or in Pillow's built-in font): the line box is Pillow's `textbbox`,
//     computed from the built-in font's hinted per-character metrics (`Fibo2DefaultFont.metrics`, exact);
//   - "mixed" path (any other character): each character is drawn with the first font that has it — the built-in
//     font, then the fallbacks (training used Aileron, then DejaVu Sans; both are in the snapshot's `fonts/`) — on a
//     common baseline; the line is ascent + descent of the built-in font tall and the sum of the advances wide.
//     Fallback advances are FreeType's hinted ones: the scaled advance rounded to a whole pixel, plus the `kern`
//     table's pair kerning as `FT_Get_Kerning` grid-fits it (Pillow's basic layout applies it).

import CoreGraphics
import CoreText
import Foundation

public final class Fibo2TextRenderer {
    public static let fontSize = 20
    public static let maxLineCharacters = 256
    static let padding = 2
    static let lineSpacing = 4

    /// A font with FreeType-style hinted advances.
    final class Face {
        let font: CTFont
        let unitsPerEm: Int
        let unitsFont: CTFont
        /// `kern` table format-0 pairs (left glyph << 16 | right glyph → font units), when the font has one.
        let kerning: [UInt32: Int]

        init(_ font: CTFont) {
            self.font = font
            self.unitsPerEm = Int(CTFontGetUnitsPerEm(font))
            self.unitsFont = CTFontCreateCopyWithAttributes(font, CGFloat(unitsPerEm), nil, nil)
            self.kerning = Face.readKerning(font)
        }

        func glyph(_ scalar: Unicode.Scalar) -> CGGlyph? {
            var units = Array(String(scalar).utf16)
            var glyphs = [CGGlyph](repeating: 0, count: units.count)
            guard CTFontGetGlyphsForCharacters(font, &units, &glyphs, units.count), glyphs[0] != 0 else { return nil }
            return glyphs[0]
        }

        /// FreeType's 16.16 scale for this size, and a font-unit distance scaled to 26.6.
        var scale: Int { ((Fibo2TextRenderer.fontSize * 64) << 16 + unitsPerEm / 2) / unitsPerEm }
        func scaled(_ units: Int) -> Int {
            let v = abs(units) * scale + 0x8000
            return units < 0 ? -(v >> 16) : v >> 16
        }

        /// The hinted advance in pixels: the scaled advance rounded to a whole pixel.
        func advance(_ glyph: CGGlyph) -> Int {
            var g = glyph
            var adv = CGSize.zero
            CTFontGetAdvancesForGlyphs(unitsFont, .horizontal, &g, &adv, 1)
            return (scaled(Int(adv.width.rounded())) + 32) >> 6
        }

        /// `FT_Get_Kerning(…, FT_KERNING_DEFAULT)` in pixels: scaled, shrunk below 25 ppem, grid-fitted.
        func kern(_ left: CGGlyph, _ right: CGGlyph) -> Int {
            guard let units = kerning[UInt32(left) << 16 | UInt32(right)], units != 0 else { return 0 }
            var v = scaled(units)
            let ppem = Fibo2TextRenderer.fontSize
            if ppem < 25 {
                let m = abs(v) * ppem
                v = v < 0 ? -((m + 12) / 25) : (m + 12) / 25
            }
            return v >= 0 ? (v + 32) >> 6 : -((-v + 32) >> 6)
        }

        static func readKerning(_ font: CTFont) -> [UInt32: Int] {
            let tag = CTFontTableTag(0x6B65_726E)  // 'kern'
            guard let data = CTFontCopyTable(font, tag, []) as Data?, data.count >= 4 else { return [:] }
            func u16(_ o: Int) -> Int { o + 1 < data.count ? Int(data[o]) << 8 | Int(data[o + 1]) : 0 }
            guard u16(0) == 0 else { return [:] }  // version 0 (the form FreeType reads for TrueType)
            var out: [UInt32: Int] = [:]
            var offset = 4
            for _ in 0..<u16(2) {
                let length = u16(offset + 2), coverage = u16(offset + 4)
                // format 0, horizontal, not minimum, not cross-stream
                if coverage >> 8 == 0, coverage & 1 == 1, coverage & 0x6 == 0 {
                    let pairs = u16(offset + 6)
                    var p = offset + 14
                    for _ in 0..<pairs where p + 5 < data.count {
                        let key = UInt32(u16(p)) << 16 | UInt32(u16(p + 2))
                        var value = u16(p + 4)
                        if value >= 0x8000 { value -= 0x10000 }
                        out[key, default: 0] += value
                        p += 6
                    }
                }
                if length == 0 { break }
                offset += length
            }
            return out
        }
    }

    let builtin: Face
    let fallbacks: [Face]
    let fallbackNames: [String]

    /// `fontsDirectory`: the snapshot's `fonts/` (Aileron-Regular.otf, DejaVuSans.ttf), the fallbacks training used.
    /// Without it, a character the built-in font lacks is drawn as its missing-glyph box.
    public init(fontsDirectory: URL? = nil) throws {
        func load(_ data: Data, _ label: String) throws -> Face {
            guard let descriptor = CTFontManagerCreateFontDescriptorFromData(data as CFData) else {
                throw Fibo2Error.loading("cannot read the font \(label)")
            }
            return Face(CTFontCreateWithFontDescriptor(descriptor, CGFloat(Self.fontSize), nil))
        }
        self.builtin = try load(Fibo2DefaultFont.ttf, "(Pillow's built-in Aileron subset)")
        var fallbacks: [Face] = []
        var names: [String] = []
        if let fontsDirectory {
            for name in ["Aileron-Regular.otf", "DejaVuSans.ttf"] {
                let url = fontsDirectory.appendingPathComponent(name)
                if let data = try? Data(contentsOf: url) {
                    fallbacks.append(try load(data, name))
                    names.append(name)
                }
            }
        }
        self.fallbacks = fallbacks
        self.fallbackNames = names
    }

    public struct AdvanceReport {
        public var checked = 0
        public var mismatched = 0
        public var examples: [String] = []
    }

    /// Gate hook: this port's hinted advances of the fallback fonts against FreeType's (font file → code point →
    /// advance in pixels, as Pillow's `getlength` reports them).
    public func advanceReport(_ golden: [String: [String: Double]]) -> [String: AdvanceReport] {
        var out: [String: AdvanceReport] = [:]
        for (name, face) in zip(fallbackNames, fallbacks) {
            var report = AdvanceReport()
            for (code, advance) in golden[name] ?? [:] {
                guard let v = UInt32(code), let scalar = Unicode.Scalar(v), let glyph = face.glyph(scalar) else { continue }
                report.checked += 1
                let own = face.advance(glyph)
                if own != Int(advance) {
                    report.mismatched += 1
                    if report.examples.count < 12 { report.examples.append("U+\(String(v, radix: 16)):\(own)≠\(Int(advance))") }
                }
            }
            out[name] = report
        }
        return out
    }

    /// Python `" ".join(text.split())`.
    static func collapseWhitespace(_ text: String) -> [Unicode.Scalar] {
        func isSpace(_ c: Unicode.Scalar) -> Bool {
            switch c.value {
            case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
                return true
            default: return false
            }
        }
        var out: [Unicode.Scalar] = []
        var pending = false
        for c in text.unicodeScalars {
            if isSpace(c) { pending = !out.isEmpty; continue }
            if pending { out.append(" "); pending = false }
            out.append(c)
        }
        return out
    }

    static func builtinHas(_ c: Unicode.Scalar) -> Bool { c.value < 128 || Fibo2DefaultFont.metrics[c.value] != nil }
    static func builtinMetrics(_ c: Unicode.Scalar) -> [Int] { Fibo2DefaultFont.metrics[c.value] ?? Fibo2DefaultFont.notdef }

    /// Pillow's `textbbox((0, 0), line)` for the built-in font: (left, top, right, bottom) from the ascender line.
    static func bbox(_ line: [Unicode.Scalar]) -> (left: Int, top: Int, right: Int, bottom: Int) {
        var (pos, xMin, xMax, top, bottom) = (0, 0, 0, 0, 0)
        for c in line {
            let m = builtinMetrics(c)
            xMin = min(xMin, pos + m[1]); xMax = max(xMax, pos + m[2])
            top = min(top, m[3]); bottom = max(bottom, m[4])
            pos += m[0]
        }
        return (xMin, Fibo2DefaultFont.ascent + top, max(xMax, pos), Fibo2DefaultFont.ascent + bottom)
    }

    struct Placed {
        var face: Face
        var glyph: CGGlyph
        var x: Int
        var baseline: Int
    }

    /// `draw_text_image`.
    public func draw(_ text: String) -> Fibo2RGBImage {
        let line = Self.collapseWhitespace(text)
        var lines = stride(from: 0, to: line.count, by: Self.maxLineCharacters).map {
            Array(line[$0 ..< min($0 + Self.maxLineCharacters, line.count)])
        }
        if lines.isEmpty { lines = [[]] }
        return line.allSatisfy(Self.builtinHas) ? drawOneFont(lines) : drawMixed(lines)
    }

    func builtinGlyph(_ c: Unicode.Scalar) -> CGGlyph { builtin.glyph(c) ?? 0 }

    func drawOneFont(_ lines: [[Unicode.Scalar]]) -> Fibo2RGBImage {
        let minBox = Self.bbox(Array("Ag".unicodeScalars))
        let minLineHeight = max(1, minBox.bottom - minBox.top)
        var placed: [Placed] = []
        var maxWidth = 1
        var boxes: [(box: (left: Int, top: Int, right: Int, bottom: Int), height: Int)] = []
        var totalHeight = 0
        for line in lines {
            let box = Self.bbox(line.isEmpty ? [" "] : line)
            let height = max(minLineHeight, box.bottom - box.top)
            boxes.append((box, height))
            maxWidth = max(maxWidth, box.right - box.left)
            totalHeight += height
        }
        totalHeight += Self.lineSpacing * (lines.count - 1)
        var y = Self.padding
        for (line, (box, height)) in zip(lines, boxes) {
            // draw.text((pad − left, y − top), anchor "la"): the pen starts at pad − left, the ascender line at y − top.
            var x = Self.padding - box.left
            let baseline = y - box.top + Fibo2DefaultFont.ascent
            for c in line {
                placed.append(Placed(face: builtin, glyph: builtinGlyph(c), x: x, baseline: baseline))
                x += Self.builtinMetrics(c)[0]
            }
            y += height + Self.lineSpacing
        }
        return render(placed, width: maxWidth + Self.padding * 2, height: totalHeight + Self.padding * 2)
    }

    func drawMixed(_ lines: [[Unicode.Scalar]]) -> Fibo2RGBImage {
        let ascent = Fibo2DefaultFont.ascent
        let lineHeight = max(1, ascent + Fibo2DefaultFont.descent)
        var placed: [Placed] = []
        var width = 1
        var y = Self.padding
        for line in lines {
            var x = Self.padding
            var previous: (face: Face, glyph: CGGlyph)? = nil
            for c in line.isEmpty ? [" "] : line {
                if Self.builtinHas(c) {
                    placed.append(Placed(face: builtin, glyph: builtinGlyph(c), x: x, baseline: y + ascent))
                    x += Self.builtinMetrics(c)[0]
                    previous = nil  // the built-in font has no `kern` table
                    continue
                }
                var chosen: (face: Face, glyph: CGGlyph)? = nil
                for face in fallbacks {
                    if let g = face.glyph(c) { chosen = (face, g); break }
                }
                guard let (face, glyph) = chosen else {
                    placed.append(Placed(face: builtin, glyph: 0, x: x, baseline: y + ascent))
                    x += Fibo2DefaultFont.notdef[0]
                    previous = nil
                    continue
                }
                // Kerning applies inside a run of one font.
                if let p = previous, p.face === face { x += face.kern(p.glyph, glyph) }
                placed.append(Placed(face: face, glyph: glyph, x: x, baseline: y + ascent))
                x += face.advance(glyph)
                previous = (face, glyph)
            }
            width = max(width, max(1, x - Self.padding))
            y += lineHeight + Self.lineSpacing
        }
        let height = lineHeight * lines.count + Self.lineSpacing * (lines.count - 1)
        return render(placed, width: width + Self.padding * 2, height: height + Self.padding * 2)
    }

    func render(_ placed: [Placed], width: Int, height: Int) -> Fibo2RGBImage {
        var gray = [UInt8](repeating: 255, count: width * height)
        gray.withUnsafeMutableBytes { buffer in
            guard let ctx = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            ctx.setShouldAntialias(true)
            ctx.setShouldSmoothFonts(false)
            ctx.setShouldSubpixelPositionFonts(false)
            ctx.setShouldSubpixelQuantizeFonts(false)
            ctx.setFillColor(gray: 0, alpha: 1)
            for p in placed {
                var glyph = p.glyph
                var position = CGPoint(x: CGFloat(p.x), y: CGFloat(height - p.baseline))
                CTFontDrawGlyphs(p.face.font, &glyph, &position, 1, ctx)
            }
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) { rgb[i * 3] = gray[i]; rgb[i * 3 + 1] = gray[i]; rgb[i * 3 + 2] = gray[i] }
        return Fibo2RGBImage(rgb: rgb, width: width, height: height)
    }
}
