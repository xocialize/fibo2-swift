// Order-preserving JSON for Fibo 2.0 captions. The model reads its caption as text, serialized the way Python's
// `json.dumps(caption, separators=(",", ":"), ensure_ascii=False)` writes it, with the keys in the order the caller
// gave them. Foundation's JSONSerialization keeps neither the key order nor Python's number formatting, so this is a
// small parser and writer that follow `json.loads` / `json.dumps`:
//   - objects keep insertion order; a repeated key keeps its first position and takes the last value (a Python dict);
//   - integers are kept as written (Python ints are arbitrary precision), floats go through Double and print as
//     Python's `repr` (Swift's shortest round-trip description is the same form: 1.0, 2.8, 1e-05, 1e+16);
//   - `NaN`, `Infinity` and `-Infinity` parse, as in Python;
//   - strings escape only `"`, `\` and the control characters below U+0020; everything else is written as is.

import Foundation

public indirect enum Fibo2JSON: Equatable, Sendable {
    case null
    case bool(Bool)
    /// An integer literal as written (canonical: no leading "+", "-0" is "0").
    case int(String)
    case double(Double)
    case string(String)
    case array([Fibo2JSON])
    case object(Fibo2JSONObject)

    /// Python truthiness: `not value`.
    public var isFalsy: Bool {
        switch self {
        case .null: return true
        case .bool(let b): return !b
        case .int(let s): return s == "0"
        case .double(let d): return d == 0
        case .string(let s): return s.isEmpty
        case .array(let a): return a.isEmpty
        case .object(let o): return o.members.isEmpty
        }
    }

    public var objectValue: Fibo2JSONObject? { if case .object(let o) = self { return o }; return nil }
    public var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    public var arrayValue: [Fibo2JSON]? { if case .array(let a) = self { return a }; return nil }
}

/// An insertion-ordered JSON object (a Python dict).
public struct Fibo2JSONObject: Equatable, Sendable {
    public struct Member: Equatable, Sendable {
        public var key: String
        public var value: Fibo2JSON
    }

    public private(set) var members: [Member] = []

    public init() {}

    public init(_ pairs: [(String, Fibo2JSON)]) {
        for (k, v) in pairs { self[k] = v }
    }

    public var keys: [String] { members.map(\.key) }

    public subscript(key: String) -> Fibo2JSON? {
        get { members.first { $0.key == key }?.value }
        set {
            if let i = members.firstIndex(where: { $0.key == key }) {
                if let newValue { members[i].value = newValue } else { members.remove(at: i) }
            } else if let newValue {
                members.append(Member(key: key, value: newValue))
            }
        }
    }

    /// `dict.pop(key, None)`.
    public mutating func pop(_ key: String) -> Fibo2JSON? {
        guard let i = members.firstIndex(where: { $0.key == key }) else { return nil }
        return members.remove(at: i).value
    }

    public func contains(_ key: String) -> Bool { members.contains { $0.key == key } }

    /// Dict equality ignores order, as in Python.
    public static func == (a: Fibo2JSONObject, b: Fibo2JSONObject) -> Bool {
        a.members.count == b.members.count && a.members.allSatisfy { b[$0.key] == $0.value }
    }
}

public struct Fibo2JSONError: Error, CustomStringConvertible {
    public let description: String
}

extension Fibo2JSON {

    // MARK: Parse (json.loads)

    public static func parse(_ text: String) throws -> Fibo2JSON {
        var p = Parser(Array(text.unicodeScalars))
        p.skipWhitespace()
        let value = try p.value()
        p.skipWhitespace()
        guard p.i == p.s.count else { throw Fibo2JSONError(description: "extra data at \(p.i)") }
        return value
    }

    struct Parser {
        let s: [Unicode.Scalar]
        var i = 0

        init(_ s: [Unicode.Scalar]) { self.s = s }

        mutating func skipWhitespace() {
            while i < s.count, s[i] == " " || s[i] == "\t" || s[i] == "\n" || s[i] == "\r" { i += 1 }
        }

        func fail(_ what: String) -> Fibo2JSONError { Fibo2JSONError(description: "\(what) at \(i)") }

        mutating func literal(_ word: String) -> Bool {
            let w = Array(word.unicodeScalars)
            guard i + w.count <= s.count, Array(s[i ..< i + w.count]) == w else { return false }
            i += w.count
            return true
        }

        mutating func value() throws -> Fibo2JSON {
            guard i < s.count else { throw fail("unexpected end") }
            switch s[i] {
            case "{": return try object()
            case "[": return try array()
            case "\"": return .string(try string())
            case "n": if literal("null") { return .null }
            case "t": if literal("true") { return .bool(true) }
            case "f": if literal("false") { return .bool(false) }
            case "N": if literal("NaN") { return .double(.nan) }
            case "I": if literal("Infinity") { return .double(.infinity) }
            case "-": if literal("-Infinity") { return .double(-.infinity) }; return try number()
            case "0"..."9": return try number()
            default: break
            }
            throw fail("unexpected character")
        }

        mutating func object() throws -> Fibo2JSON {
            i += 1
            var out = Fibo2JSONObject()
            skipWhitespace()
            if i < s.count, s[i] == "}" { i += 1; return .object(out) }
            while true {
                skipWhitespace()
                guard i < s.count, s[i] == "\"" else { throw fail("expected a property name") }
                let key = try string()
                skipWhitespace()
                guard i < s.count, s[i] == ":" else { throw fail("expected ':'") }
                i += 1
                skipWhitespace()
                out[key] = try value()
                skipWhitespace()
                guard i < s.count else { throw fail("unexpected end") }
                if s[i] == "," { i += 1; continue }
                if s[i] == "}" { i += 1; return .object(out) }
                throw fail("expected ',' or '}'")
            }
        }

        mutating func array() throws -> Fibo2JSON {
            i += 1
            var out: [Fibo2JSON] = []
            skipWhitespace()
            if i < s.count, s[i] == "]" { i += 1; return .array(out) }
            while true {
                skipWhitespace()
                out.append(try value())
                skipWhitespace()
                guard i < s.count else { throw fail("unexpected end") }
                if s[i] == "," { i += 1; continue }
                if s[i] == "]" { i += 1; return .array(out) }
                throw fail("expected ',' or ']'")
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= s.count, let v = UInt32(String(String.UnicodeScalarView(s[i ..< i + 4])), radix: 16) else {
                throw fail("bad \\u escape")
            }
            i += 4
            return v
        }

        mutating func string() throws -> String {
            i += 1
            var out = String.UnicodeScalarView()
            while true {
                guard i < s.count else { throw fail("unterminated string") }
                let c = s[i]
                i += 1
                if c == "\"" { return String(out) }
                if c.value < 0x20 { throw fail("control character in string") }  // json.loads strict mode
                guard c == "\\" else { out.append(c); continue }
                guard i < s.count else { throw fail("unterminated escape") }
                let e = s[i]
                i += 1
                switch e {
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                case "/": out.append("/")
                case "b": out.append("\u{08}")
                case "f": out.append("\u{0C}")
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "u":
                    var v = try hex4()
                    if (0xD800 ... 0xDBFF).contains(v), i + 1 < s.count, s[i] == "\\", s[i + 1] == "u" {
                        let save = i
                        i += 2
                        let low = try hex4()
                        if (0xDC00 ... 0xDFFF).contains(low) {
                            v = 0x10000 + ((v - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            i = save
                        }
                    }
                    guard let scalar = Unicode.Scalar(v) else { throw fail("lone surrogate") }
                    out.append(scalar)
                default: throw fail("bad escape")
                }
            }
        }

        mutating func number() throws -> Fibo2JSON {
            let start = i
            if i < s.count, s[i] == "-" { i += 1 }
            guard i < s.count, ("0"..."9").contains(s[i]) else { throw fail("bad number") }
            if s[i] == "0" { i += 1 } else { while i < s.count, ("0"..."9").contains(s[i]) { i += 1 } }
            var isFloat = false
            if i + 1 < s.count, s[i] == ".", ("0"..."9").contains(s[i + 1]) {
                isFloat = true
                i += 1
                while i < s.count, ("0"..."9").contains(s[i]) { i += 1 }
            }
            if i < s.count, s[i] == "e" || s[i] == "E" {
                var j = i + 1
                if j < s.count, s[j] == "+" || s[j] == "-" { j += 1 }
                if j < s.count, ("0"..."9").contains(s[j]) {
                    isFloat = true
                    i = j
                    while i < s.count, ("0"..."9").contains(s[i]) { i += 1 }
                }
            }
            let text = String(String.UnicodeScalarView(s[start ..< i]))
            if isFloat {
                guard let d = Double(text) else { throw fail("bad number") }
                return .double(d)
            }
            return .int(text == "-0" ? "0" : text)
        }
    }

    // MARK: Write (json.dumps(..., separators=(",", ":"), ensure_ascii=False))

    public func dumps() -> String {
        var out = ""
        write(into: &out)
        return out
    }

    static func writeString(_ s: String, into out: inout String) {
        out.append("\"")
        for c in s.unicodeScalars {
            switch c {
            case "\"": out.append("\\\"")
            case "\\": out.append("\\\\")
            case "\n": out.append("\\n")
            case "\r": out.append("\\r")
            case "\t": out.append("\\t")
            case "\u{08}": out.append("\\b")
            case "\u{0C}": out.append("\\f")
            default:
                if c.value < 0x20 {
                    out.append(String(format: "\\u%04x", c.value))
                } else {
                    out.unicodeScalars.append(c)
                }
            }
        }
        out.append("\"")
    }

    /// Python's `float.__repr__`.
    static func pythonRepr(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d < 0 ? "-Infinity" : "Infinity" }
        return "\(d)"
    }

    func write(into out: inout String) {
        switch self {
        case .null: out.append("null")
        case .bool(let b): out.append(b ? "true" : "false")
        case .int(let s): out.append(s)
        case .double(let d): out.append(Self.pythonRepr(d))
        case .string(let s): Self.writeString(s, into: &out)
        case .array(let a):
            out.append("[")
            for (k, v) in a.enumerated() {
                if k > 0 { out.append(",") }
                v.write(into: &out)
            }
            out.append("]")
        case .object(let o):
            out.append("{")
            for (k, m) in o.members.enumerated() {
                if k > 0 { out.append(",") }
                Self.writeString(m.key, into: &out)
                out.append(":")
                m.value.write(into: &out)
            }
            out.append("}")
        }
    }
}
