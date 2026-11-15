// Prompt shaping for Fibo 2.0, isomorphic to diffusers 0.43 `BriaFibo2Pipeline` (`_parse_prompt`,
// `_mark_text_images`, `_get_edit_prompt` and the prompt handling at the top of `__call__`).
//
// Fibo 2.0 reads a structured JSON caption (Bria's VGL) as text, in the form its training used:
//   - empty values (null, "", {}, [] and NaN) are dropped at every depth, with the `aesthetic_score` and
//     `preference_score` fields, and the caption is serialized compactly in the caller's key order;
//   - a prompt that isn't a JSON object goes as it is (a short description, or an edit's instruction alone);
//   - an edit prompt is one JSON object: a vision marker per image to edit, the caption of the result when the prompt
//     has one, then the edit instruction;
//   - each `text_render` entry with text gets an `image_tokens` entry: the vision marker of the image of its text,
//     or "" in the base prompt a generation runs its first steps with.

import Foundation

public enum Fibo2Prompt {

    /// The null caption Fibo 2.0 was trained with for classifier-free guidance.
    public static let defaultNegative =
        "{\"short_description\":\"\",\"objects\":[],\"background_setting\":\"\",\"lighting\":{},"
        + "\"aesthetics\":{},\"photographic_characteristics\":{},\"style_medium\":\"\","
        + "\"context\":\"\",\"artistic_style\":\"\"}"

    /// The model card's text-to-image negative: steers away from soft, featureless, artifact-laden images.
    public static let cardNegativeT2I =
        "{\"short_description\":\"\",\"objects\":[{\"description\":\"the main subject\",\"texture\":\"smooth, "
        + "featureless, plastic-like, airbrushed, no fine surface detail\",\"appearance_details\":\"soft blurred "
        + "edges, simplified shapes, no fine detail\",\"skin_tone_and_texture\":\"waxy airbrushed skin, poreless, "
        + "plastic smooth\"}],\"background_setting\":\"a plain seamless studio backdrop, perfectly smooth gradient, "
        + "blurred and featureless, no visible texture\",\"lighting\":{},\"aesthetics\":{\"artifacts\":["
        + "\"oversaturated colours\",\"neon colour cast\",\"overexposed highlights\",\"lens flare\",\"slight blur\","
        + "\"motion blur\",\"haze\"]},\"photographic_characteristics\":{\"depth_of_field\":\"shallow, heavy bokeh "
        + "obscuring most of the frame\",\"focus\":\"soft, slightly out of focus, lacking fine texture\"},"
        + "\"style_medium\":\"\",\"context\":\"\",\"artistic_style\":\"\"}"

    public static let visionMarker = "<|vision_start|><|image_pad|><|vision_end|>"

    public static func template(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    static let scores: Set<String> = ["aesthetic_score", "preference_score"]

    static func empty(_ v: Fibo2JSON) -> Bool {
        switch v {
        case .null: return true
        case .double(let d): return d.isNaN
        case .string(let s): return s.isEmpty
        case .array(let a): return a.isEmpty
        case .object(let o): return o.members.isEmpty
        default: return false
        }
    }

    static func clean(_ v: Fibo2JSON) -> Fibo2JSON {
        switch v {
        case .object(let o):
            var out = Fibo2JSONObject()
            for m in o.members {
                let c = clean(m.value)
                if !empty(c) { out[m.key] = c }
            }
            return .object(out)
        case .array(let a):
            return .array(a.map(clean).filter { !empty($0) })
        default:
            return v
        }
    }

    /// `_parse_prompt`: the prompt's caption in its trained form, and its edit instruction. A prompt that isn't a JSON
    /// object comes back as `.string(prompt)` with no instruction.
    public static func parse(_ prompt: String) -> (caption: Fibo2JSON?, instruction: Fibo2JSON?) {
        guard let parsed = try? Fibo2JSON.parse(prompt), case .object(var caption) = parsed else {
            return (.string(prompt), nil)
        }
        let instruction = caption.pop("edit_instruction")
        if caption.contains("structured_caption") {
            let inner = caption.pop("structured_caption")!
            guard case .object(let o) = inner else { return (inner.isFalsy ? nil : inner, instruction) }
            caption = o
        }
        let keys = Set(caption.keys)
        var out = Fibo2JSONObject()
        guard case .object(let cleaned) = clean(.object(caption)) else { return (nil, instruction) }
        for m in cleaned.members where !scores.contains(m.key) {
            var value = m.value
            if case .object(let section) = value {
                var s = Fibo2JSONObject()
                for sm in section.members where !scores.contains(sm.key) { s[sm.key] = sm.value }
                value = .object(s)
                if s.members.isEmpty { continue }
            }
            out[m.key] = value
        }
        if keys == ["short_description", "text_render"] {
            var fixed = Fibo2JSONObject()
            let short = out["short_description"]
            fixed["short_description"] = (short == nil || short!.isFalsy) ? .string("") : short!
            let text = out["text_render"]
            fixed["text_render"] = (text == nil || text!.isFalsy) ? .array([]) : text!
            out = fixed
        }
        return (out.members.isEmpty ? nil : .object(out), instruction)
    }

    /// A generation prompt as the pipeline encodes it: a JSON caption cleaned and serialized compactly; anything
    /// else unchanged.
    public static func generationPrompt(_ prompt: String) -> String {
        let (caption, _) = parse(prompt)
        if case .string = caption { return prompt }
        return (caption ?? .object(Fibo2JSONObject())).dumps()
    }

    /// The texts of a caption's `text_render` entries that have text, in order.
    public static func renderTexts(_ caption: Fibo2JSON?) -> [String] {
        guard case .object(let o) = caption, let entries = o["text_render"]?.arrayValue else { return [] }
        return entries.compactMap { entry in
            guard let text = entry.objectValue?["text"]?.stringValue, !text.isEmpty else { return nil }
            return text
        }
    }

    /// `_mark_text_images`: each `text_render` entry with text gets `image_tokens: marker`.
    public static func markTextImages(_ caption: Fibo2JSONObject, marker: String) -> Fibo2JSONObject {
        guard let entries = caption["text_render"]?.arrayValue else { return caption }
        var out = caption
        out["text_render"] = .array(entries.map { entry in
            guard var o = entry.objectValue, let text = o["text"]?.stringValue, !text.isEmpty else { return entry }
            o["image_tokens"] = .string(marker)
            return .object(o)
        })
        return out
    }

    /// `_get_edit_prompt`.
    public static func editPrompt(_ prompt: String, imageCount: Int, textImages: Bool) -> String {
        var (caption, instruction) = parse(prompt)
        if instruction == nil {
            caption = nil
            instruction = .string(prompt)
        }
        if textImages, case .object(let o) = caption {
            caption = .object(markTextImages(o, marker: visionMarker))
        }
        var edit = Fibo2JSONObject()
        if imageCount == 1 {
            edit["image"] = .string(visionMarker)
        } else {
            for k in 1 ... max(imageCount, 1) { edit["image_\(k)"] = .string(visionMarker) }
        }
        if let caption, !caption.isFalsy { edit["structured_caption"] = caption }
        edit["edit_instruction"] = instruction!
        return Fibo2JSON.object(edit).dumps()
    }

    /// One Qwen3-VL call of a request: the user turn's content (before the chat template) and the number of images
    /// of the caption's text it marks (after any images to edit).
    public struct Shaped: Equatable, Sendable {
        /// The conditional prompt, with every image marked.
        public var prompt: String
        /// The texts to draw, one image each, in marker order (after the images to edit).
        public var renderTexts: [String]
        /// A generation with text images runs its first steps (noise level above 0.9) on this prompt, without them.
        public var basePrompt: String?
    }

    /// The prompts of a request, as `__call__` builds them. `imageCount` is the number of images to edit (0 for a
    /// generation); `renderTextImages` draws the caption's texts (the pipeline's default).
    public static func shape(_ prompt: String, imageCount: Int, renderTextImages: Bool = true) -> Shaped {
        if imageCount > 0 {
            let texts = renderTextImages ? renderTexts(parse(prompt).caption) : []
            return Shaped(prompt: editPrompt(prompt, imageCount: imageCount, textImages: !texts.isEmpty),
                          renderTexts: texts, basePrompt: nil)
        }
        let p = generationPrompt(prompt)
        let texts = renderTextImages ? renderTexts(parse(p).caption) : []
        guard !texts.isEmpty, case .object(let o)? = try? Fibo2JSON.parse(p) else {
            return Shaped(prompt: p, renderTexts: [], basePrompt: nil)
        }
        return Shaped(prompt: Fibo2JSON.object(markTextImages(o, marker: visionMarker)).dumps(), renderTexts: texts,
                      basePrompt: Fibo2JSON.object(markTextImages(o, marker: "")).dumps())
    }

    /// `check_inputs` for an edit prompt: nil when valid, else the reason.
    public static func editPromptProblem(_ prompt: String) -> String? {
        guard let parsed = try? Fibo2JSON.parse(prompt), case .object(let o) = parsed else { return nil }
        guard o.contains("edit_instruction") else {
            return "to edit an image, the prompt has to be the edit instruction, or a JSON caption of the result "
                + "with an \"edit_instruction\" key"
        }
        if let inner = o["structured_caption"] {
            let ok: Bool
            switch inner { case .object, .string: ok = true; default: ok = false }
            if !ok || o.members.count > 2 {
                return "an edit prompt with a \"structured_caption\" key has to hold the caption of the result there "
                    + "(a JSON object or a string) and nothing else but the \"edit_instruction\""
            }
        }
        return nil
    }
}
