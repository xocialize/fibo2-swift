# fibo2-swift

Swift/MLX port of [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev): text-to-image, instruction
editing, masked editing and multi-image editing (up to 5 images) in one set of weights.
- **DiT:** 8.1B parameters, 44 single-stream Z-Image blocks. A Perceiver resampler turns the prompt into 256–512
  "gist" tokens; five blocks also cross-attend to the text encoder directly.
- **Conditioner:** 15 hidden-state taps of Qwen3-VL-4B-Instruct, which also sees the images to edit and small
  images of the text the caption asks for.
- **VAE:** the FLUX.2 VAE.

> **Licence: the Fibo 2.0 weights are CC BY-NC 4.0.** Non-commercial use only; commercial use needs an agreement
> with Bria (<https://platform.bria.ai/licensing/pricing?model=fibo-2>). The repo is gated: request access on the
> model page, then fetch it with your own Hugging Face token. Nothing is mirrored here or elsewhere.
>
> The Swift port code in this repository is MIT (`LICENSE`). That does not change the weights' terms.
>
> **Status: a research model for [Gaea](https://github.com/xocialize-code/gaea) plugins, not a product asset.** It
> is offered by name only and opt-in, with the licence shown (fleet decision AB-D-0226). See `PORTING-SPEC.md` §1.

The reference is diffusers 0.43 `BriaFibo2Pipeline`. The reference facts, gates and measurements are in
`PORTING-SPEC.md`.

## Use

```swift
import Fibo2

let pipeline = try await Fibo2Pipeline.load(snapshot: snapshotURL)   // a briaai/Fibo-2.0-dev snapshot
var request = Fibo2Pipeline.Request(prompt: captionJSON)             // a VGL caption, or a short description
request.negativePrompt = Fibo2Prompt.cardNegativeT2I
let result = try pipeline.generate(request)
try result.image.write(to: outURL)

// Edit: the prompt is the instruction; a mask (white = regenerate) confines it.
var edit = Fibo2Pipeline.Request(prompt: "replace the jacket with a red leather one", images: [photo])
```

Fibo 2.0 was trained on **structured JSON captions** ([VGL](https://docs.bria.ai/vgl)): `short_description`,
`objects`, `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`,
`text_render` and so on. Send that JSON as the prompt. A short plain description also works, with less control.
Bria's [`bria-vgl`](https://github.com/Bria-AI/vgl) writes such captions with Gemini; this package doesn't call it.

```bash
swift run -c release Fibo2Gate render <snapshot> --prompt @prompts/sign.json --negative card --out sign.png
```

```bash
swift run -c release Fibo2Gate render <snapshot> --prompt "replace the dog with a black cat" --image dog.png --out cat.png
```

`--quant bf16|int8|int4`, `--steps`, `--guidance`, `--seed`, `--size WxH`, `--mask`, `--drop-te` (load the encoder
per request, as the engine package does).

## Parity

Against diffusers in fp32 on the CPU (`mlxengine-image/WIP/fibo2-oracle`). fp32 gates need `MLX_ENABLE_TF32=0`.

| Piece | Gate |
|---|---|
| Prompt shaping, 9 cases | Shaped text, token ids and output sizes equal; VAE crops and Qwen3-VL's images byte-identical |
| Text images | 49/49 sizes equal to Pillow's; pixels differ by rasterizer only |
| Conditioner | fp32: bundles cos 1.000000 on text, text images, one and two edit images. bf16: 0.9954–0.99995 |
| DiT | fp32: velocity max abs diff 0.0000 (text-to-image and edit). bf16: cos 0.99974 / 0.99982 |
| VAE (flux2-vae-mlx-swift) | Encode and decode cos 0.999999 |
| End to end, same noise | Text-to-image fp32: 77.4 dB, max diff 1 LSB. Edit, bf16 against fp32: 41.1 dB |

## Lanes, memory and time (M5 Max)

| DiT lane | Velocity cos vs fp32 | Resident | Peak (encoder loaded per request) |
|---|---|---|---|
| bf16 | 0.99974 | 16.6 GB | 27.5 GB |
| int8 | 0.9989 | 9.1 GB | 20.0 GB |
| int4 (attention and FFN-down at 8 bits) | 0.9925 | 7.5 GB | 18.4 GB |

Two DiT passes per step (classifier-free guidance 5). Per step: text-to-image at 1024² 3.5 s; an edit with one
image 7.6 s, two images 12.8 s, five images 38.8 s. The card's 50 steps take about 3 minutes for text-to-image and
6 minutes for a one-image edit.

## MLXEngine package

`MLXFibo2.Fibo2DevPackage` (PackageID `fibo-2-dev`): `textToImage` and `imageEdit`. Weight licence `CC-BY-NC-4.0`,
declared package-locally and not allowlisted. Weights come from `briaai/Fibo-2.0-dev`, pinned by commit, into the
engine's model store.
- `T2IRequest`: `negativePrompt` defaults to the model card's text-to-image negative. Sizes are multiples of 16
  around one megapixel (at most 1.1 × 1024² of area).
- `IEditRequest`: 1–5 images. `metaData["mask"]` is a base64 PNG the size of the single image; white is the region
  to regenerate.
- `metaData["textImages"] = false` skips drawing the caption's texts for Qwen3-VL.

## Not here

- **Fibo 2.0 Edit Turbo** (few steps, no guidance): the same architecture, so a second checkpoint for this code.
  It is gated separately.
- LoRA loading and the model card's style presets.
- A local caption writer. A `fibo2` profile for the fleet's prompt-enhance plugin is the planned route.
