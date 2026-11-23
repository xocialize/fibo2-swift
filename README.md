# fibo2-swift

Swift/MLX port of Bria's Fibo 2.0: [Fibo 2.0 Dev](https://huggingface.co/briaai/Fibo-2.0-dev) (text-to-image,
instruction editing, masked editing and multi-image editing of up to 5 images in one set of weights) and
[Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) (the same edits in 4 steps), and
[RMBG 3.0](https://huggingface.co/briaai/RMBG-3.0), Edit Turbo fine-tuned to render an alpha matte (background
removal).
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

| DiT lane | Velocity cos vs fp32 | Resident | Process peak (encoder loaded per request) | Declared |
|---|---|---|---|---|
| bf16 | 0.99974 | 16.6 GB | 29.9 GB | 30.4 GB |
| int8 | 0.9989 | 9.1 GB | 22.5 GB | 22.8 GB |
| int4 (attention and FFN-down at 8 bits) | 0.9925 | 7.5 GB | 20.7 GB | 21.1 GB |

The peak is the encode phase of an edit: the text encoder (8.9 GB) is loaded for each request and released before
denoising, which itself needs 2.4–4.9 GB over resident.

Two DiT passes per step (classifier-free guidance 5). Per step: text-to-image at 1024² 3.5 s; an edit with one
image 7.6 s, two images 12.8 s, five images 38.8 s. The card's 50 steps take about 3 minutes for text-to-image and
6 minutes for a one-image edit.

## MLXEngine package

`MLXFibo2.Fibo2DevPackage` (PackageID `fibo-2-dev`): `textToImage` and `imageEdit`. Weight licence `CC-BY-NC-4.0`,
declared package-locally and not allowlisted. Weights come from `briaai/Fibo-2.0-dev`, pinned by commit, into the
engine's model store. `Fibo2EditTurboPackage` is described below.
- `T2IRequest`: `negativePrompt` defaults to the model card's text-to-image negative. Sizes are multiples of 16
  around one megapixel (at most 1.1 × 1024² of area).
- `IEditRequest`: 1–5 images. `metaData["mask"]` is a base64 PNG the size of the single image; white is the region
  to regenerate.
- `metaData["textImages"] = false` skips drawing the caption's texts for Qwen3-VL.

## Fibo 2.0 Edit Turbo

The same architecture, distilled from Dev for editing: 4 steps, no guidance (one DiT pass a step, and a negative prompt
does nothing). Its repo ships the same text encoder, tokenizer, VAE and fonts as Dev's, byte for byte, so only its
`transformer/` is needed beside a Dev snapshot:

```bash
swift run -c release Fibo2Gate render <edit-turbo snapshot> --shared <dev snapshot> --prompt "replace the dog with a black cat" --image dog.png --steps 4 --guidance 1 --out cat.png
```

- **Against diffusers** (bf16 on both sides, same noise, 384×768, 4 steps): one image 45.0 dB, two images 27.6 dB
  with the same composition.
- **Lanes**, same seed, one image at 720×1440: int8 is 46.8 dB from bf16, int4 37.7 dB. Four unguided steps amplify
  far less than Dev's fifty guided ones.
- **Memory** is Dev's, lane for lane.
- **Time:** 4 DiT passes instead of 100. From Dev's measured time per pass, about 15 s of denoising for a one-image
  edit at one megapixel, plus about 5 s to encode and decode. Not yet timed on a quiet GPU.
- **Engine package:** `MLXFibo2.Fibo2EditTurboPackage` (`fibo-2-edit-turbo`), `imageEdit` only. With a Dev snapshot in
  the model store it fetches only the 16 GB transformer.

## RMBG 3.0

Bria's background remover is the same architecture again: Edit Turbo fine-tuned and distilled to render a picture's
alpha matte in 4 steps. Its repo holds a transformer only, so the port is one file, `Fibo2Matte.swift`: the fixed
instruction the model was trained on, the card's settings (4 steps, no guidance, seed 0), and the picture's way in
and out. The weights are CC BY-NC 4.0, "the same terms as the Fibo 2.0 family"; commercial terms are at
<https://platform.bria.ai/licensing/pricing?model=rmbg-3.0>.

```swift
let pipeline = try await Fibo2Pipeline.load(snapshot: rmbgSnapshot, shared: devOrEditTurboSnapshot)
let result = try pipeline.generate(Fibo2Matte.request(for: photo))
let matte = Fibo2Matte.matte(from: result.image, width: photo.width, height: photo.height)   // white = keep
try Fibo2PNG.write(Fibo2Matte.cutout(photo, matte: matte), to: outURL)
```

```bash
swift run -c release Fibo2Gate matte <RMBG-3.0 snapshot> --shared <dev snapshot> --image photo.jpg --out matte.png --cutout cutout.png
```

- **The whole frame.** The model card's snippet calls the pipeline without a size, which centre-crops the picture to
  the size trained for its aspect ratio, then stretches the matte over the whole picture. The strips the crop cut
  off get no matte and the rest lands slightly off. Here the picture is stretched to that same size instead (a few
  pixels in 1024 at most) and the matte stretched back, so it covers the frame and lines up. `--fit crop` is the
  card's way: on a 5071×3294 photo the two mattes differ in 0.26 % of the pixels at the half-way line.
- **No oracle run.** The transformer is Edit Turbo's architecture with other weights, and Edit Turbo's parity gates
  cover the code path. What was checked for RMBG 3.0: the render is grey (largest R/G/B spread 6–18 of 255, where a
  wrong instruction gives a coloured picture), and the mattes of a fox, a crow on a branch and three doves are right
  by eye.
- **Memory and time** (M5 Max, encoder loaded per request, the process's peak): bf16 29.4 GB, int8 21.5 GB, int4
  19.9 GB, all under the footprints declared for a one-image edit. About 20 s a picture: 16 s of denoising, 2 s to
  encode once the text encoder's file is cached, 1–2 s to decode.
- **Lanes** against bf16, same picture: int8 moves 0.007 % of the pixels across the half-way line, int4 0.06 %.
- **Engine package:** `MLXFibo2.Fibo2RMBGPackage` (`rmbg-3`), `matting` only. `preferredKind: .binary` thresholds
  the matte at 128; `metaData["seed"]` (default 0) redraws it. Weights: `briaai/RMBG-3.0` pinned by commit for the
  transformer; the text encoder, VAE and fonts from an Edit Turbo or Dev snapshot in the store, else from Edit
  Turbo's repo.

## Not here

- LoRA loading and the model card's style presets.
- A local caption writer. A `fibo2` profile for the fleet's prompt-enhance plugin is the planned route.
