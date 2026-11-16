# fibo2-swift — Porting spec: Fibo 2.0 Dev → Swift/MLX

Task AB-T-0493 · channel decision AB-D-0226 (a Gaea plugin model) · oracle `../fibo2-oracle` (fp32 CPU torch goldens)

## 1. Licence and data

- **Weights:** CC BY-NC 4.0 (`license: other`, `license_name: bria-fibo`). The repo is gated behind Bria's
  non-commercial access form; commercial use needs an agreement with Bria.
- **Components in the snapshot:** Qwen3-VL-4B-Instruct (Apache-2.0), the FLUX.2 VAE, two fallback fonts
  (Aileron, CC0; DejaVu Sans, Bitstream Vera licence).
- **Corpus:** Bria states the model is "trained exclusively on licensed data" (its ~1B-image set). The weight
  licence is the only block.
- **Channel:** Gaea plugin only (AB-D-0226): named-only and opt-in, with the licence and `commercial_use: false`
  shown. It never enters a shipping route. The engine package declares `CC-BY-NC-4.0` package-locally; it is not
  allowlisted.
- **No mirror.** The package reads Bria's own repo, pinned to `b01e5a34a0a9540c65f585c2c281535deabac16d`. Each user
  fetches it with a Hugging Face token whose account has access. Quant tiers are made at load.
- **Port code:** MIT. Reference code: diffusers (Apache-2.0).
- **Embedded font:** Pillow's built-in font (a TrueType subset of Aileron Regular, CC0), 12.7 KB, in
  `Fibo2DefaultFont.swift`.

## 2. Reference facts (diffusers 0.43.0 `BriaFibo2Pipeline`, transformers 5.23)

### 2.1 Transformer `BriaFibo2Transformer2DModel` (937 tensors, 16.25 GB bf16, 8.13B)

- **Shape:** dim 3584, 28 heads × 128, 44 blocks, 2 noise-refiner blocks, FFN 9536 (`(dim·8/3)/64·64`),
  SiLU-gated (`w2(silu(w1 x) · w3 x)`), all Linears bias-free except the embedders and adaLN.
- **Tokens:** 2×2 patches of the 32-channel latent, features ordered (channel, dy, dx). That is exactly the FLUX.2
  VAE's packed 128-channel latent, so the port works on packed latents and never re-patchifies.
- **Block (Z-Image's):**
  - `x += tanh(gate_msa) · norm2(attn(norm1(x) · (1 + scale_msa)))`, then the same for the FFN.
  - adaLN: one Linear 256 → 4·dim per block, chunks (scale_msa, gate_msa, scale_mlp, gate_mlp).
  - RMSNorm everywhere (eps 1e-5), QK-RMSNorm per head; RoPE applied in fp32.
- **Per-token modulation:** image tokens take the real timestep; clean tokens (gist, context) take t = 0.
  The noise refiner uses one global modulation when there are no context images.
- **Perceiver resampler:** 2 layers (cross-attention to the text, self-attention, FFN; no RoPE), 512 learned
  queries. The first N are used: N = 256–512 from the prompt's token count (256 → 2048 tokens map linearly,
  rounded up to 64). The gist does not depend on the timestep, so the port computes it once per prompt.
- **Stream order:** `[gist | image | context images…]`.
- **Injection blocks 6, 14, 22, 30, 38:** `x += tanh(cross_attn_gate) · norm2(cross_attn(norm1(x), text bundle k))`,
  between the self-attention and the FFN. Text features are 7680 wide.
- **RoPE:** 3 axes (32, 48, 48), θ 256, angles in float64 then fp32.
  - gist token i: (1 + i, 0, 0);
  - image: plane 513, (row, column);
  - context image k: plane 514 + k, its own rows and columns.
- **Timestep:** fp32 sinusoid (256, cos | sin) of `t · 1000` → Linear 256 → 1024 → SiLU → 256.
- **Head:** LayerNorm (no affine, eps 1e-6) · (1 + Linear(SiLU(t_emb))) → Linear to 128, image tokens only.
- **Loader renames:** `t_embedder.mlp.0/2` → `mlp_in/mlp_out`; `final_layer.adaLN_modulation.1` → `adaLN_linear`;
  the five `layers.N.cross_attn_gate` scalars → one `cross_attn_gates` array.

### 2.2 Text encoder: Qwen3-VL-4B-Instruct, 15 hidden-state taps

- **Prompt:** `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, no system turn, ≤ 4096 tokens,
  one prompt per forward (no padding, no mask).
- **Images in the prompt:** the images to edit first (each ≤ 768 px a side), then one image per `text_render`
  text. Each sits at a `<|vision_start|><|image_pad|><|vision_end|>` marker; the pad expands to one token per
  32-px cell. Processor: min 3136 / max 1 605 632 pixels, BICUBIC, mean = std = 0.5.
- **Forward:** the full Qwen3-VL one: vision features merged, deepstack after layers 0–2, spatial M-RoPE.
- **Bundles:** six, each three `hidden_states[k]` concatenated (→ 7680):
  - Perceiver: (9, 20, 31);
  - injection blocks: (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35).
  - Each layer is scaled to unit RMS per token in fp32 (`x · rsqrt(mean(x²) + 1e-6)`) before the concat.
- **Config trap:** `text_encoder/config.json` is in the transformers 5.x layout. `rope_theta` (5 000 000) sits
  under `rope_parameters`. qwen3vl-mlx-swift < 0.7.0 fell back to 1e6.
- **transformers 5.x** records `hidden_states[1…3]` before the deepstack add. Fibo's taps start at 5, so it makes
  no difference here.

### 2.3 Prompt shaping (`_parse_prompt`, `_get_edit_prompt`)

- A JSON-object prompt is cleaned as training did, then written compactly in the caller's key order:
  - null, "", {}, [] and NaN are dropped at every depth;
  - `aesthetic_score` and `preference_score` are dropped at the top level and inside each section;
  - a caption of exactly `short_description` + `text_render` keeps both keys, even when empty.
- Anything else (a short description, an instruction) goes as typed.
- **Edit:** one JSON object: `image` (or `image_1…n`) markers, `structured_caption` when the prompt has one,
  then `edit_instruction`.
- **Text images:** each `text_render` entry with text gets `image_tokens`: the vision marker. A generation also
  encodes a base prompt with `image_tokens: ""` and runs on it while the noise level is above 0.9.
- **Negative:** encoded as given. Default: the null caption (every key, every value empty).
- The serializer follows Python's `json.dumps(…, separators=(",", ":"), ensure_ascii=False)`; `Fibo2JSON` keeps
  key order and Python's number formatting.

### 2.4 Images

- **Trained size for an aspect ratio:** multiples of 16, sides ≥ 64, area within 10 % of 1024²; the one that
  crops least, then closest to 1024², then widest.
- **An image to edit:** `convert("RGB")`, scaled to cover the size (LANCZOS, sides rounded up), centre crop.
  - One image → the output size. Several → each to the trained size for its own aspect ratio.
- **Mask:** the region (≥ 128) is painted (128, 128, 128) at the image's own size, before the resize.
- **Qwen3-VL's copy:** `thumbnail((768, 768))`, BICUBIC.
- **Pixels:** in `2·(x/255) − 1`; out `round(clamp(x/2 + 0.5) · 255)`.

### 2.5 Text images (`text_images.py`)

- Whitespace collapsed, lines of 256 characters, black on white, 2 px padding, size 20.
- **One-font path** (every character is ASCII or in Pillow's built-in font): the line box is Pillow's `textbbox`.
- **Mixed path:** each character in the first font that has it (built-in, Aileron, DejaVu Sans), on one
  baseline; the line is ascent + descent (20 + 5) tall and the sum of the advances wide.
- **The port:** the same layout from FreeType's hinted metrics, drawn with CoreText.
  - Built-in font: a per-character table (advance and ink extents) taken from Pillow 13.0 / FreeType 2.14.3.
  - Fallback fonts: the scaled advance rounded to a pixel, plus `kern`-table kerning as `FT_Get_Kerning` fits it.
  - Sizes equal Pillow's; pixels differ by rasterizer (FreeType hints, CoreGraphics doesn't).

### 2.6 Sampler

- `FlowMatchEulerDiscreteScheduler`, dynamic exponential shift: `σ' = e^μ / (e^μ + (1/σ − 1))` over
  `linspace(1, 1/steps, steps)`, μ linear in the image token count (0.5 at 256, 1.15 at 4096), terminal 0.
  Computed in float32 after a float64 linspace.
- Card defaults: 50 steps, guidance 5 (diffusers' own default is 30 steps).
- CFG: the negative runs as a second pass; combine and Euler update in fp32; latents stay fp32.
- Decode: bn de-norm in packed space, FLUX.2 decoder.

## 3. Gates (`Fibo2Gate`, against `../fibo2-oracle/goldens`)

fp32 gates need `MLX_ENABLE_TF32=0`: on an M5 GPU, MLX runs fp32 matmuls in TF32 by default (MLXEngine R-NUM-1),
about 3e-4 of the largest value per GEMM.

| Gate | Result |
|---|---|
| `prompts` (9 cases) | shaped text, token ids, output sizes equal; VAE crops and Qwen3-VL images byte-identical; pixel values within 1 LSB |
| `sizes` | 50/50 |
| `schedule` | 5/5 bit-exact |
| `textimg` | 49/49 sizes equal; advances 337/337 (Aileron), 4708/4708 (DejaVu); mean abs pixel diff 9.6 / 255 |
| `vae` | encode and decode cos 0.999999, untiled and row bands; 59.7 dB |
| `encoder` fp32 | bundles cos 1.000000, all 9 calls (text, text images, 1 and 2 edit images) |
| `dit` fp32 | every stage cos 1.000000; velocity max abs diff 0.0000 (T2I and edit layouts) |
| `encoder` bf16 | 0.99995 text-only; 0.9954–0.9991 with images (torch's own bf16: 0.987–0.997) |
| `dit` bf16 | velocity cos 0.99974 (T2I), 0.99982 (edit) |
| e2e T2I, same noise, 512², 20 steps, CFG 5 | fp32 77.4 dB, max diff 1 LSB; bf16 37.6 dB |

## 4. Lanes, memory and time (measured, M5 Max, MLX pool capped at 2 GB)

The package keeps the DiT and the VAE resident. The encoder loads per request and is released before denoising.

| DiT lane | Recipe | Velocity cos vs fp32 (T2I / edit) | Resident | Process peak | Declared total |
|---|---|---|---|---|---|
| bf16 | upstream weights | 0.99974 / 0.99982 | 16.59 GB | 29.85 GB | 30.4 GB |
| int8 | affine, group 64 | 0.9989 / 0.9996 | 9.14 GB | 22.53 GB | 22.8 GB |
| int4 | FFN up-projections (`w1`, `w3`) at 4 bits; attention projections and `w2` at 8 | 0.9925 / 0.9952 | 7.50 GB | 20.72 GB | 21.1 GB |

- Resident is MLX's pool; the process peak is `phys_footprint` on a one-image edit, about 2.4 GB above the pool's
  peak. The declared totals cover the process peaks.
- v0.1.1 drops the checkpoint dictionary before quantizing: the int8 load peaked at 24.88 GB in v0.1.0, 9.77 GB now.
- Kept at the load dtype in every quantized lane: the patch embedding, the timestep MLP, the final layer, each
  block's adaLN Linear.
- Uniform 4-bit reads 0.979 / 0.985. Six recipes were probed; see `Fibo2Weights.int4EightBitLayers`.
- Same seed, 1024², 30 steps, a caption with two texts: all three lanes draw the same scene with both texts
  right. int8 is 26.4 dB from bf16, int4 20.0 dB (30 guided steps amplify small differences).
- **Activation over resident:** the encode phase sets it, 9.22 GB (text-to-image) to 10.89 GB (an edit). Denoise
  and decode take 2.4 GB at 1024², 4.93 GB with five images. Declared 13.4 GB, which covers the process peak.
- An int8 encoder doesn't lower that transient (it is quantized from the bf16 file at load); it only helps when
  the encoder is kept resident.
- **Time per step** (two passes): text-to-image 1024² 3.5 s; edit with 1 image 7.6 s, 2 images 12.8 s,
  5 images 38.8 s. So 50 steps: about 3 min, 6 min, 11 min and 32 min.
- **e2e edit**, same noise, 384×768, 12 steps: Swift bf16 against diffusers fp32 41.1 dB.

## 5. Fibo 2.0 Edit Turbo (v0.2.0)

- **Repo:** `briaai/Fibo-2.0-edit-turbo` @ `7b133d62d92e897f1468af0427a73cc41a03cb5d`, gated separately, CC BY-NC 4.0.
- **Same architecture and scheduler config** as Dev (`transformer/config.json` and `scheduler_config.json` equal).
  Only the transformer's weights differ: `text_encoder/model.safetensors`, `vae/…`, `processor/tokenizer.json`
  and the fonts have Dev's LFS sha256.
- **Sampling:** 4 steps, `guidance_scale` 1 (no negative pass). Editing only; the card sends text-to-image to Dev.
- **Loading:** `Fibo2Pipeline.load(snapshot:shared:)` reads `transformer/` from one snapshot and the shared
  components from another. The package declares two weight sources in the Turbo repo (`transformer`, `shared`);
  a Dev snapshot already in the store satisfies `shared`.
- **Reference:** diffusers in bf16 on MPS (`e2e.py` with `FIBO2_TRANSFORMER` and `FIBO2_BF16=1`); the fp32 CPU
  reference needs 50 GB, which the machine didn't have free.
- **e2e, same noise, 384×768, 4 steps** (Swift bf16 against torch bf16): one image 45.0 dB; two images 27.6 dB,
  same composition. This is also the first end-to-end comparison of a multi-image edit.
- **Lanes** (same seed, one image, 720×1440): int8 46.8 dB from bf16, int4 37.7 dB. Memory equals Dev's.
- **Time:** not measured on a quiet GPU. From Dev's 3.8 s per pass with one image, about 15 s for the 4 steps.

## 6. Known hazards

- **Speed.** Dev is the non-distilled base: 2 DiT passes per step, 50 steps by the card. Fibo 2.0 Edit Turbo
  (same architecture, few steps, no CFG) is the practical edit model; it is gated separately.
- **TF32.** With it on (the default), fp32 encoder bundles with images read 0.9993–0.9997. Not a port difference.
- **Reboot.** A reboot moves the Metal toolchain mount; `swift package clean` before the next build.
- **Memory gauge.** `memory_pressure`'s free percentage counts inactive pages as free, and a process's RSS
  misses its Metal buffers. `../fibo2-oracle/mem_guard.sh` uses total − Σ RSS − wired.
