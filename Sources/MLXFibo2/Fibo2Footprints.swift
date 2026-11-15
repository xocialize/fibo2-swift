// Declared memory footprints (efficiency contract: resident + peak activation), MEASURED with
// `Fibo2Gate render --drop-te` (M5 Max, mlx-swift 0.32.3, Release, MLX pool capped at 2 GB, 2026-11-15, AB-T-0493).
// This is the package's own memory path: DiT (bf16 / quantized at load) plus the fp32 VAE encoder and decoder
// resident; per request the Qwen3-VL-4B encoder loads, encodes and is released, the images to edit are VAE-encoded
// one at a time in row bands, then denoise and decode (row bands).
//
//   DiT tier   resident    encode phase (+ encoder)    denoise + decode
//   bf16       16.59 GB    27.48 GB                    18.99–21.52 GB
//   int8        9.14 GB    (resident + the same)       11.58 GB (1024² text-to-image)
//   int4        7.50 GB    (resident + the same)        9.94 GB (1024² text-to-image)
//
// Activation above resident, GB (bf16; the quantized tiers are the same to 0.01):
//
//   request                                  encode phase   denoise + decode
//   text-to-image 1024²                       9.22           2.40
//   text-to-image 1424×800, 2 text images     9.54           2.58
//   edit, 1 image (720×1440)                 10.89           2.38
//   edit, 2 images, ~1 MP each               10.89           2.92
//   edit, 5 images, ~1 MP each               10.75           4.93
//
// The per-request encoder sets the peak at every point: 10.89 GB. Declared: resident rounded up 2–3 %, activation
// 11.3 GB (+3.8 %). bf16 declares 28.3 GB, int8 20.7 GB, int4 19.0 GB.
//
// An int8 encoder does not lower the transient: it is quantized from the bf16 file at load (9.42 GB against 9.22 GB).
// It only helps when the encoder is kept resident (`keepEncoderResident`).
//
// Time per step, two DiT passes (bf16, quiet machine; the quantized tiers are within 5 %): text-to-image 1024² 3.5 s;
// edit with 1 image 7.6 s, 2 images 12.8 s, 5 images 38.8 s.

import Fibo2
import MLXToolKit

enum Fibo2Footprints {
    /// Output area the footprints hold for: the band Fibo 2.0 was trained in, within 10 % of 1024².
    static let envelopeMaxPixels = 1_153_434
    static let envelopeMinPixels = 256 * 256
    /// Images to edit in one request: the model's own maximum, measured.
    static let envelopeMaxImages = Fibo2ImagePrep.maxImages

    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 17_000_000_000, peakActivationBytes: 11_300_000_000),
        QuantFootprint(quant: .int8, residentBytes: 9_400_000_000, peakActivationBytes: 11_300_000_000),
        QuantFootprint(quant: .int4, residentBytes: 7_700_000_000, peakActivationBytes: 11_300_000_000),
    ]

    /// The conditioner's weights (language model + vision tower), added to the resident term when kept resident.
    static func encoderBytes(_ quant: Fibo2TextEncoderQuant) -> UInt64 {
        quant == .int8 ? 5_200_000_000 : 9_200_000_000
    }
}
