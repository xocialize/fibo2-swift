// Fibo 2.0's sampler: diffusers `FlowMatchEulerDiscreteScheduler` with dynamic exponential shifting, as
// `BriaFibo2Pipeline` drives it. Sigmas run from 1 down to 1 / steps, shifted by the image's token count, with a
// terminal 0; the transformer takes the noise level in [0, 1]; the update is plain Euler in fp32.

import Foundation
import MLX

public enum Fibo2Schedule {
    public static let defaultSteps = 50
    public static let defaultGuidance: Float = 5.0
    /// A generation shows the images of its caption's text to Qwen3-VL only once the noise level is at or below this.
    public static let textImagesMaxNoiseLevel: Float = 0.9

    // scheduler_config.json
    static let baseImageSeqLen = 256.0, maxImageSeqLen = 4096.0, baseShift = 0.5, maxShift = 1.15

    /// `calculate_shift`.
    public static func mu(imageTokens: Int) -> Double {
        let m = (maxShift - baseShift) / (maxImageSeqLen - baseImageSeqLen)
        return Double(imageTokens) * m + (baseShift - m * baseImageSeqLen)
    }

    /// The shifted sigmas, `steps + 1` values ending in 0. The reference computes `np.linspace(1, 1 / steps, steps)` in
    /// float64, casts to float32, then shifts in float32: exp(mu) / (exp(mu) + (1 / t − 1)).
    public static func sigmas(steps: Int, imageTokens: Int) -> [Float] {
        let e = Float(exp(mu(imageTokens: imageTokens)))
        let stop = 1.0 / Double(steps)
        let delta = steps > 1 ? (stop - 1.0) / Double(steps - 1) : 0
        var out: [Float] = (0..<steps).map { i in
            let t = Float(i == steps - 1 && steps > 1 ? stop : 1.0 + Double(i) * delta)
            return e / (e + (1 / t - 1))
        }
        out.append(0)
        return out
    }

    /// The noise level the transformer is given at a sigma: (sigma · 1000) / 1000 in float32, as the pipeline computes.
    public static func noiseLevel(_ sigma: Float) -> Float { (sigma * 1000) / 1000 }

    /// One Euler update: x + (σ_next − σ) · velocity, in fp32.
    public static func step(_ x: MLXArray, velocity: MLXArray, sigma: Float, next: Float) -> MLXArray {
        x + (next - sigma) * velocity.asType(.float32)
    }
}
