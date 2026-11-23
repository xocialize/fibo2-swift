// swift-tools-version: 6.2
// fibo2-swift — Swift/MLX port of briaai/Fibo-2.0-dev, Fibo-2.0-edit-turbo and RMBG-3.0 (weights CC BY-NC 4.0, gated; see PORTING-SPEC.md §1 — the
// port code here is MIT): an 8.1B single-stream DiT of Z-Image blocks with a Perceiver "gist" resampler and five
// cross-attention injection blocks, conditioned on 15 hidden-state taps of Qwen3-VL-4B-Instruct (which also sees the
// images to edit and images of the caption's text), in the FLUX.2 VAE's latent space. One set of weights for
// text-to-image, instruction edits, masked edits and multi-reference edits.
// Reference = diffusers 0.43 `BriaFibo2Pipeline` (Apache-2.0); goldens from mlxengine-image/WIP/fibo2-oracle
// (fp32 CPU torch). Task AB-T-0493.

import PackageDescription

let package = Package(
    name: "Fibo2",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Fibo2", targets: ["Fibo2"]),
        // MLXEngine packages: Fibo2DevPackage (fibo-2-dev; textToImage + imageEdit) and Fibo2EditTurboPackage
        // (fibo-2-edit-turbo; imageEdit) and Fibo2RMBGPackage (rmbg-3; matting). Non-commercial weights: Gaea plugin
        // models (AB-D-0226), never a shipping route.
        .library(name: "MLXFibo2", targets: ["MLXFibo2"]),
        .executable(name: "Fibo2Gate", targets: ["Fibo2Gate"]),
    ],
    dependencies: [
        // 0.32.3 carries the NAX split-K GEMM fix (mlx#3810); the DiT's FFN down-projection has K = 9536.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // MLXLMCommon.THW, the image grid type of the Qwen3-VL backbone's API.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.32.3"),
        // Qwen3-VL conditioner backbone; ≥ 0.7.0: hiddenStates taps with images (vision merge, deepstack, M-RoPE).
        .package(url: "https://github.com/xocialize/qwen3vl-mlx-swift", from: "0.7.0"),
        // The FLUX.2 VAE: packed encode / decode with the bn statistics, row-band tiling.
        .package(url: "https://github.com/xocialize/flux2-vae-mlx-swift", from: "0.8.0"),
        // MLXEngine contract (MLXToolKit) + conformance gates, for the wrapper only.
        // From 0.74.1 the engine takes the fleet's mlx-swift fork (0.32.4); a root package that names the fork gets
        // it (the Gaea plugin does). Built on its own, this package resolves upstream 0.32.3 and an earlier engine.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.68.0"),
    ],
    targets: [
        .target(
            name: "Fibo2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
            ],
            path: "Sources/Fibo2"
        ),
        .target(
            name: "MLXFibo2",
            dependencies: [
                "Fibo2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXFibo2"
        ),
        .testTarget(
            name: "MLXFibo2Tests",
            dependencies: [
                "MLXFibo2", "Fibo2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXFibo2Tests"
        ),
        .executableTarget(
            name: "Fibo2Gate",
            dependencies: ["Fibo2", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXRandom", package: "mlx-swift"),
                           .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
                           .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                           .product(name: "Tokenizers", package: "swift-transformers"),
                           .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift")],
            path: "Sources/Fibo2Gate"
        ),
    ]
)
