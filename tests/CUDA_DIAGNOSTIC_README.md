# CUDA ZFP Diagnostic Benchmark

Standalone benchmark that profiles individual pipeline stages of the ZFP 3D float
decode kernel on NVIDIA GPUs. Produces the same breakdown table as the Metal
backend's `ZFP_METAL_PROFILE=3` diagnostic mode, enabling direct Apple Silicon
vs NVIDIA comparison.

## What it measures

| # | Kernel | Description |
|---|--------|-------------|
| 1 | `memcopy_3d` | Pure memory bandwidth — read 64 `uint` + write 64 `float` per block |
| 2 | `bitread_only_3d` | Bitstream reading only (discard decoded values) |
| 3 | `bitplane_only_3d` | Bitstream + 64-bit scatter to `ub[64]` (CUDA's native approach) |
| 4 | `bitplane_split32_3d` | Bitstream + split-32 scatter (Metal's optimized approach, for comparison) |
| 5 | `transform_only_3d` | Permute + uint2int + inverse lifting transform + dequantize |
| 6 | `full_decode_3d` | Complete production decode pipeline |

## Prerequisites

- NVIDIA GPU with CUDA capability ≥ 3.0
- CUDA Toolkit (nvcc) ≥ 10.0
- No project dependencies — the file is fully self-contained (no `zfp.h` needed)

## Build

```bash
# Determine your GPU's compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile (replace sm_XX with your GPU arch):
nvcc -O3 -arch=sm_80 tests/cuda_diagnostic_bench.cu -o cuda_diag_bench   # A100
nvcc -O3 -arch=sm_89 tests/cuda_diagnostic_bench.cu -o cuda_diag_bench   # RTX 4090 / L40
nvcc -O3 -arch=sm_90 tests/cuda_diagnostic_bench.cu -o cuda_diag_bench   # H100
nvcc -O3 -arch=sm_70 tests/cuda_diagnostic_bench.cu -o cuda_diag_bench   # V100
```

If you don't know your arch, use `gencode` to target multiple:

```bash
nvcc -O3 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  tests/cuda_diagnostic_bench.cu -o cuda_diag_bench
```

## Run

```bash
# Default: 217^3 (~10.2M elements), rate 8, 5 repeats
./cuda_diag_bench

# Custom grid size (N^3)
./cuda_diag_bench 256          # 256^3 ≈ 16.8M elements

# Custom rate
./cuda_diag_bench 217 16       # rate 16 (bits per value)
./cuda_diag_bench 217 32       # rate 32

# More repeats for stability
./cuda_diag_bench 217 8 10     # 10 repeats
```

## Example output

```
=== CUDA ZFP Diagnostic Benchmark ===
Grid:           217 x 217 x 217 = 10218313 elements (39.0 MB)
Padded:         220 x 220 x 220
Total blocks:   166375
Rate:           8 bits/value (512 bits/block)
Stream size:    10.2 MB
Repeats:        5

GPU:            NVIDIA A100-SXM4-80GB
Compute:        8.0
SMs:            108
...

=======================================================================
CUDA ZFP Diagnostic Results — 3D Float Decode (217^3, rate 8)
=======================================================================
Kernel                 | Description                                        | Time(ms) |   GB/s  | % of Full
-----------------------|----------------------------------------------------|----------|---------|----------
memcopy_3d             | Pure bandwidth: read 64 uint + write 64 float      |     0.12 |   325.0 |      18%
bitread_only_3d        | Bitstream read only (discard results)               |     0.34 |   114.7 |      52%
bitplane_only_3d       | Bitstream + 64-bit scatter (CUDA native)            |     0.45 |    86.7 |      69%
bitplane_split32       | Bitstream + split-32 scatter (Metal approach)        |     0.48 |    81.3 |      74%
transform_only         | Permute + uint2int + inv_lift + dequant             |     0.08 |   487.5 |      12%
full_decode_3d         | Complete production decode pipeline                 |     0.65 |    60.0 |     100%
=======================================================================
```

*(Numbers above are illustrative — actual results depend on your GPU.)*

## Comparing with Metal results

Run the Metal diagnostic on Apple Silicon:

```bash
ZFP_METAL_PROFILE=3 ctest --test-dir build-metal-native-codec -R "^testZfpCuda3dFloat$" -V
```

Then compare the tables side by side. Key questions to answer:

1. **Is 64-bit scatter faster than split-32 on NVIDIA?** (Expected: yes, since NVIDIA has native 64-bit ALUs)
2. **What fraction of full decode is bitstream reading?** (Tells us if the serial bit-reading bottleneck is architecture-specific)
3. **What is the peak memory bandwidth vs actual decode throughput?** (Shows how compute-bound vs memory-bound the kernel is)
4. **How does the transform cost compare?** (Should be similar relative fraction on both architectures)

## How it works

1. Generates smooth sinusoidal test data on the host (`sin(0.1x) * cos(0.15y) * sin(0.12z)`)
2. Compresses it on-GPU using the exact CUDA ZFP encode pipeline (inlined from the original `encode.cuh`/`encode3.cuh`)
3. Runs each diagnostic kernel with CUDA event timing
4. Reports the **minimum (best)** time across all repeats — matching the Metal convention for warm-kernel comparison
5. Outputs both a formatted terminal table and a copy-paste-ready markdown table
