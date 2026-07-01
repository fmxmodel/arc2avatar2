# Arc2Avatar — Sources of Nondeterminism
> Directive 85: Every legitimate source of nondeterminism must be explicitly
> enumerated, so the reproducibility audit's tolerance can be justified.

## Known Sources

### 1. CUDA Convolution Ordering
Certain CUDA kernels (especially grouped convolutions and attention layers in
the diffusion model) sum floating-point values in an order that depends on
kernel launch configuration and hardware generation. This produces bitwise-
different but functionally identical results across runs.

**Impact on reproducibility:** Maximum per-parameter difference < 1e-6.

### 2. Atomic Operations in 3DGS Rasterizer
The 3D Gaussian splatting rasterizer uses atomic operations for alpha blending
when multiple Gaussians project to the same pixel. The order of atomic
additions is nondeterministic within a warp.

**Impact on reproducibility:** Pixel-level differences in high-density regions.
Maximum per-parameter difference < 1e-5.

### 3. cuDNN Autotuning
cuDNN's autotuner may select different algorithm implementations across runs
or hardware configurations.

**Mitigation:** `torch.backends.cudnn.deterministic = True` is set for
reproducibility runs, but this may reduce throughput.

### Documented Tolerance for Reproducibility Audit (Directive 85)
Maximum acceptable per-parameter difference between two identical-config runs:
**1e-4** (absolute). This accounts for all known nondeterminism sources above.

> **Note:** If a new source of nondeterminism is discovered during development,
> add it to this list with its estimated impact. Do NOT silently widen the
> audit tolerance without documenting why.
