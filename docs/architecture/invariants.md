# Arc2Avatar — Immutable System Invariants
> Derived from Directive 0.4. Every invariant listed here must hold at every point
> in execution. Violations produce plausible-looking-but-wrong output, not crashes.
> CI must grep for the exact assertion strings named in "Enforced by" and fail
> the build if any assertion referenced here is missing from the code it belongs to.

| # | Invariant | Enforced by (function/assertion) |
|---|---|---|
| I1 | FLAME topology (vertex count, face count) never changes after Directive 8 loads it | `assert flame.V.shape[0] == FLAME_CANONICAL_VERT_COUNT` at the top of every function in `ANIM` and `GAUSS` |
| I2 | Every Gaussian has exactly one `vertex_id`; `len(vertex_id) == len(gaussian_means)` always | asserted after every optimizer step (Directive 11) |
| I3 | The ID embedding dimensionality is fixed at 512 and never recomputed mid-run from a different image | `assert id_embedding.shape == (512,)` at every function boundary that consumes it |
| I4 | A checkpoint's schema version (Directive 42) must match the loader's expected version, or loading must fail loudly, never silently coerce | version-check gate in the checkpoint loader |
| I5 | `PRIOR`'s fine-tuned weights (Directive 16) are read-only for the rest of the system — no subsystem may call `.backward()` through them after freezing | `requires_grad_(False)` set once, verified via a unit test that attempts and expects-to-fail a gradient computation |
| I6 | Random seed is set exactly once per run, at the earliest possible point in `ORCH` startup, and never reset mid-run | single call-site check enforced by a lint rule / grep in CI (Directive 66) |

---

## Invariant Enforcement Details

### I1 — FLAME Topology Stability
- **File locations**: `src/animation/`, `src/models/` (GAUSS)
- **Constant**: `FLAME_CANONICAL_VERT_COUNT` defined in `src/config/schema.py` (default 5023 for FLAME 2023)
- **Failure mode**: Gaussians paired with wrong vertices → visibly broken deformation with no error message

### I2 — Gaussian-Vertex Correspondence
- **File locations**: Every optimizer step in `src/optim/`, every animation function in `src/animation/`
- **Failure mode**: Silent splat deformation after `vertex_id` tensor is accidentally subsetted or renumbered

### I3 — Identity Embedding Dimensionality
- **File locations**: `src/contracts/schemas.py` (IdentityEmbedding.__post_init__), every SDS/validation entry point
- **Failure mode**: Dimensionality mismatch crashes deep inside diffusion model with opaque error

### I4 — Checkpoint Schema Versioning
- **File locations**: `src/state/checkpoint_manager.py` (validate_checkpoint)
- **Failure mode**: Loading a checkpoint from a different schema version silently corrupts tensor shapes

### I5 — Frozen Prior Weights
- **File locations**: `checkpoints/arc2face_finetuned/final.pt` loader
- **Negative test**: `tests/unit/test_prior_api.py` — attempts gradient computation, expects RuntimeError

### I6 — Single Seed Call
- **File locations**: `src/utils/seed.py` only
- **CI lint**: grep for `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` outside seed.py

> **Note**: This file is designed to grow. If any additional invariant is discovered
> while building Modules C–H (e.g., "SH coefficient tensor's K dimension never changes"),
> add it as I7, I8, etc. with its enforcement mechanism.
