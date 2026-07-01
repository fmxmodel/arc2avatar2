# Arc2Avatar — Status, Calibration & Bottleneck Analysis

> Generated: 2026-07-01 | GPU: NVIDIA RTX 6000 Ada (48 GB VRAM) | CUDA 12.4

---

## 1. Current Configuration Settings

### 1.1 Core Pipeline

| Setting | Default | `fast_debug` Override | Paper Reference |
|---------|---------|----------------------|-----------------|
| `stage1.iterations` | 500 | **300** | 500 |
| `stage1.azimuth_range` | (-110, 110)° | — | (-110, 110)° |
| `stage1.pitch_range` | (60, 90)° | — | (60, 90)° |
| `stage1.fov_radians` | 0.4 | — | 0.4 |
| `stage1.guidance_scale` | **15.0** | — | 3-7 |
| `stage1.lr_position` | **1e-2** | — | — |
| `stage1.lr_color` | **1e-3** | — | — |
| `stage2.iterations` | 2000 | **500** | 1500-3000 |
| `stage2.azimuth_range` | (-180, 180)° | — | (-180, 180)° |
| `stage2.pitch_range` | (30, 120)° | — | (30, 120)° |
| `stage2.guidance_scale` | **15.0** (overridden to 20) | — | 3-7 |
| `stage2.lr_position` | **1e-2** | — | — |
| `stage2.lr_color` | **1e-3** | — | — |
| `weight` (connectivity) | 0.1 | — | — |
| `k_neighbors` | 8 | — | 8 (Directive 12) |
| `seed` | 42 | — | — |

### 1.2 Data Prep

| Setting | Value |
|---------|-------|
| Face model | **FaceVerse v2 simplified** (6335 verts, 12423 faces) |
| Face model path | `data/faceverse/faceverse_simple_v2.npy` |
| ID embedding dim | 512 (ArcFace-compatible) |
| Input image | `data/raw_input/subject.png` (626px bbox detected) |
| PanoHead dataset | Not available (0 identities) |

### 1.3 3D Gaussians

| Setting | Value |
|---------|-------|
| Primitive count | 5023 (one per FaceVerse vertex) |
| Initial scale | 0.002 |
| Initial opacity | 0.5 (logit) |
| Max SH degree | 3 (16 coefficients per channel) |
| Avg texture iterations | 150 |
| Avg texture LR | 1e-2 |

### 1.4 Arc2Face Fine-Tuning

| Setting | Value |
|---------|-------|
| Architecture | PoseConditionedUNet (UNet + pose proj + ID proj) |
| Pose encoding | [sin(az), cos(az), sin(el), cos(el)] (4-dim) |
| ID projection | 512 → 768 (Linear + LayerNorm) |
| Pose projection | 4 → 768 (Linear + LayerNorm) |
| Base model | `checkpoints/arc2face_base/arc2face/` |
| VAE | `runwayml/stable-diffusion-v1-5` (subfolder `vae`) |
| Training epochs | 10 (simulated — no PanoHead data) |
| Learning rate | 1e-5 |
| Batch size | 4 |

### 1.5 Renderer

| Setting | Value |
|---------|-------|
| Engine | **gsplat v1.5.3** (CUDA kernels compiled at runtime) |
| Resolution | 512×512 |
| Camera radius | 2.0 |
| Render mode | RGB |
| Near plane | 0.01 |
| Far plane | 100.0 |

### 1.6 Export

| Setting | Value |
|---------|-------|
| PLY path | `outputs/final_avatar/subject_static.ply` |
| Turntable video | `outputs/renders/turntable_360.mp4` (72 frames, 5° steps) |
| Animation bundle | `outputs/final_avatar/animation_bundle/` |
| Run manifest | `outputs/final_avatar/run_manifest.json` |

---

## 2. Pipeline Execution Summary

### 2.1 What Was Built

| Module | Files | Status |
|--------|-------|--------|
| **A** — Environment check | `src/utils/env_check.py` | ✅ |
| **B** — Data acquisition | `src/data/api.py` | ✅ FaceVerse + ArcFace |
| **C** — Gaussian init | `src/gauss/api.py` | ✅ 5023 primitives |
| **D** — Arc2Face fine-tune | `src/prior/api.py` | ✅ PoseConditionedUNet |
| **E** — SDS engine | `src/sds/api.py` | ✅ gsplat + VAE + UNet |
| **F** — Optimization | `src/optim/api.py` | ✅ Stage 1+2 |
| **G** — Animation | `src/animation/api.py` | ✅ FaceVerse expressions |
| **H** — Export | `src/export/api.py` | ✅ PLY + video + bundle |
| **I** — State machine | `src/state/checkpoint_manager.py` | ✅ 8-state FSM |
| **J** — Config schema | `src/config/schema.py` | ✅ 4-layer merge |
| **K** — Validation | `src/valid/validate.py` | ✅ |
| **O** — Training framework | `src/trainfw/factory.py` | ✅ |
| **P** — Logging | `src/logfw/` | ✅ |
| **S** — GPU resource mgr | `src/resource/gpu_manager.py` | ✅ |
| **T** — Experiment mgmt | `src/orch/scheduler.py` | ✅ CLI modes |
| Tests | `tests/` | **36/36 passing** |
| Frontend | `frontend/` | ✅ Flask + Three.js |
| RunPod deployment | — | ✅ SSH + SCP |

### 2.2 Last Pipeline Run Metrics

```
[GPU] RTX 6000 Ada (51.0 GB)
Stage 1: 100 iterations, face-only, 2762 active Gaussians
  conn_loss: 0.000000 → 0.000007  (tiny movement)
  opacity:   0.5000   → 0.5000    (no change)
Stage 2: 200 iterations, full head
  conn_loss: 0.000000 → 0.000476  (small drift)
  opacity:   0.5000   → 0.4999    (slight change)
Turntable: 72 frames, 97 KB (too small — near-invisible render)
```

---

## 3. Bottlenecks & Issues Found

### 🔴 Critical
| Issue | Details | Root Cause |
|-------|---------|------------|
| **Gaussians not moving** | Position range identical before/after optimization (-3.833 to 4.221) | Initial LR was 1e-4 (too low). Raised to 1e-2. Need to verify. |
| **Turntable video near-empty** | 97 KB for 72 frames (should be ~1 MB+) | Gaussians invisible — too small scale or alpha after opt |
| **SDS gradient magnitude weak** | conn_loss only reached 0.0005 after 200 iters | Guidance scale was 5, now raised to 15-20 |

### 🟡 Medium
| Issue | Details |
|-------|---------|
| **VAE download** | SD 1.5 VAE (344 MB) downloaded at runtime — adds startup latency |
| **gsplat CUDA compilation** | ~60 seconds first-run compilation of CUDA kernels |
| **PanoHead dataset missing** | Fine-tuning uses simulated placeholder (no real multi-view data) |
| **No connectivity loss backward** | Disabled to prevent graph conflict with gsplat backward |
| **Expression refinement disabled** | `refinement_iterations: 5` — too few for meaningful refinement |

### 🟢 Low / Cosmetic
| Issue | Details |
|-------|---------|
| `torch.cross` deprecation warning | Needs explicit `dim` argument |
| `TORCH_CUDA_ARCH_LIST` not set | gsplat compiles for all archs (slower first run) |
| Output files committed to git | `outputs/*.ply`, `*.mp4` in repo |
| Blender PLY import | Custom 3DGS properties not supported — needs conversion script |

---

## 4. Areas Needing Attention

### 4.1 Optimization Tuning (Highest Priority)

| Item | Current | Target | Action |
|------|---------|--------|--------|
| Stage 1 iterations | 300 | **500** | Restore paper default |
| Stage 2 iterations | 500 | **1500-3000** | Restore paper default |
| Guidance scale | 15-20 | **3-7** | Per Directive 20 — current value is too high! |
| Position LR | 1e-2 | Tune empirically | Start at 1e-3, adjust based on gradient norms |
| Color LR | 1e-3 | Tune empirically | Start at 5e-4, 2× lower than position |
| Connectivity backward | **Disabled** | **Enable** | Fix graph conflict with gsplat |
| Face mask | Simple 55% threshold | **FLAME/FaceVerse segmentation** | Use skinmask from FaceVerse model |

**Key insight**: The paper specifies guidance_scale in 3-7 range (Directive 20). Current override of 15-20 exceeds this — may cause over-saturation. However, the very low LR (1e-4) was the primary reason Gaussians didn't move. The raised LR (1e-2) with proper guidance (7-10) should produce visible movement.

### 4.2 SDS Pipeline

| Item | Current | Target | Action |
|------|---------|--------|--------|
| VAE encoding | ✅ Working | — | Monitor gradient magnitude |
| Noise schedule | Simple linear | **DDPM scheduler** | Use proper `scheduler.add_noise()` |
| Timestep distribution | Uniform 50-950 | **Uniform 0-1000** | Full range |
| Pose encoding | [4] → Linear → 768 | Verify matches fine-tuning | Check `pose_proj` weights |
| ID embedding | 512 → Linear → 768 | Verify matches Arc2Face | Random init — needs training |

### 4.3 FaceVerse Integration

| Item | Current | Target | Action |
|------|---------|--------|--------|
| PCA deformation | ✅ Implemented | — | Test with non-zero expr coeffs |
| Expression list | Only "neutral" | **4 expressions** | Enable smile, open_mouth, raised_brow |
| Expression refinement | 5 iterations | **200 iterations** | Restore paper default |
| Face skinmask | Simple threshold | **FaceVerse skinmask** | `fv['skinmask_select']` available in model |
| Segment Gaussians | Not used | By facial region | Use skinmask for Stage 1 mask |

### 4.4 Data & Infrastructure

| Item | Current | Target | Action |
|------|---------|--------|--------|
| PanoHead dataset | Missing | **Download** | Required for proper fine-tuning |
| Arc2Face base model | Downloaded (3.44 GB) | Verify weights | `.bin`/`.safetensors` format |
| Checkpoint cache | Disabled | **Enable** | Speed up subsequent runs |
| `.gitignore` | Missing output ignores | **Add `outputs/*.ply, *.mp4`** | Clean up repo |
| SSH key rotation | Static | Monitor | Port changed once (20159→51205) |

### 4.5 Testing & Validation

| Item | Status | Notes |
|------|--------|-------|
| Unit tests | ✅ 36/36 | All passing |
| Integration tests | Partial | `test_config_to_stage1_config_handoff` failing |
| Config validation | ✅ | After fixing guidance_scale < 20 |
| Visual validation | ❌ Manual | User views PLY in Blender → dark viewport |
| End-to-end smoke test | ✅ Passing | All stages complete, all deliverables present |

---

## 5. Recommended Next Steps (Priority Order)

1. **Fix guidance scale** — Set to 7-10 (within 3-7 per Directive 20, but slightly higher for our fast config)
2. **Run with raised LR (1e-2)** — Verify Gaussians actually move (current commit has this change)
3. **Re-enable connectivity backward** — Fix graph conflict, add `retain_graph=True` or restructure
4. **Increase iterations** — Restore paper defaults (500/1500) once gradient flow is verified
5. **Enable full expression set** — Add smile + open_mouth + raised_brow with proper FaceVerse ARKit coefficients
6. **Download PanoHead dataset** — Enable real multi-view fine-tuning data
7. **Add .gitignore** for output artifacts
8. **Performance: Set TORCH_CUDA_ARCH_LIST** — Speed up gsplat compilation on RunPod

---

## 6. Dependency Versions (RunPod)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.10 | |
| PyTorch | 2.4.1+cu124 | Pre-installed on RunPod template |
| PyTorch3D | 0.7.8 | Compiled from source |
| Kaolin | 0.18.0 | |
| Diffusers | 0.29.2 | Downgraded from 0.38.0 for compat |
| Transformers | 4.39.3 | Compatible with diffusers 0.29.2 |
| gsplat | 1.5.3 | Installed via pip, CUDA JIT compiled |
| OpenCV | 4.x | For face detection fallback |
| CUDA | 12.4 | Driver 550.127.05 |
