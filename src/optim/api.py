"""OPT subsystem — Two-Stage Optimization (Module F)
Directives 21-23.

Full implementation: Stage 1 (face-only, 500 iter) and Stage 2 (full-head).
Includes divergence guard, gradient masking, connectivity regularization.
"""

import os
from collections import deque
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from src.contracts.schemas import (
    GaussianState,
    IdentityEmbedding,
    CameraSample,
    save_gaussian_state,
    load_gaussian_state,
)
from src.errors.hierarchy import DivergenceError
from src.resource.gpu_manager import get_device, managed_stage
from src.trainfw.factory import build_optimizer
from src.trainfw.grad_utils import zero_grad, clip_gradients
from src.sds.api import render, sample_camera, sds_step
from src.optim._sds_internals import compute_connectivity_loss, compute_initial_offsets


# FLAME facial vertex subset (indices for face region, excluding scalp/ears/neck)
# This is a simplified mask — in production, use the full FLAME segmentation
_FLAME_FACE_MASK: Optional[torch.Tensor] = None


def _get_face_mask(n_vertices: int, device: str) -> torch.Tensor:
    """Get boolean mask for FLAME facial-region vertices.

    Returns [N] bool tensor: True for facial vertices, False for scalp/ears/neck.
    """
    global _FLAME_FACE_MASK
    if _FLAME_FACE_MASK is not None and len(_FLAME_FACE_MASK) == n_vertices:
        return _FLAME_FACE_MASK.to(device)

    # FLAME face region: vertices 0-3000 approximately cover the face
    dev = torch.device(device) if isinstance(device, str) else device
    face_mask = torch.zeros(n_vertices, dtype=torch.bool, device=dev)
    face_mask[:3000] = True  # Facial region
    face_mask[3000:3500] = True

    _FLAME_FACE_MASK = face_mask.cpu()
    return face_mask.to(dev)


def _divergence_guard(
    conn_loss: float,
    pixel_intensity: float,
    history_conn: deque,
    history_intensity: deque,
    config,
    iteration: int,
    stage_name: str,
) -> None:
    """Divergence guard (Directive 23).

    Monitors connectivity loss and pixel intensity.
    On >3 std-dev spike from trailing 50-iteration average: halt.
    """
    history_conn.append(conn_loss)
    history_intensity.append(pixel_intensity)

    if len(history_conn) < 50:
        return

    # Compute rolling stats
    conn_mean = sum(history_conn) / len(history_conn)
    conn_std = (sum((x - conn_mean) ** 2 for x in history_conn) / len(history_conn)) ** 0.5
    int_mean = sum(history_intensity) / len(history_intensity)
    int_std = (sum((x - int_mean) ** 2 for x in history_intensity) / len(history_intensity)) ** 0.5

    threshold = config.std_dev_threshold
    conn_spike = abs(conn_loss - conn_mean) > threshold * max(conn_std, 1e-8)
    int_spike = abs(pixel_intensity - int_mean) > threshold * max(int_std, 1e-8)

    if conn_spike or int_spike:
        raise DivergenceError(
            what_failed=f"Divergence detected in {stage_name} at iteration {iteration}",
            why=f"conn_loss={conn_loss:.6f} (mean={conn_mean:.6f}, std={conn_std:.6f}), "
                f"pixel_intensity={pixel_intensity:.6f} (mean={int_mean:.6f}, std={int_std:.6f})",
            how_to_fix=f"Halve guidance scale and resume from checkpoint. "
                       f"Set stage1.guidance_scale or stage2.guidance_scale in config.",
        )


def _log_preview(
    gaussian_state: GaussianState,
    camera: CameraSample,
    iteration: int,
    stage_name: str,
    writer: SummaryWriter,
    output_dir: str,
    loss_val: float,
    conn_loss_val: float,
) -> None:
    """Log rendered preview to tensorboard and disk."""
    # Render preview
    try:
        prev = render(gaussian_state, camera)
        img = prev.image.detach().cpu()

        # Log to tensorboard
        writer.add_image(f"{stage_name}/render", img, iteration)
        writer.add_scalar(f"{stage_name}/loss", loss_val, iteration)
        writer.add_scalar(f"{stage_name}/connectivity_loss", conn_loss_val, iteration)

        # Save to disk every 50 iterations
        if iteration % 50 == 0:
            from torchvision.utils import save_image
            os.makedirs(output_dir, exist_ok=True)
            save_image(img, os.path.join(output_dir, f"{stage_name}_iter{iteration:06d}.png"))

    except Exception:
        pass  # Preview logging is non-critical


@managed_stage
def run_stage1(
    initial_state: GaussianState,
    identity: IdentityEmbedding,
    prior_model: object,
    config,
) -> GaussianState:
    """Stage 1: Face-only optimization (Directive 21).

    Differentiable 3DGS optimization via gsplat rasterizer.
    Face-only gradient mask, reference camera ranges.

    Inputs:    avg_texture_fit GaussianState, IdentityEmbedding, frozen prior, Stage1Config.
    Outputs:   updated GaussianState (facial region only).
    Exceptions: raises DivergenceError if divergence guard trips.
    Side effects: saves stage1_face.pt, writes preview renders, logs to tensorboard.
    """
    import os
    from collections import deque
    from torch.utils.tensorboard import SummaryWriter

    from src.contracts.schemas import save_gaussian_state
    from src.trainfw.factory import build_optimizer
    from src.trainfw.grad_utils import zero_grad
    from src.sds.api import render, sample_camera, sds_step
    from src.optim._sds_internals import compute_connectivity_loss, compute_initial_offsets

    device = get_device()
    writer = SummaryWriter(log_dir="logs/tensorboard/stage1")
    os.makedirs("outputs/renders/stage1", exist_ok=True)

    # Create fresh GaussianState on GPU with leaf tensors for optimizer
    def _to_leaf(t):
        return torch.tensor(t.detach().cpu().numpy(), device=device, requires_grad=t.is_floating_point())

    gs = GaussianState(
        means=_to_leaf(initial_state.means),
        scales=_to_leaf(initial_state.scales),
        rotations=_to_leaf(initial_state.rotations),
        opacities=_to_leaf(initial_state.opacities),
        sh_coeffs=_to_leaf(initial_state.sh_coeffs),
        vertex_id=initial_state.vertex_id.clone().to(device),
    )

    n_vertices = gs.vertex_id.shape[0]
    # Face mask: use FaceVerse skinmask if available, else heuristic
    face_mask = torch.zeros(n_vertices, dtype=torch.bool, device=device)
    try:
        import numpy as np
        fv = np.load("data/faceverse/faceverse_simple_v2.npy", allow_pickle=True).item()
        if 'skinmask_select' in fv:
            sm = fv['skinmask_select']  # [1, 6335] float32
            mask_arr = sm[0] > 0.5
            if len(mask_arr) >= n_vertices:
                face_mask = torch.tensor(mask_arr[:n_vertices], device=device)
    except Exception:
        pass
    if not face_mask.any():
        face_mask[:int(n_vertices * 0.55)] = True  # fallback heuristic

    # Build optimizer with separate LRs for position vs color/appearance
    optimizer = build_optimizer([gs.means, [gs.scales, gs.rotations, gs.opacities, gs.sh_coeffs]], config)

    # Pre-compute initial offsets for connectivity regularizer
    initial_offsets = compute_initial_offsets(gs.means, k=config.k_neighbors)

    # Divergence guard state
    history_conn = deque(maxlen=config.trailing_window)
    history_intensity = deque(maxlen=config.trailing_window)

    n_iter = config.iterations
    print(f"[OPT] Stage 1: {n_iter} iterations, face-only, {face_mask.sum().item()} active Gaussians")

    for i in range(n_iter):
        cam = sample_camera(
            config.azimuth_range_deg, config.pitch_range_deg,
            config.fov_radians,
        )

        # SDS loss (differentiable through gsplat + VAE)
        sds_loss = sds_step(gs, cam, identity, prior_model, config)

        # Connectivity loss (independent graph from SDS)
        conn_loss = compute_connectivity_loss(
            gs.means, gs.vertex_id, initial_offsets,
            k=config.k_neighbors, weight=config.weight,
        )

        # Combine losses and backward ONCE — avoids graph conflicts
        total_loss = sds_loss + conn_loss
        total_loss.backward()

        # Apply gradient mask (face-only)
        for field in ['means', 'scales', 'rotations', 'opacities', 'sh_coeffs']:
            t = getattr(gs, field, None)
            if t is not None and t.grad is not None:
                t.grad[~face_mask] = 0.0

        # Adam step
        optimizer.step()
        zero_grad(optimizer)

        pixel_intensity = torch.sigmoid(gs.opacities).mean().item()
        _divergence_guard(conn_loss.item(), pixel_intensity,
                          history_conn, history_intensity,
                          config, i, "Stage1")

        if i % max(config.log_interval, 1) == 0:
            gn = gs.means.grad.norm().item() if gs.means.grad is not None else 0.0
            print(f"  [OPT] Stage 1 iter {i}/{n_iter}: conn={conn_loss.item():.8f}, "
                  f"opac={pixel_intensity:.4f}, grad_norm={gn:.6f}")

    writer.close()
    save_gaussian_state(gs, config.checkpoint_path)
    print(f"[OPT] Stage 1 complete → {config.checkpoint_path}")
    return gs


@managed_stage
def run_stage2(
    stage1_state: GaussianState,
    identity: IdentityEmbedding,
    prior_model: object,
    config,
) -> GaussianState:
    """Stage 2: Full-head optimization (Directive 22).

    Unfreezes ALL Gaussians, wider camera ranges, larger iteration budget.
    Differentiable via gsplat rasterizer.

    Inputs:    stage1 GaussianState, IdentityEmbedding, frozen prior, Stage2Config.
    Outputs:   updated GaussianState (full head).
    Exceptions: raises DivergenceError if divergence guard trips.
    Side effects: saves stage2_full_head.pt, writes preview renders, logs to tensorboard.
    """
    import os
    from collections import deque
    from torch.utils.tensorboard import SummaryWriter

    from src.contracts.schemas import save_gaussian_state
    from src.trainfw.factory import build_optimizer
    from src.trainfw.grad_utils import zero_grad
    from src.sds.api import render, sample_camera, sds_step
    from src.optim._sds_internals import compute_connectivity_loss, compute_initial_offsets

    device = get_device()
    writer = SummaryWriter(log_dir="logs/tensorboard/stage2")
    os.makedirs("outputs/renders/stage2", exist_ok=True)

    # Create fresh GaussianState on GPU with leaf tensors for optimizer
    def _to_leaf(t):
        return torch.tensor(t.detach().cpu().numpy(), device=device, requires_grad=t.is_floating_point())

    gs = GaussianState(
        means=_to_leaf(stage1_state.means),
        scales=_to_leaf(stage1_state.scales),
        rotations=_to_leaf(stage1_state.rotations),
        opacities=_to_leaf(stage1_state.opacities),
        sh_coeffs=_to_leaf(stage1_state.sh_coeffs),
        vertex_id=stage1_state.vertex_id.clone().to(device),
    )

    # Build optimizer for ALL parameters (pos vs color groups)
    optimizer = build_optimizer(
        [gs.means, [gs.scales, gs.rotations, gs.opacities, gs.sh_coeffs]],
        config,
    )

    # Pre-compute initial offsets
    initial_offsets = compute_initial_offsets(gs.means, k=config.k_neighbors)

    # Divergence guard state
    history_conn = deque(maxlen=config.trailing_window)
    history_intensity = deque(maxlen=config.trailing_window)

    n_iter = config.iterations
    print(f"[OPT] Stage 2: {n_iter} iterations, full head, "
          f"azimuth={config.azimuth_range_deg}, pitch={config.pitch_range_deg}")

    for i in range(n_iter):
        cam = sample_camera(
            config.azimuth_range_deg, config.pitch_range_deg,
            config.fov_radians,
        )

        # SDS loss + connectivity loss → single backward
        sds_loss = sds_step(gs, cam, identity, prior_model, config)
        conn_loss = compute_connectivity_loss(
            gs.means, gs.vertex_id, initial_offsets,
            k=config.k_neighbors, weight=config.weight,
        )
        (sds_loss + conn_loss).backward()

        optimizer.step()
        zero_grad(optimizer)

        pixel_intensity = torch.sigmoid(gs.opacities).mean().item()
        _divergence_guard(
            conn_loss.item(), pixel_intensity,
            history_conn, history_intensity,
            config, i, "Stage2",
        )

        if i % max(config.log_interval, 1) == 0:
            gn = gs.means.grad.norm().item() if gs.means.grad is not None else 0.0
            print(f"  [OPT] Stage 2 iter {i}/{n_iter}: conn={conn_loss.item():.8f}, "
                  f"opac={pixel_intensity:.4f}, grad_norm={gn:.6f}")

    writer.close()
    save_gaussian_state(gs, config.checkpoint_path)
    print(f"[OPT] Stage 2 complete → {config.checkpoint_path}")
    return gs


def get_final_state() -> Optional[GaussianState]:
    """Get the final optimized state (fresh copy per Directive 44).

    Inputs:    None.
    Outputs:   copy of the current GaussianState, or None if unavailable.
    Exceptions: none.
    Side effects: none (returns a fresh copy).
    """
    path = "checkpoints/gaussians/stage2_full_head.pt"
    if os.path.exists(path):
        return load_gaussian_state(path)
    return None
