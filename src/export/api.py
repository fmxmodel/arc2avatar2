"""EXPORT subsystem — Export & Packaging (Module H)
Directives 28-31, 86.

Full implementation: .ply export, turntable video, animation bundle,
reproducibility manifest.
"""

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from src.contracts.schemas import (
    GaussianState,
    FaceVerseMesh,
    ExpressionState,
    RunManifest,
    save_run_manifest,
    save_versioned,
    compute_file_hash,
)
from src.errors.hierarchy import ExportError
from src.resource.gpu_manager import managed_stage
from src.sds.api import render, sample_camera


def _gaussian_to_ply(gaussian_state: GaussianState) -> str:
    """Build PLY string from GaussianState.

    Standard 3DGS-compatible format: position, scale+rotation, opacity, SH.
    """
    device = gaussian_state.means.device
    N = gaussian_state.means.shape[0]

    means = gaussian_state.means.cpu().numpy()
    scales = torch.exp(gaussian_state.scales).cpu().numpy()  # exp() activation
    rotations = F.normalize(gaussian_state.rotations, dim=1).cpu().numpy()
    opacities = torch.sigmoid(gaussian_state.opacities).cpu().numpy()
    sh_dc = gaussian_state.sh_coeffs[:, :, 0].cpu().numpy()  # [N, 3]

    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
"""
    lines = []
    for i in range(N):
        lines.append(
            f"{means[i,0]:.6f} {means[i,1]:.6f} {means[i,2]:.6f} "
            f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} "
            f"{rotations[i,0]:.6f} {rotations[i,1]:.6f} {rotations[i,2]:.6f} {rotations[i,3]:.6f} "
            f"{opacities[i,0]:.6f} "
            f"{sh_dc[i,0]:.6f} {sh_dc[i,1]:.6f} {sh_dc[i,2]:.6f}"
        )

    return header + "\n".join(lines)


def export_static_ply(gaussian_state: GaussianState, output_path: str) -> None:
    """Export static avatar as 3DGS-compatible .ply (Directive 28).

    Inputs:    GaussianState (stage2_full_head), output path.
    Outputs:   None (writes .ply file).
    Exceptions: raises ExportError on write failure.
    Side effects: writes subject_static.ply.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ply_str = _gaussian_to_ply(gaussian_state)
        with open(output_path, "w") as f:
            f.write(ply_str)
        print(f"[EXPORT] Saved static PLY: {output_path} ({gaussian_state.means.shape[0]} Gaussians)")
    except Exception as e:
        raise ExportError(
            what_failed="PLY export failed",
            why=str(e),
            how_to_fix="Check disk space and GaussianState validity",
        )


def render_turntable(gaussian_state: GaussianState, config) -> None:
    """Render turntable validation video (Directive 29).

    5° azimuth increments, full 360°, fixed pitch at neutral viewing angle.

    Inputs:    GaussianState, ExportConfig.
    Outputs:   None (writes video file).
    Exceptions: raises RenderingError on render failure.
    Side effects: writes turntable_360.mp4.
    """
    try:
        from src.resource.gpu_manager import get_device
        device = get_device()

        # Move Gaussian state to GPU device for rendering
        gs_device = gaussian_state
        for field in ['means', 'scales', 'rotations', 'opacities', 'sh_coeffs']:
            t = getattr(gs_device, field, None)
            if t is not None and t.device != device:
                setattr(gs_device, field, t.to(device))

        output_path = getattr(config, "turntable_path", "outputs/renders/turntable_360.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        angle_step = getattr(config, "turntable_angle_step_deg", 5.0)
        pitch = 60.0  # Neutral viewing angle

        frames = []
        for az in np.arange(0, 360, angle_step):
            cam = sample_camera((az, az), (pitch, pitch), 0.4, 2.0)
            result = render(gs_device, cam)

            img = result.image.detach().cpu()
            # Convert to HWC uint8 for video
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            frames.append(img_bgr)

        # Write video
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

        print(f"[EXPORT] Turntable video: {output_path} ({len(frames)} frames)")

    except Exception as e:
        raise ExportError(
            what_failed="Turntable render failed",
            why=str(e),
            how_to_fix="Check render pipeline and disk space",
        )


@managed_stage
def package_animation_bundle(
    gaussian_state: GaussianState,
    expression_states: List[ExpressionState],
    faceverse_mesh: FaceVerseMesh,
    output_dir: str,
) -> None:
    """Package animation-ready bundle (Directive 30).

    Includes: static Gaussian state, vertex_id correspondence, FaceVerse basis,
    refined-expression patches, and machine-readable manifest.

    Inputs:    GaussianState, list of ExpressionState, FaceVerseMesh, output directory.
    Outputs:   None (writes bundle files).
    Exceptions: raises ExportError on packaging failure.
    Side effects: writes animation_bundle/ with manifest.json.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save static state
        static_path = os.path.join(output_dir, "static_state.pt")
        save_versioned(gaussian_state, static_path)

        # Save FaceVerse basis
        fv_path = os.path.join(output_dir, "faceverse_basis.pt")
        save_versioned(faceverse_mesh, fv_path)

        # Build manifest
        refinement_map = {}
        for expr in expression_states:
            if expr.requires_refinement and expr.refined_patch is not None:
                patch_path = f"expr_{expr.name}_refined.pt"
                save_versioned(expr, os.path.join(output_dir, patch_path))
                refinement_map[expr.name] = {
                    "has_refinement": True,
                    "path": patch_path,
                }
            else:
                refinement_map[expr.name] = {
                    "has_refinement": False,
                    "path": None,
                }

        manifest = {
            "schema_version": 1,
            "static_gaussian_state": "static_state.pt",
            "vertex_id_correspondence": "static_state.pt (vertex_id field)",
            "faceverse_basis": "faceverse_basis.pt",
            "expressions": refinement_map,
            "total_gaussians": gaussian_state.means.shape[0],
        }

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[EXPORT] Animation bundle: {output_dir} ({len(expression_states)} expressions)")

    except Exception as e:
        raise ExportError(
            what_failed="Animation bundle packaging failed",
            why=str(e),
            how_to_fix="Check output directory permissions and space",
        )


def write_run_manifest(
    config,
    checkpoint_hashes: Dict[str, str],
    quality_results: Dict[str, bool],
    output_path: str,
) -> None:
    """Write reproducibility manifest (Directive 31).

    Write-once: if file already exists, raises ExportError (Directive 41).

    Inputs:    resolved PipelineConfig, checkpoint hash dict, quality gate results, output path.
    Outputs:   None (writes JSON file — write-once, never modified).
    Exceptions: raises ExportError if file already exists.
    Side effects: writes run_manifest.json.
    """
    if os.path.exists(output_path):
        raise ExportError(
            what_failed="Run manifest already exists",
            why=f"File exists: {output_path}. Manifest is write-once per Directive 41.",
            how_to_fix="Move or delete existing manifest, or re-run the pipeline",
        )

    import hashlib
    import git

    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit = repo.head.commit.hexsha
    except Exception:
        git_commit = "unknown"

    manifest = RunManifest(
        run_id=config.run_id or "dev-run",
        config_snapshot={
            "stage1_iterations": config.stage1.iterations,
            "stage2_iterations": config.stage2.iterations,
            "stage1_azimuth": config.stage1.azimuth_range_deg,
            "stage1_pitch": config.stage1.pitch_range_deg,
            "stage2_azimuth": config.stage2.azimuth_range_deg,
            "stage2_pitch": config.stage2.pitch_range_deg,
            "guidance_scale": config.stage1.guidance_scale,
            "seed": config.seed,
        },
        checkpoint_hashes=checkpoint_hashes,
        input_image_hash=compute_file_hash(config.data_prep.input_image_path),
        stage1_iterations_actual=config.stage1.iterations,
        stage2_iterations_actual=config.stage2.iterations,
        quality_gate_results=quality_results,
    )

    save_run_manifest(manifest, output_path)
    print(f"[EXPORT] Run manifest: {output_path}")


# Import for PLY export
import torch.nn.functional as F
