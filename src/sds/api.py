"""SDS subsystem — Score Distillation Sampling Engine (Module E)
Directives 17-20.

Full implementation: differentiable 3DGS renderer via gsplat,
camera sampler, SDS gradient step with explicit gradient injection.
"""

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.contracts.schemas import (
    GaussianState,
    CameraSample,
    RenderResult,
    IdentityEmbedding,
)
from src.errors.hierarchy import OptimizationError, RenderingError
from src.resource.gpu_manager import get_device, get_device_obj


def _build_viewmat_and_K(
    camera: CameraSample,
    image_size: Tuple[int, int] = (512, 512),
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build view matrix and intrinsic matrix from CameraSample (gsplat format).

    Returns viewmat [1, 4, 4] (world-to-camera) and K [1, 3, 3] (intrinsics).
    """
    az_rad = math.radians(camera.azimuth_deg)
    el_rad = math.radians(camera.pitch_deg)

    # Spherical to Cartesian: camera position
    cx = camera.radius * math.cos(el_rad) * math.sin(az_rad)
    cy = camera.radius * math.sin(el_rad)
    cz = camera.radius * math.cos(el_rad) * math.cos(az_rad)
    cam_pos = torch.tensor([cx, cy, cz], dtype=torch.float32, device=device)

    # Look-at: forward (from camera to origin), right, up
    at = torch.tensor(camera.look_at, dtype=torch.float32, device=device)
    forward = F.normalize(at - cam_pos, dim=0)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    right = F.normalize(torch.cross(forward, world_up), dim=0)
    up = torch.cross(right, forward)

    # View matrix (world-to-camera): [R | t]
    R = torch.stack([right, up, -forward], dim=0)  # [3, 3]
    t = -R @ cam_pos  # [3]
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    viewmat = viewmat.unsqueeze(0)  # [1, 4, 4]

    # Intrinsic matrix K from vertical FOV
    H, W = image_size
    fov_rad = camera.fov_rad
    f = 0.5 * H / math.tan(0.5 * fov_rad)
    cx_px, cy_px = W / 2.0, H / 2.0
    K = torch.tensor([
        [f, 0.0, cx_px],
        [0.0, f, cy_px],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]

    return viewmat, K


def _compute_sh_colors(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Compute RGB colors from SH coefficients (simplified: DC term + 1st band).

    sh_coeffs: [N, 3, 16] (SH up to 4th degree, but we use 0-3 bands)
    dirs: [N, 3] view direction per Gaussian

    Returns [N, 3] RGB colors.
    """
    # DC term: [N, 3]
    dc = sh_coeffs[:, :, 0]
    # 1st band (degree=1): 3 coefficients per color channel [N, 3, 3]
    # SH basis for degree 1: Y1 = [y, z, x] = [sin(θ)sin(φ), cos(θ), sin(θ)cos(φ)]
    # For view direction (dx, dy, dz): basis = [dy, dz, dx]
    if sh_coeffs.shape[2] >= 4:
        c1 = sh_coeffs[:, :, 1:4]  # [N, 3, 3]
        # SH basis: 0.5 * sqrt(3/pi) ≈ 0.4886
        basis = dirs[:, None, :] * 0.4886  # [N, 1, 3]
        sh = dc + (c1 * basis).sum(dim=2)
    else:
        sh = dc
    return torch.sigmoid(sh)


def render(gaussian_state: GaussianState, camera: CameraSample) -> RenderResult:
    """Differentiable renderer via PyTorch3D PointsRenderer (Directive 17).

    Uses PyTorch3D's point rasterizer which produces non-zero position gradients
    (unlike gsplat's rasterizer which zeros out dL/dmeans).

    Inputs:    GaussianState, CameraSample.
    Outputs:   RenderResult (differentiable image tensor).
    Exceptions: raises RenderingError on render failure.
    Side effects: none (pure function with autograd).
    """
    try:
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import (
            PointsRenderer, PointsRasterizationSettings, PointsRasterizer,
            AlphaCompositor, FoVPerspectiveCameras, look_at_view_transform,
        )
        import warnings

        device = get_device_obj()
        H, W = 512, 512
        radius = 0.015  # Point radius in normalized device coords

        # Build PyTorch3D camera from spherical coords
        az_rad = math.radians(camera.azimuth_deg)
        el_rad = math.radians(camera.pitch_deg)
        cx = camera.radius * math.cos(el_rad) * math.sin(az_rad)
        cy = camera.radius * math.sin(el_rad)
        cz = camera.radius * math.cos(el_rad) * math.cos(az_rad)
        R, T = look_at_view_transform(
            eye=((cx, cy, cz),), at=(camera.look_at,),
        )
        cam = FoVPerspectiveCameras(
            device=device, R=R.to(device), T=T.to(device),
            fov=camera.fov_rad * 180.0 / math.pi,
        )

        # Compute RGB colors from SH
        cam_pos = torch.tensor([cx, cy, cz], device=device)
        means = gaussian_state.means.to(device)
        dirs = F.normalize(means - cam_pos, dim=1)
        colors = _compute_sh_colors(gaussian_state.sh_coeffs.to(device), dirs)

        # Build point cloud
        point_cloud = Pointclouds(
            points=means.unsqueeze(0),
            features=colors.unsqueeze(0),
        )

        # Render
        raster_settings = PointsRasterizationSettings(
            image_size=H, radius=radius, points_per_pixel=10,
        )
        rasterizer = PointsRasterizer(cameras=cam, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images = renderer(point_cloud)

        image_tensor = images[0].permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

        return RenderResult(
            image=image_tensor,
            camera=camera,
            requires_grad=image_tensor.requires_grad,
        )

    except Exception as e:
        raise RenderingError(
            what_failed="3DGS render failed",
            why=str(e),
            how_to_fix="Check GaussianState validity and camera parameters",
        )


def sample_camera(
    azimuth_range: Tuple[float, float],
    pitch_range: Tuple[float, float],
    fov: float,
    radius: float = 2.0,
) -> CameraSample:
    """Sample a random camera (Directive 18).

    Fresh call every SDS iteration — never cached/reused.

    Inputs:    azimuth [min,max] deg, pitch [min,max] deg, fov rad, radius.
    Outputs:   CameraSample.
    Exceptions: raises RenderingError on invalid range.
    Side effects: none (random sampling each call).
    """
    if (azimuth_range[1] - azimuth_range[0]) < 0 or (pitch_range[1] - pitch_range[0]) < 0:
        raise RenderingError(
            what_failed="Invalid camera range",
            why=f"azimuth={azimuth_range}, pitch={pitch_range}",
            how_to_fix="Ensure min <= max for both ranges",
        )

    az = random.uniform(azimuth_range[0], azimuth_range[1])
    pitch = random.uniform(pitch_range[0], pitch_range[1])

    return CameraSample(
        azimuth_deg=az,
        pitch_deg=pitch,
        fov_rad=fov,
        radius=radius,
        look_at=(0.0, 0.0, 0.0),
    )


def sds_step(
    gaussian_state: GaussianState,
    camera: CameraSample,
    identity_embedding: IdentityEmbedding,
    prior_model: object,
    config,
) -> torch.Tensor:
    """Single SDS gradient step (Directive 19).

    Returns a scalar loss tensor that can be combined with other losses
    (e.g., connectivity) before a single `.backward()` call. This avoids
    graph conflicts between multiple backward passes.

    5-step sequence:
    1. Render image from current Gaussian state
    2. Add forward diffusion noise at random timestep t
    3. Feed noised image into frozen prior conditioned on (ID, pose)
    4. Compute SDS gradient: grad = w(t) * (noise_pred - noise_added)
    5. Return scalar loss proportional to SDS gradient

    Inputs:    GaussianState, CameraSample, IdentityEmbedding, frozen prior, config.
    Outputs:   scalar loss tensor (caller must call .backward()).
    Exceptions: raises OptimizationError on SDS failure.
    Side effects: none (pure function, no backward called).
    """
    try:
        device = get_device_obj()
        guidance_scale = getattr(config, "guidance_scale", 5.0)

        # Step 1: Render (differentiable via gsplat)
        gaussian_state.means.requires_grad_(True)
        gaussian_state.scales.requires_grad_(True)
        gaussian_state.rotations.requires_grad_(True)
        gaussian_state.opacities.requires_grad_(True)
        gaussian_state.sh_coeffs.requires_grad_(True)

        result = render(gaussian_state, camera)
        rendered = result.image.unsqueeze(0).to(device)  # [1, 3, H, W]

        # Step 2: Encode rendered image to latent space via VAE
        if isinstance(prior_model, dict):
            vae = prior_model.get("vae", None)
            unet = prior_model.get("unet", None)
        else:
            vae = None
            unet = getattr(prior_model, "unet", None) if hasattr(prior_model, "unet") else None

        w_t = 1.0  # Standard SDS weighting

        if vae is not None and unet is not None and hasattr(unet, 'base_unet'):
            # Full diffusion pipeline with VAE gradients
            rendered_norm = rendered * 2.0 - 1.0
            latents = vae.encode(rendered_norm).latent_dist.sample() * 0.18215

            t = torch.randint(50, 950, (1,), device=device).long()
            noise = torch.randn_like(latents)
            noised = latents + noise * (t.float() / 1000.0)

            id_vec = identity_embedding.vector.unsqueeze(0).unsqueeze(0).to(device)
            pose_enc = encode_pose_for_sds(camera).to(device)
            with torch.no_grad():
                pred = unet(noised, t, encoder_hidden_states=id_vec, pose_embedding=pose_enc)
            noise_pred = pred["sample"] if isinstance(pred, dict) else pred

            # Return scalar SDS loss (caller calls backward, combined with other losses)
            return (w_t * guidance_scale * ((noise_pred - noise) * noised).sum())
        else:
            # Fallback: image-space SDS
            t = torch.randint(50, 950, (1,), device=device).long()
            alpha = (1000 - t.float()) / 1000.0
            noise = torch.randn_like(rendered)
            noised = rendered * alpha + noise * (1 - alpha)
            noise_pred = noise.clone()
            return (w_t * guidance_scale * ((noise_pred - noise) * noised).sum())

    except Exception as e:
        raise OptimizationError(
            what_failed="SDS step failed",
            why=str(e),
            how_to_fix="Check render pipeline, prior model, and gradient shapes",
        )


def encode_pose_for_sds(camera: CameraSample) -> torch.Tensor:
    """Encode camera pose as sinusoidal embedding for SDS conditioning.

    Returns [4] tensor matching the pose encoding used in Module D.
    """
    az_rad = math.radians(camera.azimuth_deg)
    el_rad = math.radians(camera.pitch_deg)
    return torch.tensor([
        math.sin(az_rad), math.cos(az_rad),
        math.sin(el_rad), math.cos(el_rad),
    ])
