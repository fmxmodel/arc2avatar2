"""
Arc2Avatar — Data Contracts (Module K)
========================================
Directive 40: Every object that crosses a module boundary must have an explicit schema.
Directive 41: Ownership CRUD table lives in each subsystem's charter file.
Directive 42: Serialization format bindings — each object type has exactly one format.

No module may pass a bare dict or untyped tensor across a module boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Schema version ───────────────────────────────────────────────────────────
SCHEMA_VERSION: int = 1


# ── Schema dataclasses ───────────────────────────────────────────────────────

@dataclass
class GaussianState:
    """The complete state of all 3D Gaussian primitives.

    Shapes:
        means:    [N, 3]   float32 — 3D mean positions
        scales:   [N, 3]   float32 — pre-activation scales (exp() at render)
        rotations:[N, 4]   float32 — unit quaternions
        opacities:[N, 1]   float32 — pre-activation opacities (sigmoid at render)
        sh_coeffs:[N, 3, K] float32 — SH coefficients, K = (max_sh_degree+1)^2
        vertex_id:[N]      int64   — index into FlameMesh.V

    Invariant I2: means.shape[0] == vertex_id.shape[0] == N for all fields.
    """
    means: torch.Tensor      # [N, 3]  float32
    scales: torch.Tensor     # [N, 3]  float32, pre-activation
    rotations: torch.Tensor  # [N, 4]  float32, unit quaternion
    opacities: torch.Tensor  # [N, 1]  float32, pre-activation
    sh_coeffs: torch.Tensor  # [N, 3, K] float32
    vertex_id: torch.Tensor  # [N]     int64

    def __post_init__(self) -> None:
        """Enforce Invariant I2 at construction time."""
        n = self.means.shape[0]
        assert self.scales.shape[0] == n, \
            f"scales shape[0] {self.scales.shape[0]} != means shape[0] {n}"
        assert self.rotations.shape[0] == n, \
            f"rotations shape[0] {self.rotations.shape[0]} != means shape[0] {n}"
        assert self.opacities.shape[0] == n, \
            f"opacities shape[0] {self.opacities.shape[0]} != means shape[0] {n}"
        assert self.sh_coeffs.shape[0] == n, \
            f"sh_coeffs shape[0] {self.sh_coeffs.shape[0]} != means shape[0] {n}"
        assert self.vertex_id.shape[0] == n, \
            f"vertex_id shape[0] {self.vertex_id.shape[0]} != means shape[0] {n} (Invariant I2)"
        assert self.means.shape[1] == 3, f"means.shape[1] must be 3, got {self.means.shape[1]}"
        assert self.scales.shape[1] == 3, f"scales.shape[1] must be 3, got {self.scales.shape[1]}"
        assert self.rotations.shape[1] == 4, f"rotations.shape[1] must be 4, got {self.rotations.shape[1]}"


@dataclass
class FlameMesh:
    """FLAME template mesh — the geometric scaffold.

    Shapes:
        V:         [Nv, 3]      float32 — canonical vertex positions
        F:         [Nf, 3]      int64   — triangle indices
        shape_bs:  [Nv, 3, 300] float32 — shape blendshape basis
        expr_bs:   [Nv, 3, 100] float32 — expression blendshape basis
        pose_bs:   [Nv, 3, 36]  float32 — pose corrective basis

    Invariant I1: Nv is fixed at the canonical FLAME vertex count.
    """
    V: torch.Tensor        # [Nv, 3]      float32
    F: torch.Tensor        # [Nf, 3]      int64
    shape_bs: torch.Tensor # [Nv, 3, 300] float32
    expr_bs: torch.Tensor  # [Nv, 3, 100] float32
    pose_bs: torch.Tensor  # [Nv, 3, 36]  float32

    def __post_init__(self) -> None:
        nv = self.V.shape[0]
        assert self.V.shape[1] == 3, f"V.shape[1] must be 3, got {self.V.shape[1]}"
        assert self.F.shape[1] == 3, f"F.shape[1] must be 3, got {self.F.shape[1]}"
        assert self.shape_bs.shape == (nv, 3, 300), \
            f"shape_bs shape mismatch: {self.shape_bs.shape}"
        assert self.expr_bs.shape == (nv, 3, 100), \
            f"expr_bs shape mismatch: {self.expr_bs.shape}"
        assert self.pose_bs.shape == (nv, 3, 36), \
            f"pose_bs shape mismatch: {self.pose_bs.shape}"


@dataclass
class IdentityEmbedding:
    """The identity embedding vector for a subject.

    Shapes:
        vector: [512] float32, L2-normalized

    Invariant I3: dimensionality is fixed at 512.
    """
    vector: torch.Tensor         # [512] float32, L2-normalized
    source_image_hash: str       # sha256 of the exact input image bytes

    def __post_init__(self) -> None:
        assert self.vector.shape == (512,), \
            f"IdentityEmbedding vector shape must be (512,), got {self.vector.shape} (Invariant I3)"
        # Verify L2-normalized
        norm = self.vector.norm().item()
        assert abs(norm - 1.0) < 1e-5, \
            f"IdentityEmbedding vector must be L2-normalized, got norm={norm}"


@dataclass
class CameraSample:
    """A single camera sample for rendering.

    All angles in degrees; look_at is always (0,0,0) (head-centered).
    """
    azimuth_deg: float
    pitch_deg: float
    fov_rad: float
    radius: float
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RenderResult:
    """The result of a differentiable render pass."""
    image: torch.Tensor           # [3, H, W] float32, range [0, 1]
    camera: CameraSample
    requires_grad: bool = True    # True when produced inside an SDS step


@dataclass
class OptimizerState:
    """Snapshot of an optimizer's internal state for checkpoint/resume."""
    step_count: int
    param_groups: List[dict]      # torch optimizer internal state, wrapped
    lr_schedule_state: dict


@dataclass
class ExpressionState:
    """State of a single facial expression for animation.

    Each expression has blendshape coefficients and an optional refinement patch.
    """
    name: str
    flame_expr_coeffs: torch.Tensor   # [100] float32
    flame_pose_coeffs: torch.Tensor   # [36]  float32
    requires_refinement: bool = False  # set by Directive 25's trigger
    refined_patch: Optional[GaussianState] = None  # None until Directive 26 runs


@dataclass
class RunManifest:
    """Reproducibility manifest for a single run.

    Write-once per Directive 41: never modified after creation.
    """
    run_id: str
    config_snapshot: dict
    checkpoint_hashes: Dict[str, str]
    input_image_hash: str
    stage1_iterations_actual: int
    stage2_iterations_actual: int
    quality_gate_results: Dict[str, bool]
    schema_version: int = SCHEMA_VERSION


# ── Serialization helpers (Directive 42) ─────────────────────────────────────

def save_versioned(obj: Any, path: str, version: int = SCHEMA_VERSION) -> str:
    """Save an arbitrary object with schema version metadata.

    Always writes to a temporary path first, then atomically renames.
    Returns the content hash (sha256) of the saved file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    data = {"schema_version": version, "data": obj}
    torch.save(data, tmp_path)

    # Compute content hash before atomic rename
    with open(tmp_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    os.rename(tmp_path, path)
    return content_hash


def load_versioned(path: str, expected_version: int = SCHEMA_VERSION) -> Any:
    """Load a versioned checkpoint.

    Checks schema_version against expected_version (Invariant I4).
    Fails loudly on mismatch — no silent coercion.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError(
            f"Checkpoint {path} has no schema_version field. "
            f"Cannot verify Invariant I4."
        )

    actual_version = data["schema_version"]
    if actual_version != expected_version:
        raise ValueError(
            f"Checkpoint schema version mismatch: expected {expected_version}, "
            f"got {actual_version} (Invariant I4). Path: {path}"
        )

    return data["data"]


# ── Format-specific save/load helpers ────────────────────────────────────────

def save_gaussian_state(state: GaussianState, path: str) -> str:
    """Save a GaussianState as .pt with schema version (Directive 42 binding)."""
    assert path.endswith(".pt"), f"GaussianState must be saved as .pt, got: {path}"
    return save_versioned(state, path)


def load_gaussian_state(path: str) -> GaussianState:
    """Load a GaussianState from .pt."""
    assert path.endswith(".pt"), f"GaussianState must be loaded from .pt, got: {path}"
    return load_versioned(path)


def save_flame_mesh(mesh: FlameMesh, path: str) -> str:
    """Save a FlameMesh as .pt with schema version."""
    assert path.endswith(".pt"), f"FlameMesh must be saved as .pt, got: {path}"
    return save_versioned(mesh, path)


def load_flame_mesh(path: str) -> FlameMesh:
    """Load a FlameMesh from .pt."""
    assert path.endswith(".pt")
    return load_versioned(path)


def save_identity_embedding(
    embedding: IdentityEmbedding,
    npy_path: str,
    json_path: str,
) -> str:
    """Save IdentityEmbedding as .npy (vector) + .json (sidecar hash).

    The source_image_hash is NEVER packed into the .npy file (Directive 42 hard rule).
    """
    assert npy_path.endswith(".npy"), f"Vector must be saved as .npy, got: {npy_path}"
    assert json_path.endswith(".json"), f"Metadata must be saved as .json, got: {json_path}"

    np.save(npy_path, embedding.vector.cpu().numpy())

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "source_image_hash": embedding.source_image_hash,
    }
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    # Return combined content hash
    with open(npy_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_identity_embedding(npy_path: str, json_path: str) -> IdentityEmbedding:
    """Load IdentityEmbedding from .npy + .json pair."""
    assert npy_path.endswith(".npy")
    assert json_path.endswith(".json")

    vector = torch.from_numpy(np.load(npy_path)).float()

    with open(json_path, "r") as f:
        meta = json.load(f)

    return IdentityEmbedding(
        vector=vector,
        source_image_hash=meta.get("source_image_hash", ""),
    )


def save_run_manifest(manifest: RunManifest, path: str) -> None:
    """Save RunManifest as .json with schema_version."""
    assert path.endswith(".json"), f"RunManifest must be saved as .json, got: {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = asdict(manifest)
    data["schema_version"] = SCHEMA_VERSION
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_run_manifest(path: str) -> RunManifest:
    """Load RunManifest from .json."""
    assert path.endswith(".json")
    with open(path, "r") as f:
        data = json.load(f)
    return RunManifest(**{k: v for k, v in data.items() if k != "schema_version"})


def compute_file_hash(path: str) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
