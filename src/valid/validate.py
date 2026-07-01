"""
Arc2Avatar — Validation Module (Module Q)
===========================================
Directive 59: Per-module re-check of every api.py return value.
Directive 60: Mechanical acceptance criteria table.
Directive 61: Base sanity checks (NaN/Inf, zero-length, missing files, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ValidationReport:
    """Report from a single validation check."""
    subsystem: str
    passed: bool
    checks: Dict[str, bool]
    details: Dict[str, str]
    errors: List[str]


def base_sanity_check(obj: Any, name: str = "object") -> ValidationReport:
    """Base sanity checker (Directive 61).

    Runs on ANY schema object before subsystem-specific checks:
    - NaN/Inf in any tensor field
    - Zero-length tensors where positive length required
    - Missing files where a field is a path
    - Mesh validity (for FlameMesh)
    - Embedding unit-norm (for IdentityEmbedding)
    """
    checks: Dict[str, bool] = {}
    details: Dict[str, str] = {}
    errors: List[str] = []

    if is_dataclass(obj):
        import inspect
        from dataclasses import fields

        for f in fields(obj):
            val = getattr(obj, f.name)

            # Check tensors for NaN/Inf
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any():
                    checks[f"tensor_{f.name}_nan"] = False
                    errors.append(f"{name}.{f.name} contains NaN values")
                else:
                    checks[f"tensor_{f.name}_nan"] = True

                if torch.isinf(val).any():
                    checks[f"tensor_{f.name}_inf"] = False
                    errors.append(f"{name}.{f.name} contains Inf values")
                else:
                    checks[f"tensor_{f.name}_inf"] = True

                # Check zero-length
                if val.numel() == 0:
                    checks[f"tensor_{f.name}_nonempty"] = False
                    errors.append(f"{name}.{f.name} is empty (0 elements)")
                else:
                    checks[f"tensor_{f.name}_nonempty"] = True

            # Check string paths for existence
            if isinstance(val, str) and val.endswith((".pt", ".npy", ".json", ".yaml", ".ply")):
                import os
                exists = os.path.exists(val)
                checks[f"path_{f.name}_exists"] = exists
                if not exists:
                    details[f"path_{f.name}"] = f"not found: {val}"

    # Special checks for known types
    type_name = type(obj).__name__

    if type_name in ("FlameMesh", "FaceVerseMesh"):
        # Check no degenerate faces
        if hasattr(obj, "F") and isinstance(obj.F, torch.Tensor):
            # Check for faces where all three vertices are the same
            degen = (obj.F[:, 0] == obj.F[:, 1]) & (obj.F[:, 1] == obj.F[:, 2])
            if degen.any():
                checks["mesh_no_degenerate_faces"] = False
                errors.append(f"{type_name} has {degen.sum().item()} degenerate (zero-area) faces")
            else:
                checks["mesh_no_degenerate_faces"] = True

    if type_name == "IdentityEmbedding":
        # Check unit-norm
        if hasattr(obj, "vector") and isinstance(obj.vector, torch.Tensor):
            norm = obj.vector.norm().item()
            is_unit = abs(norm - 1.0) < 1e-5
            checks["embedding_unit_norm"] = is_unit
            if not is_unit:
                errors.append(f"IdentityEmbedding norm={norm:.6f}, expected 1.0")

    return ValidationReport(
        subsystem="VALID",
        passed=len(errors) == 0,
        checks=checks,
        details=details,
        errors=errors,
    )


def validate_stage1_output(
    state: Any,
    connectivity_loss: float,
    id_similarity_pre: float,
    id_similarity_post: float,
    config: Any,
) -> ValidationReport:
    """Validate Stage 1 output (Directive 60 acceptance criteria).

    Mechanical criteria:
    1. connectivity loss below threshold
    2. ID cosine similarity vs. embedding has INCREASED from average-texture baseline
    """
    checks: Dict[str, bool] = {}
    errors: List[str] = []

    # Run base sanity first
    base = base_sanity_check(state, "GaussianState(stage1)")
    if not base.passed:
        return base

    # Criterion 1: connectivity loss below threshold
    conn_threshold = getattr(config, "connectivity", None)
    max_loss = getattr(conn_threshold, "drift_tolerance", 1e-4) * 10
    checks["connectivity_loss_below_threshold"] = connectivity_loss < max_loss
    if not checks["connectivity_loss_below_threshold"]:
        errors.append(
            f"Stage 1 connectivity loss {connectivity_loss:.6f} exceeds threshold {max_loss:.6f}"
        )

    # Criterion 2: ID similarity has increased
    checks["id_similarity_increased"] = id_similarity_post > id_similarity_pre
    if not checks["id_similarity_increased"]:
        errors.append(
            f"Stage 1 ID similarity {id_similarity_post:.4f} did not increase "
            f"from baseline {id_similarity_pre:.4f}"
        )

    return ValidationReport(
        subsystem="OPT",
        passed=len(errors) == 0,
        checks=checks,
        details={
            "connectivity_loss": f"{connectivity_loss:.6f}",
            "id_similarity_pre": f"{id_similarity_pre:.4f}",
            "id_similarity_post": f"{id_similarity_post:.4f}",
        },
        errors=errors,
    )


def validate_stage2_output(
    state: Any,
    rendered_views: Dict[str, torch.Tensor],
    config: Any,
) -> ValidationReport:
    """Validate Stage 2 output (Directive 60 acceptance criteria).

    Mechanical criteria:
    1. No NaN/Inf in any Gaussian parameter (handled by base sanity)
    2. No fully-black or fully-uniform-color frame across 8 canonical azimuths
    """
    checks: Dict[str, bool] = {}
    errors: List[str] = []

    # Run base sanity
    base = base_sanity_check(state, "GaussianState(stage2)")
    if not base.passed:
        return base

    # Criterion 2: Check rendered views for black/uniform frames
    all_valid = True
    for angle, img in rendered_views.items():
        if img.numel() == 0:
            checks[f"render_{angle}_nonempty"] = False
            errors.append(f"Empty render at angle {angle}")
            all_valid = False
            continue

        # Check for fully black (all pixels near 0)
        mean_intensity = img.mean().item()
        if mean_intensity < 0.01:
            checks[f"render_{angle}_not_black"] = False
            errors.append(f"Nearly black render at angle {angle} (mean={mean_intensity:.4f})")
            all_valid = False
        else:
            checks[f"render_{angle}_not_black"] = True

        # Check for fully uniform (std near 0)
        std_intensity = img.std().item()
        if std_intensity < 0.005:
            checks[f"render_{angle}_not_uniform"] = False
            errors.append(f"Uniform-color render at angle {angle} (std={std_intensity:.4f})")
            all_valid = False
        else:
            checks[f"render_{angle}_not_uniform"] = True

    checks["all_renders_valid"] = all_valid
    return ValidationReport(
        subsystem="OPT",
        passed=len(errors) == 0,
        checks=checks,
        details={"view_count": str(len(rendered_views))},
        errors=errors,
    )


def validate_refinement_output(
    cosine_similarity: float,
    config: Any,
) -> ValidationReport:
    """Validate refinement patch output (Directive 60 / Directive 27).

    Mechanical criterion: ID cosine similarity >= configured threshold.
    """
    threshold = getattr(config, "id_similarity_threshold", 0.85)
    passed = cosine_similarity >= threshold

    errors = []
    if not passed:
        errors.append(
            f"Refinement ID similarity {cosine_similarity:.4f} "
            f"below threshold {threshold:.4f}"
        )

    return ValidationReport(
        subsystem="ANIM",
        passed=passed,
        checks={"id_similarity_above_threshold": passed},
        details={
            "cosine_similarity": f"{cosine_similarity:.4f}",
            "threshold": str(threshold),
        },
        errors=errors,
    )
