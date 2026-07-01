"""CONTRACTS subsystem — Data Contracts (Module K)
Directives 40-42.
"""
# All schemas live in src/contracts/schemas.py
# This api.py re-exports the public interface.

from src.contracts.schemas import (
    GaussianState,
    FaceVerseMesh,
    IdentityEmbedding,
    CameraSample,
    RenderResult,
    OptimizerState,
    ExpressionState,
    RunManifest,
    save_versioned,
    load_versioned,
    save_gaussian_state,
    load_gaussian_state,
    save_faceverse_mesh,
    load_faceverse_mesh,
    save_identity_embedding,
    load_identity_embedding,
    save_run_manifest,
    load_run_manifest,
    compute_file_hash,
    SCHEMA_VERSION,
)
