"""
Arc2Avatar — Centralized Configuration Schema (Module J)
========================================================
Directive 36: All tunable values from Part 1 must be fields in this dataclass hierarchy.
Directive 37: Four-layer config merge (system defaults → experiment → project → CLI).
Directive 38-39: Validation before any computation.

No file outside this hierarchy may contain a bare numeric literal that duplicates a
Part 1 reference value (500, 0.4, -110, 110, 60, 90, -180, 180, 30, 120, etc.).
"""

import copy
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ── FLAME constants ──────────────────────────────────────────────────────────
FLAME_CANONICAL_VERT_COUNT: int = 5023  # FLAME 2023 canonical vertex count
FLAME_CANONICAL_FACE_COUNT: int = 9976  # FLAME 2023 canonical face count
FLAME_SHAPE_BS_DIM: int = 300
FLAME_EXPR_BS_DIM: int = 100
FLAME_POSE_BS_DIM: int = 36
ID_EMBEDDING_DIM: int = 512


# ── Module-level config dataclasses ──────────────────────────────────────────

@dataclass
class EnvCheckConfig:
    """Configuration for Module A environment checks (Directive 3-4)."""
    cuda_required: bool = True
    pytorch3d_required: bool = True
    kaolin_required: bool = True
    diffusers_required: bool = True
    python_min_version: str = "3.10.0"


@dataclass
class DataPrepConfig:
    """Configuration for Module B data acquisition (Directives 5-9).

    Reference values (not casually overrideable):
    - min_face_bbox_size: 256 px
    - min_panohead_identities: 1000
    - min_panohead_angles: 8
    """
    input_image_path: str = "data/raw_input/subject.png"
    min_face_bbox_size: int = 256
    arc2face_base_path: str = "checkpoints/arc2face_base/"
    flame_template_path: str = "data/flame_template/generic_model.pkl"
    flame_obj_path: str = "data/flame_template/template.obj"
    flame_loaded_path: str = "data/flame_template/flame_loaded.pt"
    id_embedding_path: str = "data/embeddings/subject_id_embedding.npy"
    id_embedding_json_path: str = "data/embeddings/subject_id_embedding.json"
    panohead_path: str = "data/panohead_synth/"
    min_panohead_identities: int = 1000
    min_panohead_angles: int = 8
    dataset_version: str = "v1"
    dataset_manifest_path: str = "data/panohead_synth/dataset_manifest.json"


@dataclass
class GaussianInitConfig:
    """Configuration for Module C initialization (Directives 10-13).

    Reference values (not casually overrideable):
    - init_scale: 0.002 in FLAME unit space
    - avg_texture_iterations: 100-150 (tunable within range)
    """
    init_scale: float = 0.002
    init_opacity: float = 0.5  # mid-value, stored as inverse-sigmoid
    max_sh_degree: int = 3  # K = (max_sh_degree+1)^2
    init_state_path: str = "checkpoints/gaussians/init_state.pt"
    avg_texture_path: str = "checkpoints/gaussians/avg_texture_fit.pt"
    avg_texture_iterations: int = 150
    avg_texture_lr: float = 1e-2


@dataclass
class ConnectivityConfig:
    """Configuration for the connectivity regularizer (Directive 12)."""
    k_neighbors: int = 8
    weight: float = 0.1
    drift_tolerance: float = 1e-4


@dataclass
class FinetuneConfig:
    """Configuration for Module D Arc2Face fine-tuning (Directives 14-16)."""
    arch_init_path: str = "checkpoints/arc2face_finetuned/arch_init.pt"
    final_path: str = "checkpoints/arc2face_finetuned/final.pt"
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 10
    log_interval_steps: int = 100
    validation_angles_deg: Tuple[float, ...] = (
        0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0
    )
    freeze_early_layers: bool = True
    num_early_layers_frozen: int = 4


@dataclass
class Stage1Config:
    """Configuration for Stage 1 face-only optimization (Directive 21).

    Reference values (NOT tunable — do not override casually):
    - iterations: 500
    - fov_radians: 0.4
    - azimuth_range_deg: (-110.0, 110.0)
    - pitch_range_deg: (60.0, 90.0)
    - guidance_scale: 5.0 (in 3-7 range per Directive 20)
    """
    iterations: int = 500
    fov_radians: float = 0.4
    azimuth_range_deg: Tuple[float, float] = (-110.0, 110.0)
    pitch_range_deg: Tuple[float, float] = (60.0, 90.0)
    guidance_scale: float = 5.0
    lr_position: float = 1e-3
    lr_color: float = 5e-4
    log_interval: int = 50
    checkpoint_path: str = "checkpoints/gaussians/stage1_face.pt"


@dataclass
class Stage2Config:
    """Configuration for Stage 2 full-head optimization (Directive 22).

    Reference values:
    - azimuth_range_deg: (-180.0, 180.0)
    - pitch_range_deg: (30.0, 120.0)
    - iterations: 1500-3000 (tunable)
    """
    iterations: int = 2000
    fov_radians: float = 0.4
    azimuth_range_deg: Tuple[float, float] = (-180.0, 180.0)
    pitch_range_deg: Tuple[float, float] = (30.0, 120.0)
    guidance_scale: float = 5.0
    lr_position: float = 1e-3
    lr_color: float = 5e-4
    log_interval: int = 50
    checkpoint_path: str = "checkpoints/gaussians/stage2_full_head.pt"


@dataclass
class SDSConfig:
    """Configuration for the SDS engine (Module E, Directives 17-20).

    Reference:
    - guidance_scale: 3-7 range (well below text-to-image SDS defaults of 15-100)
    """
    guidance_scale_default: float = 5.0
    render_width: int = 512
    render_height: int = 512
    camera_radius: float = 2.0


@dataclass
class DivergenceGuardConfig:
    """Configuration for the divergence guard (Directive 23)."""
    std_dev_threshold: float = 3.0
    trailing_window: int = 50
    max_retries: int = 2
    guidance_scale_halve_factor: float = 0.5


@dataclass
class AnimationConfig:
    """Configuration for Module G animation (Directives 24-27)."""
    id_similarity_threshold: float = 0.85
    mouth_open_jaw_threshold: float = 0.3
    refinement_guidance_scale: float = 7.0
    refinement_iterations: int = 200
    refinement_lr: float = 1e-3
    expressions: Tuple[str, ...] = ("neutral", "smile", "open_mouth", "raised_brow")


@dataclass
class ExportConfig:
    """Configuration for Module H export (Directives 28-31)."""
    ply_path: str = "outputs/final_avatar/subject_static.ply"
    turntable_path: str = "outputs/renders/turntable_360.mp4"
    animation_bundle_path: str = "outputs/final_avatar/animation_bundle/"
    animation_manifest_path: str = "outputs/final_avatar/animation_bundle/manifest.json"
    run_manifest_path: str = "outputs/final_avatar/run_manifest.json"
    turntable_angle_step_deg: float = 5.0


@dataclass
class ManualApprovalConfig:
    """Human-in-the-loop quality gates (Directive 34)."""
    require_approval_after_finetune: bool = False
    require_approval_after_stage1: bool = False
    require_approval_after_stage2: bool = False
    require_approval_after_refinement: bool = False
    confirmation_file_pattern: str = ".approval_{stage}_confirmed"


@dataclass
class ResourceBudgetConfig:
    """Per-subsystem VRAM budget in MB (Directive 50)."""
    PRIOR: int = 12000
    GAUSS: int = 4000
    SDS: int = 6000
    OPT: int = 8000
    ANIM: int = 4000
    default: int = 2000


@dataclass
class LoggingConfig:
    """Configuration for Module P logging (Directives 56-58)."""
    structured_log_dir: str = "logs/structured/"
    artifact_dir: str = "logs/artifacts/"
    tensorboard_dir: str = "logs/tensorboard/"
    wandb_dir: str = "logs/wandb/"
    log_level: str = "INFO"


@dataclass
class ExperimentConfig:
    """Configuration for Module T experiment management (Directives 69-72)."""
    registry_path: str = "logs/experiment_registry.jsonl"
    benchmarks_history_path: str = "logs/benchmarks/history.csv"


# ── Top-level pipeline config ────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configuration — one nested dataclass per Part-1 module."""
    env_check: EnvCheckConfig = field(default_factory=EnvCheckConfig)
    data_prep: DataPrepConfig = field(default_factory=DataPrepConfig)
    gaussian_init: GaussianInitConfig = field(default_factory=GaussianInitConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    sds: SDSConfig = field(default_factory=SDSConfig)
    divergence_guard: DivergenceGuardConfig = field(default_factory=DivergenceGuardConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    manual_approval: ManualApprovalConfig = field(default_factory=ManualApprovalConfig)
    resource_budget: ResourceBudgetConfig = field(default_factory=ResourceBudgetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Global settings
    seed: int = 42
    run_id: str = ""
    device: str = "cuda:0"  # Will be resolved by RESOURCE module
    data_root: str = "."
    checkpoint_root: str = "."
    output_root: str = "."
    run_state_path: str = "run_state.json"
    run_report_path: str = "outputs/final_avatar/run_report.json"


# ── Validation ───────────────────────────────────────────────────────────────

@dataclass
class ValidationError:
    """A single config validation error."""
    field_path: str
    message: str
    observed_value: Optional[Any] = None
    expected: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.field_path}] {self.message}" + (
            f" (got: {self.observed_value})" if self.observed_value is not None else ""
        ) + (
            f" (expected: {self.expected})" if self.expected else ""
        )


def _field_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _validate_dataclass(
    obj: Any,
    path: str,
    errors: List[ValidationError],
) -> None:
    """Recursively validate a dataclass instance and its nested fields."""
    if not is_dataclass(obj):
        return

    for f in fields(obj):
        fp = _field_path(path, f.name)
        val = getattr(obj, f.name)

        # Recurse into nested dataclasses
        if is_dataclass(val):
            _validate_dataclass(val, fp, errors)
            continue

        # Type-specific checks
        if f.type is int or f.type is float or str(f.type) in ("int", "float"):
            if val is not None and val <= 0:
                errors.append(ValidationError(
                    field_path=fp,
                    message="must be a positive number",
                    observed_value=val,
                ))

        if "guidance_scale" in f.name:
            if val is not None and not (0 < val < 20):
                errors.append(ValidationError(
                    field_path=fp,
                    message="guidance scale must be positive and reasonable (< 20)",
                    observed_value=val,
                ))

        if "iterations" in f.name:
            if val is not None and (not isinstance(val, int) or val < 1):
                errors.append(ValidationError(
                    field_path=fp,
                    message="iteration count must be a positive integer",
                    observed_value=val,
                ))

        if "azimuth" in f.name.lower() or "pitch" in f.name.lower():
            if isinstance(val, tuple) and len(val) == 2:
                if val[0] >= val[1]:
                    errors.append(ValidationError(
                        field_path=fp,
                        message="angle range must satisfy min < max",
                        observed_value=val,
                    ))


def validate_config(cfg: PipelineConfig) -> List[ValidationError]:
    """Validate a fully-resolved PipelineConfig (Directive 38).

    Checks: path fields exist/is creatable, angle ranges, iteration counts,
    guidance_scale > 0, enum fields are legal members.
    """
    errors: List[ValidationError] = []
    _validate_dataclass(cfg, "", errors)

    # Path validations (check parent directories exist or are creatable)
    path_fields = [
        ("data_prep.input_image_path", cfg.data_prep.input_image_path),
        ("data_prep.flame_template_path", cfg.data_prep.flame_template_path),
        ("gaussian_init.init_state_path", cfg.gaussian_init.init_state_path),
        ("finetune.final_path", cfg.finetune.final_path),
        ("stage1.checkpoint_path", cfg.stage1.checkpoint_path),
        ("stage2.checkpoint_path", cfg.stage2.checkpoint_path),
        ("export.ply_path", cfg.export.ply_path),
        ("export.turntable_path", cfg.export.turntable_path),
        ("export.run_manifest_path", cfg.export.run_manifest_path),
    ]
    for field_name, path_val in path_fields:
        if path_val:
            parent = os.path.dirname(path_val) if os.path.dirname(path_val) else "."
            if not os.path.exists(parent):
                errors.append(ValidationError(
                    field_path=field_name,
                    message=f"parent directory does not exist: {parent}",
                    observed_value=path_val,
                    expected="existing parent directory",
                ))

    return errors


# ── Config loading / merging ─────────────────────────────────────────────────

def _deep_merge(base: Any, override: Any) -> Any:
    """Recursive deep merge of two objects.

    If both are dataclasses of the same type, merge field by field.
    If both are dicts, merge key by key.
    Otherwise, override wins.
    """
    if base is None or override is None:
        return override if override is not None else base

    if is_dataclass(base) and is_dataclass(override) and type(base) is type(override):
        merged = copy.deepcopy(base)
        for f in fields(base):
            setattr(merged, f.name, _deep_merge(
                getattr(base, f.name), getattr(override, f.name)
            ))
        return merged

    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for k, v in override.items():
            if k in merged:
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged

    # Leaf values: override wins
    return copy.deepcopy(override)


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning empty dict if not found."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Recursively convert a dict (from YAML) into a nested dataclass tree."""
    if not is_dataclass(cls):
        return data

    field_defs = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            ftype = f.type
            # Handle nested dataclasses
            if is_dataclass(ftype) and isinstance(val, dict):
                kwargs[f.name] = _dict_to_dataclass(val, ftype)
            elif hasattr(ftype, "__origin__") and ftype.__origin__ is tuple:
                # Tuple types — keep as is if list/tuple
                kwargs[f.name] = tuple(val) if isinstance(val, (list, tuple)) else val
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def resolve_config(
    experiment: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """Resolve configuration through four-layer merge (Directive 37).

    Layers (later overrides earlier):
    1. System defaults (baked into dataclass field defaults)
    2. Experiment YAML override
    3. Project overrides YAML
    4. CLI runtime overrides

    Returns a fully-resolved PipelineConfig.
    """
    # Layer 1: System defaults (already baked into dataclass __init__)
    cfg = PipelineConfig()

    # Layer 2: Experiment defaults
    if experiment:
        exp_path = f"configs/experiments/{experiment}.yaml"
        exp_data = _load_yaml(exp_path)
        if exp_data:
            exp_cfg = _dict_to_dataclass(exp_data, PipelineConfig)
            cfg = _deep_merge(cfg, exp_cfg)

    # Layer 3: Project overrides
    proj_data = _load_yaml("configs/project_overrides.yaml")
    if proj_data:
        proj_cfg = _dict_to_dataclass(proj_data, PipelineConfig)
        cfg = _deep_merge(cfg, proj_cfg)

    # Layer 4: CLI runtime overrides
    if cli_args:
        cli_cfg = _dict_to_dataclass(cli_args, PipelineConfig)
        cfg = _deep_merge(cfg, cli_cfg)

    return cfg


def load_and_validate_config(
    experiment: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """Load, resolve, and validate config (Directive 39).

    This is the literal first executable step of the entire system.
    Prints EVERY validation error and exits non-zero if any are found.
    """
    cfg = resolve_config(experiment, cli_args)
    errors = validate_config(cfg)

    if errors:
        print("=" * 60)
        print("CONFIGURATION VALIDATION FAILED")
        print("=" * 60)
        for err in errors:
            print(f"  {err}")
        print(f"\n{len(errors)} error(s) found. Aborting before any computation.")
        sys.exit(1)

    return cfg
