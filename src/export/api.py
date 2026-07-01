"""EXPORT subsystem — Export & Packaging (Module H)
Directives 28-31, 86.
"""

def export_static_ply(gaussian_state, output_path: str) -> None:
    """
    Inputs:    GaussianState (stage2_full_head), output path.
    Outputs:   None (writes .ply file).
    Exceptions: raises ExportError on write failure.
    Side effects: writes subject_static.ply.
    """
    pass


def render_turntable(gaussian_state, config) -> None:
    """
    Inputs:    GaussianState, ExportConfig.
    Outputs:   None (writes video file).
    Exceptions: raises RenderingError on render failure.
    Side effects: writes turntable_360.mp4.
    """
    pass


def package_animation_bundle(gaussian_state, expression_states, flame_mesh, output_dir: str) -> None:
    """
    Inputs:    GaussianState, list of ExpressionState, FlameMesh, output directory.
    Outputs:   None (writes bundle files).
    Exceptions: raises ExportError on packaging failure.
    Side effects: writes animation_bundle/ with manifest.json.
    """
    pass


def write_run_manifest(config, checkpoint_hashes, quality_results, output_path: str) -> None:
    """
    Inputs:    resolved PipelineConfig, checkpoint hash dict, quality gate results, output path.
    Outputs:   None (writes JSON file — write-once, never modified).
    Exceptions: raises ExportError if file already exists.
    Side effects: writes run_manifest.json.
    """
    pass
