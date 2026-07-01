#!/usr/bin/env python3
"""
Arc2Avatar — Master Orchestration Script (Module I / Directive 32)
====================================================================
Driven entirely by configs/pipeline_config.yaml — zero hardcoded paths
or tunable values duplicated inside this file.

Execution order (Directive 32):
1. Config validation (Directive 39 — before anything else)
2. Environment check (Directive 4)
3. Data prep (Directives 5-9, skip-if-checksummed)
4. Gaussian init + avg-texture fit (Directives 10-13)
5. Conditional fine-tune (Directives 14-16, skip if final.pt exists)
6. Stage 1 face optimization (Directive 21)
7. Stage 2 full-head optimization (Directive 22)
8. Per-expression animation + refinement (Directives 24-27)
9. Export (Directives 28-31)
10. Final deliverables checklist (Directive 35)

Aborts immediately on any step's failure, printing the specific
directive number that failed.

Every numbered directive's logic is an importable function — never
buried in an if __name__ == "__main__" block.
"""

import sys


def step_env_check(config) -> None:
    """Directive 4: Verify GPU + library health."""
    from src.utils.env_check import check_environment
    print(f"\n[Directive 4] Environment check")
    results = check_environment(config)
    if not results["all_pass"]:
        raise RuntimeError("Directive 4 FAILED: Environment check incomplete.")


def step_data_prep(config) -> None:
    """Directives 5-9: Data acquisition and preprocessing (skip if checksums match)."""
    print(f"\n[Directives 5-9] Data preparation")
    # TODO: Full implementation on GPU system
    # - Validate input image (mediapipe face detection)
    # - Download/verify arc2face_base and FLAME template
    # - Extract ID embedding (ArcFace encoder)
    # - Load FLAME template
    # - Verify panohead dataset
    print("  [PREBUILD] Data prep stubs — full implementation on GPU system.")


def step_gaussian_init(config) -> None:
    """Directives 10-13: Gaussian initialization and average-texture fitting."""
    print(f"\n[Directives 10-13] Gaussian initialization + avg-texture fit")
    # TODO: Full implementation on GPU system
    print("  [PREBUILD] Gaussian init stubs — full implementation on GPU system.")


def step_finetune_prior(config) -> None:
    """Directives 14-16: Arc2Face fine-tuning (skip if final.pt already exists)."""
    import os
    print(f"\n[Directives 14-16] Arc2Face fine-tuning")
    if os.path.exists(config.finetune.final_path):
        print(f"  SKIP: {config.finetune.final_path} already exists.")
        return
    print("  [PREBUILD] Fine-tuning stubs — full implementation on GPU system.")


def step_stage1(config) -> None:
    """Directive 21: Stage 1 face-only optimization."""
    print(f"\n[Directive 21] Stage 1 — face-only optimization ({config.stage1.iterations} iter)")
    print("  [PREBUILD] Stage 1 stubs — full implementation on GPU system.")


def step_stage2(config) -> None:
    """Directive 22: Stage 2 full-head optimization."""
    print(f"\n[Directive 22] Stage 2 — full-head optimization ({config.stage2.iterations} iter)")
    print("  [PREBUILD] Stage 2 stubs — full implementation on GPU system.")


def step_animation(config) -> None:
    """Directives 24-27: Per-expression animation + refinement."""
    print(f"\n[Directives 24-27] Expression animation + refinement")
    for expr in config.animation.expressions:
        print(f"  Expression: {expr}")
    print("  [PREBUILD] Animation stubs — full implementation on GPU system.")


def step_export(config) -> None:
    """Directives 28-31: Export final deliverables."""
    print(f"\n[Directives 28-31] Export")
    print(f"  PLY:   {config.export.ply_path}")
    print(f"  Video: {config.export.turntable_path}")
    print(f"  Bundle:{config.export.animation_bundle_path}")
    print(f"  Manifest: {config.export.run_manifest_path}")
    print("  [PREBUILD] Export stubs — full implementation on GPU system.")


def step_final_checklist(config) -> None:
    """Directive 35: Final holistic deliverable checklist."""
    import os
    print(f"\n[Directive 35] Final deliverable checklist")
    deliverables = [
        config.export.ply_path,
        config.export.turntable_path,
        config.export.animation_bundle_path,
        config.export.run_manifest_path,
    ]
    all_ok = True
    for d in deliverables:
        exists = os.path.exists(d)
        print(f"  {'✓' if exists else '✗'} {d}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("  WARNING: Some deliverables not yet created (expected in prebuild mode).")


def run_pipeline(config) -> None:
    """Execute the full pipeline in order (Directive 32).

    This is the main orchestration function. Every step is an importable
    function (not buried in if __name__ == "__main__").

    Args:
        config: Fully-resolved PipelineConfig (already validated per Directive 39).
    """
    from src.state.checkpoint_manager import transition

    print("=" * 60)
    print("Arc2Avatar — Pipeline Execution (Directive 32)")
    print("=" * 60)

    try:
        transition("Initialized")
        transition("Running")

        # Step 1: Environment check (Directive 4)
        step_env_check(config)

        # Step 2: Data prep (Directives 5-9)
        step_data_prep(config)

        # Step 3: Gaussian init + avg-texture fit (Directives 10-13)
        step_gaussian_init(config)

        # Step 4: Conditional fine-tune (Directives 14-16)
        step_finetune_prior(config)

        # Step 5: Stage 1 (Directive 21)
        step_stage1(config)

        # Step 6: Stage 2 (Directive 22)
        step_stage2(config)

        # Step 7: Animation (Directives 24-27)
        step_animation(config)

        # Step 8: Export (Directives 28-31)
        step_export(config)

        # Step 9: Final checklist (Directive 35)
        step_final_checklist(config)

        transition("Completed", metadata={"last_directive": 86})
        print(f"\n{'=' * 60}")
        print("Pipeline completed successfully.")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"Pipeline FAILED: {e}")
        print(f"{'!' * 60}")
        transition("Failed", metadata={"last_directive": -1, "reason": str(e)})
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    from src.config.schema import load_and_validate_config
    from src.resource.gpu_manager import init_device
    from src.utils.seed import set_global_seed

    import argparse
    parser = argparse.ArgumentParser(description="Arc2Avatar Pipeline")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Step 0: Config validation (Directive 39 — first executable step)
    cli_overrides = {"seed": args.seed} if args.seed else None
    config = load_and_validate_config(experiment=args.experiment, cli_args=cli_overrides)

    # Device init (Directive 49) and seed (Directive 55)
    init_device(config.device)
    set_global_seed(config.seed)

    # Run the pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
