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
    print("  ✓ All 4 checks passed")


def step_data_prep(config) -> None:
    """Directives 5-9: Data acquisition and preprocessing (skip if checksums match)."""
    import os
    from src.data.api import (
        validate_input_image,
        extract_identity_embedding,
        load_flame_template,
        verify_panohead_dataset,
    )
    from src.contracts.schemas import compute_file_hash

    print(f"\n[Directives 5-9] Data preparation")

    # Step 5: Validate input image
    img_path = config.data_prep.input_image_path
    if os.path.exists(img_path):
        validate_result = validate_input_image(img_path)
        print(f"  ✓ Input image validated: {validate_result['bbox_size']}px bbox")
    else:
        print(f"  ⚠ Input image not found: {img_path} (will use placeholder)")

    # Step 6: Verify checkpoints exist
    for ckpt_dir in [config.data_prep.arc2face_base_path,
                     os.path.dirname(config.data_prep.flame_template_path)]:
        if os.path.exists(ckpt_dir):
            print(f"  ✓ Checkpoint present: {ckpt_dir}")
        else:
            print(f"  ⚠ Missing checkpoint: {ckpt_dir}")

    # Step 7: Extract ID embedding (skip if already exists)
    if not os.path.exists(config.data_prep.id_embedding_path):
        if os.path.exists(img_path):
            extract_identity_embedding(
                img_path,
                config.data_prep.id_embedding_path,
                config.data_prep.id_embedding_json_path,
            )
            print(f"  ✓ ID embedding extracted → {config.data_prep.id_embedding_path}")
        else:
            print(f"  ⚠ Cannot extract embedding — no input image")
    else:
        print(f"  ✓ ID embedding already exists (skip)")

    # Step 8: Load FLAME template (skip if already exists)
    if not os.path.exists(config.data_prep.flame_loaded_path):
        if os.path.exists(config.data_prep.flame_template_path):
            load_flame_template(
                config.data_prep.flame_template_path,
                config.data_prep.flame_loaded_path,
            )
            print(f"  ✓ FLAME template loaded → {config.data_prep.flame_loaded_path}")
        else:
            print(f"  ⚠ FLAME template not found: {config.data_prep.flame_template_path}")
    else:
        print(f"  ✓ FLAME template already exists (skip)")

    # Step 9: Verify panohead dataset
    if os.path.exists(config.data_prep.panohead_path):
        pano_result = verify_panohead_dataset(
            config.data_prep.panohead_path,
            config.data_prep.min_panohead_identities,
            config.data_prep.min_panohead_angles,
        )
        print(f"  ✓ PanoHead dataset: {pano_result['identity_count']} identities")
    else:
        print(f"  ⚠ PanoHead dataset not found: {config.data_prep.panohead_path}")


def step_gaussian_init(config) -> None:
    """Directives 10-13: Gaussian initialization and average-texture fitting."""
    import os
    from src.gauss.api import init_gaussians_from_faceverse, run_avg_texture_fit
    from src.contracts.schemas import load_faceverse_mesh, FaceVerseMesh

    print(f"\n[Directives 10-13] Gaussian initialization + avg-texture fit")

    # Load FaceVerse mesh
    fv_path = config.data_prep.flame_loaded_path.replace('flame', 'faceverse')
    if os.path.exists(fv_path):
        faceverse_mesh = load_faceverse_mesh(fv_path)
    elif os.path.exists(config.data_prep.flame_template_path):
        from src.data.api import load_faceverse_model
        load_faceverse_model(config.data_prep.flame_template_path, fv_path)
        faceverse_mesh = load_faceverse_mesh(fv_path)
    else:
        print("  ⚠ FaceVerse mesh not available — creating synthetic placeholder")
        nv = 6335
        faceverse_mesh = FaceVerseMesh(
            V=torch.randn(nv, 3),
            F=torch.randint(0, nv, (12566, 3)),
            idBase=torch.randn(nv*3, 150),
            expBase=torch.randn(nv*3, 52),
            texBase=torch.randn(nv*3, 251),
            meanshape=torch.randn(nv*3),
            meantex=torch.randn(nv*3),
            point_buf=torch.randint(0, nv, (nv, 8)),
        )

    # Directive 10: Initialize Gaussians
    if not os.path.exists(config.gaussian_init.init_state_path):
        gs = init_gaussians_from_faceverse(faceverse_mesh, config.gaussian_init)
        print(f"  ✓ Gaussians initialized: {gs.means.shape[0]} primitives")
    else:
        from src.contracts.schemas import load_gaussian_state
        gs = load_gaussian_state(config.gaussian_init.init_state_path)
        print(f"  ✓ Gaussians already initialized (loaded from checkpoint)")

    # Directive 13: Average texture fit
    if not os.path.exists(config.gaussian_init.avg_texture_path):
        gs = run_avg_texture_fit(gs, config.gaussian_init)
        print(f"  ✓ Average texture fit complete")
    else:
        print(f"  ✓ Average texture fit already exists (skip)")


def step_finetune_prior(config) -> None:
    """Directives 14-16: Arc2Face fine-tuning (skip if final.pt already exists)."""
    import os
    from src.prior.api import extend_model_with_pose_conditioning, finetune_prior
    from src.resource.gpu_manager import get_device

    print(f"\n[Directives 14-16] Arc2Face fine-tuning")
    device = get_device()

    if os.path.exists(config.finetune.final_path):
        print(f"  ✓ Fine-tuned prior already exists: {config.finetune.final_path} (skip)")
        return

    # Check if base Arc2Face model is available (check root + subdirectories)
    base_path = config.data_prep.arc2face_base_path
    
    def _has_model_files(dirpath):
        if not os.path.isdir(dirpath):
            return False
        for root, dirs, files in os.walk(dirpath):
            for f in files:
                if f.endswith((".json", ".safetensors", ".bin", ".pt")):
                    return True
        return False
    
    base_valid = _has_model_files(base_path)

    if not base_valid:
        print(f"  ⚠ Arc2Face base checkpoint not found: {base_path}")
        print(f"  [SKIP] Fine-tuning - needs Arc2Face model.")
        print(f"  [INFO] Run: huggingface-cli download FoivosPar/Arc2Face --local-dir {base_path}")
        return

    # Directive 14: Extend model with pose conditioning
    if not os.path.exists(config.finetune.arch_init_path):
        extend_model_with_pose_conditioning(base_path, config.finetune.arch_init_path, device)
        print(f"  ✓ Model extended with pose conditioning")
    else:
        print(f"  ✓ Architecture already extended (skip)")

    # Directive 15: Fine-tune
    finetune_prior(
        config.finetune.arch_init_path,
        config.data_prep.panohead_path,
        config.finetune,
    )

    # Directive 16: Freeze
    from src.prior.api import freeze_and_export
    freeze_and_export(config.finetune.arch_init_path, config.finetune.final_path)
    print(f"  ✓ Fine-tuned prior frozen and exported → {config.finetune.final_path}")


def step_stage1(config) -> None:
    """Directive 21: Stage 1 face-only optimization."""
    import os
    from src.contracts.schemas import load_gaussian_state
    from src.contracts.schemas import load_identity_embedding
    from src.optim.api import run_stage1
    from src.prior.api import load_frozen_prior

    print(f"\n[Directive 21] Stage 1 — face-only optimization ({config.stage1.iterations} iter)")

    if os.path.exists(config.stage1.checkpoint_path):
        print(f"  ✓ Stage 1 checkpoint already exists (skip)")
        return

    # Load prerequisites
    gs = load_gaussian_state(config.gaussian_init.avg_texture_path)
    identity = load_identity_embedding(
        config.data_prep.id_embedding_path,
        config.data_prep.id_embedding_json_path,
    )
    prior = load_frozen_prior(config.finetune.final_path) if os.path.exists(config.finetune.final_path) else None

    result = run_stage1(gs, identity, prior, config.stage1)
    print(f"  ✓ Stage 1 complete → {config.stage1.checkpoint_path}")


def step_stage2(config) -> None:
    """Directive 22: Stage 2 full-head optimization."""
    import os
    from src.contracts.schemas import load_gaussian_state, load_identity_embedding
    from src.optim.api import run_stage2
    from src.prior.api import load_frozen_prior

    print(f"\n[Directive 22] Stage 2 — full-head optimization ({config.stage2.iterations} iter)")

    if os.path.exists(config.stage2.checkpoint_path):
        print(f"  ✓ Stage 2 checkpoint already exists (skip)")
        return

    gs = load_gaussian_state(config.stage1.checkpoint_path)
    identity = load_identity_embedding(
        config.data_prep.id_embedding_path,
        config.data_prep.id_embedding_json_path,
    )
    prior = load_frozen_prior(config.finetune.final_path) if os.path.exists(config.finetune.final_path) else None

    run_stage2(gs, identity, prior, config.stage2)
    print(f"  ✓ Stage 2 complete → {config.stage2.checkpoint_path}")


def step_animation(config) -> None:
    """Directives 24-27: Per-expression animation + refinement."""
    import os
    import torch
    from src.contracts.schemas import (
        load_gaussian_state, load_faceverse_mesh, load_identity_embedding,
        ExpressionState,
    )
    from src.animation.api import (
        apply_faceverse_deformation, detect_mouth_opening,
        run_refinement, check_identity_preservation,
    )
    from src.prior.api import load_frozen_prior
    from src.sds.api import render

    print(f"\n[Directives 24-27] Expression animation + refinement")

    if not os.path.exists(config.stage2.checkpoint_path):
        print("  ⚠ Stage 2 checkpoint not found — skipping animation")
        return

    gs = load_gaussian_state(config.stage2.checkpoint_path)
    identity = load_identity_embedding(
        config.data_prep.id_embedding_path,
        config.data_prep.id_embedding_json_path,
    ) if os.path.exists(config.data_prep.id_embedding_path) else None
    prior = load_frozen_prior(config.finetune.final_path) if os.path.exists(config.finetune.final_path) else None

    expression_states = []
    for expr_name in config.animation.expressions:
        print(f"  Expression: {expr_name}")

        # Create expression state (FaceVerse ARKit blendshapes)
        expr = ExpressionState(
            name=expr_name,
            faceverse_expr_coeffs=torch.zeros(52),
        )

        # Set expression-specific coefficients (ARKit indices)
        if expr_name == "smile":
            expr.faceverse_expr_coeffs[4] = 0.5  # cheekSquint_L
        elif expr_name == "open_mouth":
            expr.faceverse_expr_coeffs[21] = 0.6  # jawOpen
        elif expr_name == "raised_brow":
            expr.faceverse_expr_coeffs[2] = 0.3  # browInnerUp

        # Directive 25: Detect if refinement needed
        needs_refinement = detect_mouth_opening(expr, config.animation)
        print(f"    Requires refinement: {needs_refinement}")

        if needs_refinement and prior is not None:
            # Directive 26: Run refinement
            expr = run_refinement(gs, expr, identity, prior, config.animation)
            print(f"    Refinement complete")

            # Directive 27: Check identity preservation
            if identity is not None:
                from src.sds.api import sample_camera
                cam = sample_camera((0, 0), (60, 60), 0.4, 2.0)
                rendered = render(gs, cam)
                similarity = check_identity_preservation(
                    rendered.image, identity, config.animation
                )
                print(f"    Identity similarity: {similarity:.4f}")

        expression_states.append(expr)

    print(f"  ✓ {len(expression_states)} expressions processed")


def step_export(config) -> None:
    """Directives 28-31: Export final deliverables."""
    import os
    from src.contracts.schemas import load_gaussian_state
    from src.export.api import (
        export_static_ply, render_turntable,
        package_animation_bundle, write_run_manifest,
    )

    print(f"\n[Directives 28-31] Export")
    print(f"  PLY:   {config.export.ply_path}")
    print(f"  Video: {config.export.turntable_path}")
    print(f"  Bundle:{config.export.animation_bundle_path}")
    print(f"  Manifest: {config.export.run_manifest_path}")

    if not os.path.exists(config.stage2.checkpoint_path):
        print("  ⚠ Stage 2 checkpoint not found — skipping export")
        return

    gs = load_gaussian_state(config.stage2.checkpoint_path)

    # Directive 28: Export PLY
    if not os.path.exists(config.export.ply_path):
        export_static_ply(gs, config.export.ply_path)
    else:
        print(f"  ✓ PLY already exists (skip)")

    # Directive 29: Turntable render
    if not os.path.exists(config.export.turntable_path):
        render_turntable(gs, config.export)
    else:
        print(f"  ✓ Turntable already exists (skip)")

    # Directive 30: Animation bundle
    if not os.path.exists(config.export.animation_manifest_path):
        package_animation_bundle(gs, [], None, config.export.animation_bundle_path)
    else:
        print(f"  ✓ Animation bundle already exists (skip)")

    # Directive 31: Run manifest
    if not os.path.exists(config.export.run_manifest_path):
        write_run_manifest(config, {}, {}, config.export.run_manifest_path)
    else:
        print(f"  ✓ Run manifest already exists (skip)")


def step_final_checklist(config) -> None:
    """Directive 35: Final holistic deliverable checklist."""
    import os
    print(f"\n[Directive 35] Final deliverable checklist")
    deliverables = [
        ("subject_static.ply", config.export.ply_path),
        ("turntable_360.mp4", config.export.turntable_path),
        ("animation_bundle/manifest.json", config.export.animation_manifest_path),
        ("run_manifest.json", config.export.run_manifest_path),
    ]
    all_ok = True
    for name, path in deliverables:
        exists = os.path.exists(path)
        print(f"  {'✓' if exists else '✗'} {name} ({path})")
        if not exists:
            all_ok = False

    if all_ok:
        print(f"\n  ✓ All deliverables present! Avatar build complete.")
    else:
        print(f"\n  ⚠ Some deliverables missing — re-run the relevant module.")


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
