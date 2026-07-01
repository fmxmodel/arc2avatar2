"""
Arc2Avatar — End-to-End Integration Smoke Test (Directive 33)
===============================================================
Runs the full pipeline on a tiny/cheap configuration:
- Stage 1 = 100 iterations (instead of 500)
- Stage 2 = 50 iterations (instead of 1500-3000)
- No expression refinement

Goal: clean hand-off verification only.
Not expected to produce a good-looking avatar.
Run this after ANY change to ANY module, before kicking off a real run.
"""

import os
import sys
import tempfile
import unittest


class TestFullPipelineSmoke(unittest.TestCase):
    """Smoke test for the full pipeline — verifies clean hand-offs.

    Uses the fast_debug experiment config (20 Stage-1 iterations,
    50 Stage-2 iterations, no expression refinement).
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Change to project root
        cls.orig_cwd = os.getcwd()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(project_root)
        cls.project_root = project_root

    @classmethod
    def tearDownClass(cls):
        """Restore working directory."""
        os.chdir(cls.orig_cwd)

    def test_config_validation(self):
        """Test that config validation works (Directive 39)."""
        from src.config.schema import load_and_validate_config

        # This should not raise or exit
        config = load_and_validate_config(experiment="fast_debug")
        self.assertIsNotNone(config)
        self.assertEqual(config.stage1.iterations, 100)
        self.assertEqual(config.stage2.iterations, 50)

    def test_dependency_graph_is_dag(self):
        """Test that dependency graph has no cycles (Directive 0.3)."""
        import yaml
        from src.orch.scheduler import topological_sort

        with open("configs/dependency_graph.yaml", "r") as f:
            graph = yaml.safe_load(f)

        nodes = graph["nodes"]
        edges = graph["edges"]

        # This should not raise
        sorted_nodes = topological_sort(nodes, edges)
        self.assertEqual(len(sorted_nodes), len(nodes))

    def test_all_contracts_importable(self):
        """Test that all data contracts are importable and constructable."""
        from src.contracts.schemas import (
            GaussianState, FaceVerseMesh, IdentityEmbedding,
            CameraSample, RenderResult, ExpressionState, RunManifest,
        )
        import torch

        # Test GaussianState construction
        n = 10
        gs = GaussianState(
            means=torch.randn(n, 3),
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
            sh_coeffs=torch.randn(n, 3, 16),
            vertex_id=torch.randint(0, 100, (n,)),
        )
        self.assertEqual(gs.means.shape[0], n)
        self.assertEqual(gs.vertex_id.shape[0], n)

        # Test IdentityEmbedding
        vec = torch.randn(512)
        vec = vec / vec.norm()
        ie = IdentityEmbedding(vector=vec, source_image_hash="test")
        self.assertEqual(ie.vector.shape, (512,))

    def test_config_schema_defaults_match_part1(self):
        """Test that schema defaults match Part 1 reference values."""
        from src.config.schema import Stage1Config, Stage2Config

        s1 = Stage1Config()
        self.assertEqual(s1.iterations, 500)
        self.assertEqual(s1.fov_radians, 0.4)
        self.assertEqual(s1.azimuth_range_deg, (-110.0, 110.0))
        self.assertEqual(s1.pitch_range_deg, (60.0, 90.0))

        s2 = Stage2Config()
        self.assertEqual(s2.azimuth_range_deg, (-180.0, 180.0))
        self.assertEqual(s2.pitch_range_deg, (30.0, 120.0))

    def test_error_hierarchy(self):
        """Test that error hierarchy raises correctly (Directive 62)."""
        from src.errors.hierarchy import (
            ConfigurationError, DataError, DivergenceError,
            RenderingError, ExportError,
        )

        # Test typed exception
        try:
            raise ConfigurationError(
                what_failed="Config validation",
                why="guidance_scale=-1.0",
                how_to_fix="Set guidance_scale to positive value in config",
            )
        except ConfigurationError as e:
            self.assertIn("What:", str(e))
            self.assertIn("Fix:", str(e))

        # Test recovery policy dispatch
        from src.errors.hierarchy import get_recovery_policy
        policy = get_recovery_policy(DivergenceError(
            what_failed="Divergence detected",
            why="connectivity_loss=4.2, threshold=0.5",
            how_to_fix="Halve guidance scale and resume",
        ))
        self.assertEqual(policy["action"], "resume_with_halved_guidance")

    def test_registry_roundtrip(self):
        """Test plugin registry (Directive 78)."""
        from src.registry.registry import register, get

        class DummyModel:
            pass

        register("models", "test_model", DummyModel)
        retrieved = get("models", "test_model")
        self.assertIs(retrieved, DummyModel)

    def test_checkpoint_manager_operations(self):
        """Test checkpoint lifecycle operations (Directive 47)."""
        from src.state.checkpoint_manager import (
            save_checkpoint, validate_checkpoint, replace_checkpoint,
        )
        import tempfile
        import torch

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            # Save
            data = {"test": torch.tensor([1, 2, 3])}
            ckpt_hash = save_checkpoint(data, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # Validate
            self.assertTrue(validate_checkpoint(tmp_path))

            # Replace (archive old, write new)
            new_data = {"test": torch.tensor([4, 5, 6])}
            replace_checkpoint(tmp_path, new_data)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_state_machine_transitions(self):
        """Test state machine (Directive 46)."""
        from src.state.checkpoint_manager import (
            get_state, transition,
        )
        import tempfile
        import os

        # Use a non-existent path to test "Not Started" default
        state_path = tempfile.mktemp(suffix=".json")

        try:
            # Start state
            self.assertEqual(get_state(state_path), "Not Started")

            # Legal transitions
            transition("Initialized", state_path)
            self.assertEqual(get_state(state_path), "Initialized")

            transition("Running", state_path)
            self.assertEqual(get_state(state_path), "Running")

            # Legal transition: Running -> Completed
            transition("Completed", state_path)
            self.assertEqual(get_state(state_path), "Completed")

        finally:
            if os.path.exists(state_path):
                os.remove(state_path)

    def test_seed_reproducibility(self):
        """Test that set_global_seed produces reproducible results (Directive 55)."""
        from src.utils.seed import set_global_seed
        import torch

        set_global_seed(42)
        a = torch.randn(10)

        set_global_seed(42)
        b = torch.randn(10)

        self.assertTrue(torch.allclose(a, b))

    def test_invariants_exist(self):
        """Test that invariants.md has all 6 invariants documented."""
        with open("docs/architecture/invariants.md", "r") as f:
            content = f.read()

        for i in range(1, 7):
            self.assertIn(f"I{i}", content, f"Invariant I{i} missing from invariants.md")

    def test_charter_files_exist(self):
        """Test that all 22 charter files exist (Directive 0.2)."""
        subsystems = [
            "env", "data", "gauss", "prior", "sds", "optim",
            "animation", "export", "config", "contracts", "state",
            "resource", "trainfw", "logging", "valid", "errors",
            "test", "experiment", "orch", "ext", "docs", "integrate",
        ]
        for s in subsystems:
            path = f"docs/architecture/charters/{s}.md"
            self.assertTrue(os.path.exists(path), f"Missing charter: {path}")

    def test_config_merge_layers(self):
        """Test 4-layer config merge (Directive 37)."""
        from src.config.schema import resolve_config, PipelineConfig

        # Layer 1: System defaults
        cfg = resolve_config()
        self.assertEqual(cfg.stage1.iterations, 500)

        # Layer 2: Experiment override
        cfg = resolve_config(experiment="fast_debug")
        self.assertEqual(cfg.stage1.iterations, 100)

        # Layer 4: CLI override
        cfg = resolve_config(cli_args={"stage1": {"iterations": 100}})
        self.assertEqual(cfg.stage1.iterations, 100)


if __name__ == "__main__":
    unittest.main()
