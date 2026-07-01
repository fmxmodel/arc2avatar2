"""
Tests for Module J — Configuration Schema
Directive 65: One test per documented Exception, one test per output shape/type.
"""

import unittest


class TestConfigSchema(unittest.TestCase):
    """Test configuration schema and validation."""

    def test_stage1_config_defaults(self):
        """Test Stage1Config defaults match Part 1 reference values."""
        from src.config.schema import Stage1Config
        cfg = Stage1Config()
        self.assertEqual(cfg.iterations, 500)
        self.assertEqual(cfg.fov_radians, 0.4)
        self.assertEqual(cfg.azimuth_range_deg, (-110.0, 110.0))
        self.assertEqual(cfg.pitch_range_deg, (60.0, 90.0))
        self.assertEqual(cfg.guidance_scale, 7.0)

    def test_stage2_config_defaults(self):
        """Test Stage2Config defaults."""
        from src.config.schema import Stage2Config
        cfg = Stage2Config()
        self.assertEqual(cfg.azimuth_range_deg, (-180.0, 180.0))
        self.assertEqual(cfg.pitch_range_deg, (30.0, 120.0))

    def test_validate_config_invalid_angle_range(self):
        """Test validation catches invalid angle range (min >= max)."""
        from src.config.schema import PipelineConfig, validate_config

        cfg = PipelineConfig()
        cfg.stage1.azimuth_range_deg = (110.0, -110.0)  # Invalid
        errors = validate_config(cfg)
        self.assertTrue(any("azimuth" in e.field_path for e in errors))

    def test_validate_config_negative_iterations(self):
        """Test validation catches non-positive iteration count."""
        from src.config.schema import PipelineConfig, validate_config

        cfg = PipelineConfig()
        cfg.stage1.iterations = -1
        errors = validate_config(cfg)
        self.assertTrue(any("iterations" in e.field_path for e in errors))

    def test_resolve_config_fast_debug(self):
        """Test experiment override works (fast_debug: 500 Stage-1 iter)."""
        from src.config.schema import resolve_config
        cfg = resolve_config(experiment="fast_debug")
        self.assertEqual(cfg.stage1.iterations, 500)

    def test_resolve_config_cli_override(self):
        """Test CLI override takes precedence."""
        from src.config.schema import resolve_config
        cfg = resolve_config(cli_args={"stage1": {"iterations": 100}})
        self.assertEqual(cfg.stage1.iterations, 100)
