"""
Tests for Module K — Data Contracts
Directive 65: Verify output shape/type against Directive 40 schemas.
"""

import unittest
import torch


class TestGaussianState(unittest.TestCase):
    """Test GaussianState schema (Directive 40)."""

    def setUp(self):
        self.n = 10
        self.k = 16  # (max_sh_degree=3+1)^2 = 16

    def test_valid_construction(self):
        """Test GaussianState constructs correctly with valid shapes."""
        from src.contracts.schemas import GaussianState

        gs = GaussianState(
            means=torch.randn(self.n, 3),
            scales=torch.randn(self.n, 3),
            rotations=torch.randn(self.n, 4),
            opacities=torch.randn(self.n, 1),
            sh_coeffs=torch.randn(self.n, 3, self.k),
            vertex_id=torch.randint(0, 100, (self.n,)),
        )
        self.assertEqual(gs.means.shape, (self.n, 3))
        self.assertEqual(gs.vertex_id.shape, (self.n,))

    def test_i2_invariant_enforced(self):
        """Test Invariant I2: vertex_id length must match means length."""
        from src.contracts.schemas import GaussianState

        with self.assertRaises(AssertionError):
            GaussianState(
                means=torch.randn(self.n, 3),
                scales=torch.randn(self.n, 3),
                rotations=torch.randn(self.n, 4),
                opacities=torch.randn(self.n, 1),
                sh_coeffs=torch.randn(self.n, 3, self.k),
                vertex_id=torch.randint(0, 100, (self.n + 1,)),  # Mismatch
            )

    def test_wrong_means_dim(self):
        """Test that wrong means dimension is caught."""
        from src.contracts.schemas import GaussianState

        with self.assertRaises(AssertionError):
            GaussianState(
                means=torch.randn(self.n, 4),  # Wrong: should be 3
                scales=torch.randn(self.n, 3),
                rotations=torch.randn(self.n, 4),
                opacities=torch.randn(self.n, 1),
                sh_coeffs=torch.randn(self.n, 3, self.k),
                vertex_id=torch.randint(0, 100, (self.n,)),
            )


class TestIdentityEmbedding(unittest.TestCase):
    """Test IdentityEmbedding schema (Directive 40)."""

    def test_valid_construction(self):
        """Test IdentityEmbedding with valid L2-normalized vector."""
        from src.contracts.schemas import IdentityEmbedding

        vec = torch.randn(512)
        vec = vec / vec.norm()
        ie = IdentityEmbedding(vector=vec, source_image_hash="abc123")
        self.assertEqual(ie.vector.shape, (512,))

    def test_i3_invariant_enforced(self):
        """Test Invariant I3: vector must be 512-d."""
        from src.contracts.schemas import IdentityEmbedding

        with self.assertRaises(AssertionError):
            vec = torch.randn(256)  # Wrong dimensionality
            IdentityEmbedding(vector=vec, source_image_hash="abc123")

    def test_l2_normalization_enforced(self):
        """Test that non-unit-norm vector is rejected."""
        from src.contracts.schemas import IdentityEmbedding

        with self.assertRaises(AssertionError):
            vec = torch.randn(512) * 10  # Not L2-normalized
            IdentityEmbedding(vector=vec, source_image_hash="abc123")


class TestFaceVerseMesh(unittest.TestCase):
    """Test FaceVerseMesh schema (Directive 40)."""

    def test_valid_construction(self):
        """Test FaceVerseMesh with valid shapes."""
        from src.contracts.schemas import FaceVerseMesh

        nv = 6335
        mesh = FaceVerseMesh(
            V=torch.randn(nv, 3),
            F=torch.randint(0, nv, (12566, 3)),
            idBase=torch.randn(nv*3, 150),
            expBase=torch.randn(nv*3, 52),
            texBase=torch.randn(nv*3, 251),
            meanshape=torch.randn(nv*3),
            meantex=torch.randn(nv*3),
            point_buf=torch.randint(0, nv, (nv, 8)),
        )
        self.assertEqual(mesh.V.shape, (nv, 3))


class TestSerialization(unittest.TestCase):
    """Test serialization helpers (Directive 42)."""

    def test_save_versioned_roundtrip(self):
        """Test save_versioned / load_versioned roundtrip."""
        import tempfile
        import os
        from src.contracts.schemas import save_versioned, load_versioned

        data = {"key": "value", "num": 42}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_versioned(data, path)
            loaded = load_versioned(path)
            self.assertEqual(loaded["key"], "value")
            self.assertEqual(loaded["num"], 42)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_schema_version_mismatch_caught(self):
        """Test Invariant I4: version mismatch fails loudly."""
        import tempfile
        import os
        import torch
        from src.contracts.schemas import load_versioned

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            # Save with version 99
            torch.save({"schema_version": 99, "data": {}}, path)
            with self.assertRaises(ValueError):
                load_versioned(path, expected_version=1)
        finally:
            if os.path.exists(path):
                os.remove(path)
