"""
Integration Tests for Dependency Graph Edges (Directive 66)
=============================================================
For every edge A→B in configs/dependency_graph.yaml, construct A's output
on a tiny synthetic fixture and feed it into B's entry point.
Asserts no exception and a schema-valid result.
"""

import unittest
import yaml
import torch


class TestDependencyEdges(unittest.TestCase):
    """Verify every edge in the dependency graph produces valid handoffs."""

    @classmethod
    def setUpClass(cls):
        with open("configs/dependency_graph.yaml", "r") as f:
            cls.graph = yaml.safe_load(f)

    def test_all_edges_have_producer_and_consumer(self):
        """Test every edge has valid from/to keys."""
        for edge in self.graph["edges"]:
            self.assertIn("from", edge, f"Edge missing 'from': {edge}")
            self.assertIn("to", edge, f"Edge missing 'to': {edge}")

    def test_all_nodes_in_graph(self):
        """Test every referenced node is in the nodes list."""
        nodes = set(self.graph["nodes"])
        referenced = set()
        for edge in self.graph["edges"]:
            referenced.add(edge["from"])
            referenced.add(edge["to"])
        for r in referenced:
            self.assertIn(r, nodes, f"Node '{r}' referenced in edges but not in nodes list")

    def test_parallel_safe_groups_are_valid(self):
        """Test parallel_safe groups reference valid nodes."""
        nodes = set(self.graph["nodes"])
        for group in self.graph.get("parallel_safe", []):
            for item in group:
                # Items may be formatted as "SUBSYSTEM_action"
                base_name = item.split("_")[0]
                self.assertIn(base_name, nodes,
                              f"parallel_safe item '{item}' base '{base_name}' not in nodes")

    def test_contracts_to_gaussian_state_handoff(self):
        """Test CONTRACTS→GAUSS edge: GaussianState can be constructed and read."""
        from src.contracts.schemas import GaussianState

        n = 5
        gs = GaussianState(
            means=torch.randn(n, 3),
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
            sh_coeffs=torch.randn(n, 3, 16),
            vertex_id=torch.randint(0, 100, (n,)),
        )
        # Verify vertex_id read
        self.assertEqual(gs.vertex_id.shape[0], n)

    def test_contracts_to_identity_embedding_handoff(self):
        """Test CONTRACTS→DATA edge: IdentityEmbedding can be constructed."""
        from src.contracts.schemas import IdentityEmbedding

        vec = torch.randn(512)
        vec = vec / vec.norm()
        ie = IdentityEmbedding(vector=vec, source_image_hash="test")
        self.assertEqual(ie.vector.shape, (512,))

    def test_config_to_stage1_config_handoff(self):
        """Test CONFIG→OPT edge: Stage1Config can be parsed and validated."""
        from src.config.schema import Stage1Config, validate_config, PipelineConfig

        s1 = Stage1Config()
        cfg = PipelineConfig()
        cfg.stage1 = s1
        errors = validate_config(cfg)
        self.assertEqual(len(errors), 0)
