# Adding a Module — Onboarding Checklist
> Directive 82: Numbered, copy-pasteable checklist.

## Steps

1. **Register in registry (Directive 78)**
   - Add your module to `src/registry/registry.py`'s `_REGISTRY` if it falls under one of the five kinds: `datasets`, `models`, `losses`, `renderers`, `optimizers`.

2. **Create the charter (Directive 0.2)**
   - Create `docs/architecture/charters/<subsystem>.md` with all four template fields.
   - The `Does NOT own` field is mandatory and non-empty.

3. **Add dependency graph edges (Directive 0.3)**
   - Add your module to `configs/dependency_graph.yaml`'s `nodes` list.
   - Add all edges from your module to its dependencies, and from dependents to your module.

4. **Create api.py (Directive 43)**
   - Create `src/<subsystem>/api.py` with 4-section docstrings (Inputs, Outputs, Exceptions, Side effects).
   - Prefix internal helpers with `_`.

5. **Add data contracts if needed (Directive 40)**
   - Add any new schema types to `src/contracts/schemas.py`.
   - Add `__post_init__` assertions for shape/dtype invariants.

6. **Add configuration if needed (Directive 36)**
   - Add a nested `@dataclass` to `src/config/schema.py`.
   - Add corresponding YAML defaults to `configs/pipeline_config.yaml`.

7. **Write unit tests (Directive 65)**
   - Create `tests/unit/test_<subsystem>_api.py`.
   - One test per documented Exception.
   - One test verifying output shape/type against Directive 40 schema.

8. **Write integration tests (Directive 66)**
   - For every edge from your module in the dependency graph, add a test in `tests/integration/`.

9. **Add acceptance criteria (Directive 60)**
   - Add your module's mechanical acceptance check to the VALID subsystem.

10. **Add CRUD ownership (Directive 41)**
    - Add ownership rows to your charter for any new schema types.
