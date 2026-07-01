# Adding a Model — Onboarding Checklist
> Directive 82: Numbered, copy-pasteable checklist.

## Steps

1. **Implement the model class**
   - Must implement the relevant ABC from `src/ext/api.py`:
     - `IdentityEncoder` for face recognition models.
     - `Renderer` for differentiable renderers.
     - `MeshModel` for mesh/topology models.
     - `AnimationDriver` for animation/deformation models.

2. **Register in registry (Directive 78)**
   ```python
   from src.registry.registry import register
   register("models", "my_model_name", MyModelClass)
   ```

3. **Verify contract compliance (Directive 79)**
   - Input/output schemas must match Directive 40 types.
   - Must preserve applicable invariants from Directive 0.4.

4. **Add configuration (Directive 36)**
   - Add model selection field to relevant config (e.g., `identity_encoder: "my_model_name"`).
   - Config references model by registry string name, never by import path.

5. **Add checkpoint handling (Directive 47)**
   - Save/load via `src/state/checkpoint_manager.py`.
   - Must use atomic temp-write-then-rename pattern.

6. **Unit tests (Directive 65)**
   - Test output shape/type matches expected schema.
   - Test invariant preservation.

7. **Integration tests (Directive 66)**
   - Test as part of the full pipeline edge the model belongs to.

8. **Document in extension_points.md (Directive 79)**
   - Add a worked example of registering a new model.
