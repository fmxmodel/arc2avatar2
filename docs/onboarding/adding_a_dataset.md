# Adding a Dataset — Onboarding Checklist
> Directive 82: Numbered, copy-pasteable checklist.

## Steps

1. **Place the dataset**
   - In `data/<dataset_name>/` following the project's data directory convention.
   - Never mutate a dataset directory in place once versioned (Directive 70).

2. **Create dataset manifest (Directive 70)**
   - Compute content hash manifest for the directory.
   - Store in `data/<dataset_name>/dataset_manifest.json`.

3. **Register in registry (Directive 78)**
   ```python
   from src.registry.registry import register
   register("datasets", "my_dataset", MyDatasetClass)
   ```

4. **Add data loading logic**
   - Add to `src/data/api.py` or a helper in `src/data/`.
   - Must return typed schemas from `src/contracts/schemas.py`.

5. **Add configuration (Directive 36)**
   - Add dataset path and version fields to `DataPrepConfig`.

6. **Add version check (Directive 70)**
   - Verify on-disk content hash matches configured version before proceeding.

7. **Verification threshold (Directive 9)**
   - Add minimum-identity and minimum-angle thresholds if applicable.
   - Log warning (don't silently proceed) if under threshold.

8. **Unit / integration tests (Directives 65-66)**
   - Test data loading produces correct schema types.
   - Test hash mismatch detection.
