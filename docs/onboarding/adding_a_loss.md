# Adding a Loss — Onboarding Checklist
> Directive 82: Numbered, copy-pasteable checklist.

## Steps

1. **Implement the loss function**
   - Create in the appropriate subsystem's internal module (e.g., `src/optim/_losses.py`).
   - Must accept `torch.Tensor` inputs and return a scalar `torch.Tensor`.

2. **Register in registry (Directive 78)**
   ```python
   from src.registry.registry import register
   register("losses", "my_loss_name", MyLossFunction)
   ```

3. **Add configuration (Directive 36)**
   - Add loss-specific fields to the relevant config dataclass in `src/config/schema.py`.

4. **Wire into training loop**
   - Import via registry: `loss_fn = get("losses", config.loss_name)`.
   - Add loss term to the appropriate training loop (Directives 21, 22, or 26).

5. **Unit test (Directive 65)**
   - Test with known inputs: verify gradient flow and expected value ranges.
   - Test edge cases: zero inputs, NaN propagation.

6. **Integration test (Directive 66)**
   - Verify the loss integrates correctly in the training loop without breaking gradient shapes.
