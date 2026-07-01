# Subsystem: TRAINFW
# Charter — Directive 0.2

## Ownership
Owns: src/trainfw/factory.py (build_optimizer, build_scheduler), src/trainfw/grad_utils.py (clip, zero_grad, gradient accumulation, autocast), src/utils/seed.py

## Does NOT own
Does NOT own: the specific loss functions used in SDS or fine-tuning (SDS/PRIOR own these); checkpoint management (STATE owns this)

## Public API
Public API: src/trainfw/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONFIG
Depended on by: OPT, PRIOR, SDS, ANIM, TEST
