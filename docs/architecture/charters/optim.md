# Subsystem: OPT
# Charter — Directive 0.2

## Ownership
Owns: Stage-1 (Directive 21) and Stage-2 (Directive 22) optimization loops, divergence guard (Directive 23), checkpoint saving for stage1_face.pt and stage2_full_head.pt

## Does NOT own
Does NOT own: the renderer or camera sampler (SDS owns these); the connectivity regularizer implementation (GAUSS owns this in _sds_internals.py); animation blendshape application (ANIM owns this)

## Public API
Public API: src/optim/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: GAUSS, SDS, CONTRACTS, TRAINFW, VALID
Depended on by: ANIM, TEST
