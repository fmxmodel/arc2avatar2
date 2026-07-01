# Subsystem: CONTRACTS
# Charter — Directive 0.2

## Ownership
Owns: src/contracts/schemas.py (GaussianState, FlameMesh, IdentityEmbedding, CameraSample, RenderResult, OptimizerState, ExpressionState, RunManifest), save/load helpers, format-binding table

## Does NOT own
Does NOT own: any business logic beyond type definitions and serialization; the state machine or checkpoint lifecycle (STATE owns these); the error hierarchy definitions (ERROR owns these)

## Public API
Public API: src/contracts/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: (cross-cutting; imported by all)
Depended on by: GAUSS, SDS, OPT, ANIM, EXPORT, VALID, EXT, DOCS, DATA, PRIOR, TEST
