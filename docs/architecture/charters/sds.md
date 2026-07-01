# Subsystem: SDS
# Charter — Directive 0.2

## Ownership
Owns: src/render/ (differentiable renderer), camera sampler, src/optim/api.py (SDS gradient step function), guidance scale parameter

## Does NOT own
Does NOT own: the optimization loop that calls SDS repeatedly (OPT owns this); checkpoint management (STATE owns this); animation or export (ANIM/EXPORT own these)

## Public API
Public API: src/sds/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: GAUSS, PRIOR, CONTRACTS, TRAINFW
Depended on by: OPT, TEST
