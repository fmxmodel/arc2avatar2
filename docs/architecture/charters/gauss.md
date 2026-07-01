# Subsystem: GAUSS
# Charter — Directive 0.2

## Ownership
Owns: checkpoints/gaussians/*.pt, the vertex_id correspondence tensor, the connectivity-regularizer loss function (src/optim/_sds_internals.py), checkpoint init + avg-texture fit

## Does NOT own
Does NOT own: camera sampling (SDS owns this); the optimizer that consumes GAUSS parameters (OPT owns this); FLAME blendshape application at inference/animation time (ANIM owns this, even though it reads GAUSS vertex_id tensor)

## Public API
Public API: src/gauss/api.py

## CRUD Ownership Table (Directive 41)

| Object | Created by | Modified by | Read by | Destroyed by |
|---|---|---|---|---|
| `GaussianState` | GAUSS (Directive 10) | OPT (Stage 1/2), ANIM (blendshape displacement) | SDS, EXPORT | never explicitly; superseded by newer checkpoint versions |

## Dependency Graph Reference (Directive 0.3)
Depends on: DATA, CONTRACTS
Depended on by: SDS, OPT, TEST
