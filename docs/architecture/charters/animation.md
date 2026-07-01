# Subsystem: ANIM
# Charter — Directive 0.2

## Ownership
Owns: src/animation/ (blendshape-to-Gaussian displacement), expression refinement (Directive 26), identity-preservation gate (Directive 27), expr_<name>_refined.pt files

## Does NOT own
Does NOT own: the core FLAME model or its topology (DATA loaded it; GAUSS uses it); the SDS gradient formula (SDS owns this); final export format decisions (EXPORT owns these)

## Public API
Public API: src/animation/api.py

## CRUD Ownership Table (Directive 41)

| Object | Created by | Modified by | Read by | Destroyed by |
|---|---|---|---|---|
| `ExpressionState` | ANIM (Directive 24) | ANIM (Directive 26, refinement patch only) | EXPORT | never; archived per Directive 58 |

## Dependency Graph Reference (Directive 0.3)
Depends on: OPT, CONTRACTS, VALID
Depended on by: EXPORT, TEST
