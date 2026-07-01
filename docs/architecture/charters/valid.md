# Subsystem: VALID
# Charter — Directive 0.2

## Ownership
Owns: validate_<subsystem>_output() functions (Directive 59-61), base sanity checker (NaN/Inf, zero-length, missing files, mesh validity, embedding norm), acceptance criteria (Directive 60 table)

## Does NOT own
Does NOT own: the actual error recovery or exception raising (ERROR owns this); the module compute logic (each module owns its own computation)

## Public API
Public API: src/valid/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONTRACTS
Depended on by: OPT, ANIM, EXPORT, TEST
