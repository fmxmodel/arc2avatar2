# Subsystem: ENV
# Charter — Directive 0.2

## Ownership
Owns: interpreter, CUDA, package versions in configs/requirements_lock.txt, src/utils/env_check.py

## Does NOT own
Does NOT own: package version selection (the lock file reflects choices made by the system designer, not ENV); GPU device selection (RESOURCE owns this); random seed initialization (TRAINFW owns this through seed.py)

## Public API
Public API: src/env/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: (nothing)
Depended on by: DATA, CONFIG, RESOURCE, TEST
