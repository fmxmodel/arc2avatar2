# Subsystem: TEST
# Charter — Directive 0.2

## Ownership
Owns: tests/unit/, tests/integration/, tests/regression/test_identity_preservation.py, tests/benchmarks/, CI lint rules

## Does NOT own
Does NOT own: the actual pipeline logic (every other subsystem owns its own code); the CI/CD infrastructure configuration

## Public API
Public API: src/test/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: all subsystems
Depended on by: INTEGRATE
