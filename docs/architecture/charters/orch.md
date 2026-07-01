# Subsystem: ORCH
# Charter — Directive 0.2

## Ownership
Owns: src/orch/scheduler.py (Directive 73-76: --directive/--module/--full/--custom modes, dependency resolution, parallel execution, progress reporting), src/run_pipeline.py (Directive 32)

## Does NOT own
Does NOT own: the actual module computation (each module owns its own computation); the error hierarchy definitions (ERROR owns these)

## Public API
Public API: src/orch/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: STATE, RESOURCE
Depended on by: TEST, INTEGRATE
