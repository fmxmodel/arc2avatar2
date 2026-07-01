# Subsystem: CONFIG
# Charter — Directive 0.2

## Ownership
Owns: src/config/schema.py (all dataclass definitions), configs/pipeline_config.yaml, configs/experiments/*.yaml, configs/project_overrides.yaml, resolve_config(), validate_config()

## Does NOT own
Does NOT own: environment variable or hardware discovery (ENV owns these); runtime state transitions (STATE owns these); experiment registry (EXPERIMENT owns this)

## Public API
Public API: src/config/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: ENV
Depended on by: STATE, TRAINFW, LOG, TEST
