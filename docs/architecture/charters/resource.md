# Subsystem: RESOURCE
# Charter — Directive 0.2

## Ownership
Owns: src/resource/gpu_manager.py (Directive 49-51), @managed_stage decorator, resource_budget config parsing, single device-init call site

## Does NOT own
Does NOT own: the optimizer or scheduler construction (TRAINFW owns these); logging or validation logic (LOG/VALID own these)

## Public API
Public API: src/resource/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: ENV
Depended on by: ORCH, TEST
