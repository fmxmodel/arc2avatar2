# Subsystem: STATE
# Charter — Directive 0.2

## Ownership
Owns: run_state.json pipeline state machine (Directive 46), src/state/checkpoint_manager.py (Directives 47-48)

## Does NOT own
Does NOT own: the actual pipeline computation logic (all other subsystems own their respective computations); GPU resource management (RESOURCE owns this)

## Public API
Public API: src/state/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONFIG
Depended on by: EXPERIMENT, ORCH, TEST
