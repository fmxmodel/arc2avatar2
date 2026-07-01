# Subsystem: EXPERIMENT
# Charter — Directive 0.2

## Ownership
Owns: logs/experiment_registry.jsonl (Directive 69), dataset versioning (Directive 70), checkpoint lineage (Directive 71), resolved_config metadata (Directive 72)

## Does NOT own
Does NOT own: the actual scheduling logic (ORCH owns this); the configuration resolution (CONFIG owns this)

## Public API
Public API: src/experiment/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: LOG, STATE
Depended on by: TEST, INTEGRATE
