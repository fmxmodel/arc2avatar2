# Subsystem: LOG
# Charter — Directive 0.2

## Ownership
Owns: src/logging/framework.py (get_logger), logs/structured/<run_id>.jsonl, logs/artifacts/<run_id>/manifest.json, archive_artifact()

## Does NOT own
Does NOT own: the validation criteria or acceptance thresholds (VALID owns these); the error recovery policy (ERROR owns this)

## Public API
Public API: src/logging/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONFIG
Depended on by: EXPERIMENT, ERROR, TEST
