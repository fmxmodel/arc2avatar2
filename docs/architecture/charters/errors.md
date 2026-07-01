# Subsystem: ERROR
# Charter — Directive 0.2

## Ownership
Owns: src/errors/hierarchy.py (Arc2AvatarError exception tree Directive 62), recovery policy dispatch (Directive 63), three-field diagnostics (Directive 64)

## Does NOT own
Does NOT own: the logging infrastructure (LOG owns this); the state machine transitions (STATE owns this)

## Public API
Public API: src/errors/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: LOG
Depended on by: TEST
