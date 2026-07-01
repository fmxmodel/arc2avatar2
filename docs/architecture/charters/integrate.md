# Subsystem: INTEGRATE
# Charter — Directive 0.2

## Ownership
Owns: End-to-end dependency audit (Directive 83), data-flow audit (Directive 84), reproducibility audit (Directive 85), outputs/final_avatar/run_report.json with SHA-256 signature (Directive 86)

## Does NOT own
Does NOT own: any individual module computation (each subsystem owns its own computation); the run manifest or state file (EXPORT/STATE own these)

## Public API
Public API: src/integrate/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: all subsystems
Depended on by: (nothing — terminal audit node)
