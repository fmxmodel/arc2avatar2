# Subsystem: DOCS
# Charter — Directive 0.2

## Ownership
Owns: docs/generated/api/ (Directive 80 CI sphinx-apidoc), dependency + data-flow diagrams (Directive 81), docs/onboarding/*.md (Directive 82)

## Does NOT own
Does NOT own: the actual API implementation that the docs describe (each subsystem owns its own code); the extension point definitions (EXT owns the ABCs and registry)

## Public API
Public API: src/docs/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONTRACTS
Depended on by: INTEGRATE
