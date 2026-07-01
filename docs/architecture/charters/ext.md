# Subsystem: EXT
# Charter — Directive 0.2

## Ownership
Owns: Four ABCs: IdentityEncoder, Renderer, MeshModel, AnimationDriver (Directive 77), src/registry/registry.py (Directive 78), docs/architecture/extension_points.md (Directive 79)

## Does NOT own
Does NOT own: the default implementations of the ABCs (each default is registered as a plugin, not special-cased in ORCH/OPT/SDS); the onboarding documentation content beyond the extension contract (DOCS owns overall docs)

## Public API
Public API: src/ext/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: CONTRACTS
Depended on by: INTEGRATE
