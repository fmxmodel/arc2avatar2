# Subsystem: EXPORT
# Charter — Directive 0.2

## Ownership
Owns: outputs/final_avatar/subject_static.ply, outputs/renders/turntable_360.mp4, outputs/final_avatar/animation_bundle/ + manifest, outputs/final_avatar/run_manifest.json

## Does NOT own
Does NOT own: the animation blendshape logic (ANIM owns this); the optimization state (OPT owns this); checkpoint versioning decisions (STATE owns this)

## Public API
Public API: src/export/api.py

## CRUD Ownership Table (Directive 41)

| Object | Created by | Modified by | Read by | Destroyed by |
|---|---|---|---|---|
| `RunManifest` | EXPORT (Directive 31) | never — write-once (enforced: file existence check prevents overwrite) | INTEGRATE (Directive 86) | never; permanent record |

## Dependency Graph Reference (Directive 0.3)
Depends on: ANIM, CONTRACTS, VALID
Depended on by: TEST, INTEGRATE
