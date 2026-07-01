# Subsystem: PRIOR
# Charter — Directive 0.2

## Ownership
Owns: checkpoints/arc2face_base/ (download presence), checkpoints/arc2face_finetuned/{arch_init.pt,final.pt} (architecture extension + fine-tuning + freeze)

## Does NOT own
Does NOT own: the SDS gradient formula or renderer (SDS owns these); the optimizer that updates Gaussians (OPT owns this); animation or export logic (ANIM/EXPORT own these)

## Public API
Public API: src/prior/api.py

## Dependency Graph Reference (Directive 0.3)
Depends on: DATA, CONTRACTS, TRAINFW
Depended on by: SDS, TEST
