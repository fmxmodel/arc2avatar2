# Subsystem: DATA
# Charter — Directive 0.2

## Ownership
Owns: data/raw_input/subject.png (validator), data/embeddings/subject_id_embedding.npy+.json, data/flame_template/flame_loaded.pt, data/panohead_synth/ (presence check + manifest)

## Does NOT own
Does NOT own: Gaussian scaffold creation (GAUSS owns this); Arc2Face model fine-tuning (PRIOR owns this); camera sampling or SDS gradient computation (SDS owns these)

## Public API
Public API: src/data/api.py

## CRUD Ownership Table (Directive 41)

| Object | Created by | Modified by | Read by | Destroyed by |
|---|---|---|---|---|
| `IdentityEmbedding` | DATA (Directive 7) | never — read-only after creation (immutable; enforce via checksum check at every read site) | SDS, VALID (Directive 27) | end of run (not persisted beyond run_manifest hash reference) |

## Dependency Graph Reference (Directive 0.3)
Depends on: ENV
Depended on by: GAUSS, PRIOR, TEST
