# Released models (Git LFS)

Load either file with `triplevar.ensemble.load_model(path)` or use it directly on the CLI
(`triplevar predict --bundle …`). Each file contains the weights, architecture,
fitted standardisation, feature layout (DNABERT-2 3840 | ESM-2 2560 | frequency 71)
and class order `[-2, -1, 1, 2, 3]` = P, LP, LB, B, normal.

| File | What | Use |
|---|---|---|
| `triplevar_v1.pt` | release model (2026-10-12): 5-seed ensemble of `configs/production.yaml`; each member trained on all labeled rsIDs except one grouped validation fold (early stopping); trusted labels only; frequency dropout 0.3; weight matrices stored in float16 | new predictions |
| `triplevar_abstract_2025-09-11.pt` | the ASCPT 2026 abstract model (`best_model__(2025_09_11)_16|13.pt`) with the scaler re-fitted on its published split | reproducing the abstract |

Inputs must be built with this package (`triplevar build-external` / `prepare` + `embed`):
the DNA embedding expects the transcript-orientation genomic window of
`triplevar.genes` and five-chunk DNABERT-2 pooling, the protein embedding ESM-2 3B
layer 36 mean pooling.
