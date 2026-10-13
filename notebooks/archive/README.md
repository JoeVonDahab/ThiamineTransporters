# Archived research notebooks (May 2025 – Feb 2026)

The notebooks that produced the published TripleVar results, kept for
provenance. They are **not** needed to run anything: every step now lives in the
`triplevar` package. Outputs are left in place; an NCBI API key and e-mail
address were redacted. They ran in the `dnabert2` conda env (Python 3.8,
PyTorch 1.13, transformers 4.29, fair-esm 2.0) and expect the original
directory layout (`final_dataset/`, `embeddings/`, `models/`, ...).

| Notebook | What it did | Now |
|---|---|---|
| `01_prepare_data_SLC19A2.ipynb` | dbSNP export → alleles, genomic window, SPDI allele fixes, mRNA, Kozak-aware translation, frequency features, normal-row imputation → `final_dataset_SLC19A2_model_5.pkl` | `triplevar.prepare`, `triplevar.sequences`, `triplevar.genes` |
| `01_prepare_data_SLC19A3_and_literature_variants.ipynb` | same for SLC19A3; cells 58–91 also built the 22 literature variants (`custom_mutation.csv`) | same + `triplevar.sequences.protein_change_to_variant` |
| `02_embeddings_and_training_both_genes.ipynb` | DNABERT-2 (5 × 768) and ESM-2 3B embeddings for labeled set and pool; early combined-gene training (Aug 2025). Its Feb-2026 cell 54 is an unfinished DNABERT-only test | `triplevar.embeddings` |
| `03_final_training_pool_and_literature_predictions.ipynb` | grid search + training of the September models; predictions for VUS, the 31,199-row pool and the literature set. **The abstract model (`best_model__(2025_09_11)_16|13.pt`) and the `results/FINAL_MODEL` CSVs came from here** (the FINAL_MODEL weights were never saved) | `triplevar.training`, `triplevar.reproduce`, `triplevar.predict` |
| `03b_model2_model3_variants.ipynb` | the 2-embedding (`model_2…`, no frequency) and `model_3…` variants of 2025-09-22 | `configs/search/a03_*` ablation |
| `04_results_analysis.ipynb`, `04b_…` | selection of SLC19A2 variants for the HEK-cell assays, plasmid sanity checks | — |
| `early_per_gene/` | first per-gene models (May–Jul 2025) and a synthetic-variant demo | superseded |
| `figures/` | PNGs the notebooks wrote into `graphs/` | — |

Issues found while porting (all handled in the package; see `docs/AUDIT.md`):
row-level split letting alleles of one rsID straddle train/test, labels copied
across alleles of multi-allelic rsIDs, test set reused for model selection,
unsaved scalers, `x_3` column order taken from the DataFrame, frequency lost for
SPDI-corrected alleles, and no random seeds for training.
