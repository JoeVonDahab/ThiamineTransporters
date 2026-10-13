# data/

Only `raw/dbsnp/` is versioned. Everything else is generated or linked locally.

| Path | What | How to get it |
|---|---|---|
| `raw/dbsnp/SLC19A{2,3}_dbsnp_export.txt` | dbSNP web tabular exports (GRCh38) behind the published dataset; labels are dbSNP's per-rsID clinical significance | versioned |
| `reference/SLC19A{2,3}.window.fa` | wild-type genomic windows (gene + flanks, + strand) | `triplevar reference` (NCBI) or `triplevar reference --from-legacy data/legacy` |
| `legacy/` | symlink to the original working directory (`DNA_bert_GIT_Modified/Both Genes`) holding the published pickles, embeddings and checkpoints | `ln -s "/path/to/DNA_bert_GIT_Modified/Both Genes" data/legacy` |
| `processed/labeled.{npz,csv}` | 2,333 labeled rows (864 ClinVar-labeled variants, 507 VUS, 962 wild-type rows) with DNABERT-2 / ESM-2 / frequency features | `triplevar import-legacy` |
| `processed/pool.{npz,csv}` | 31,199 unlabeled dbSNP rows used for prioritisation | `triplevar import-legacy` |
| `processed/literature*.{npz,csv}` | literature variants (abstract set; rule-labelled transport and binding sets) | `triplevar import-legacy`, `scripts/build_literature_sets.py` |
| `processed/clinvar_timesplit.{npz,csv}` | 97 ClinVar variants classified after the training snapshot | `scripts/build_clinvar_timesplit.py` |
| `external_raw/` | downloaded ClinVar / UniProt records and the two 2024 Nat Commun papers + source data (not redistributed) | see `validation/external/*.md` |

`.npz` files hold the three feature blocks (`dna` 3840, `protein` 2560, `tabular` 71) and `y`
when labeled; the matching `.csv` holds one metadata row per variant in the same order.
