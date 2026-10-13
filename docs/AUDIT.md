# Audit of the 2025 research code (October 2026)

Findings from porting the notebooks in `notebooks/archive/` to the `triplevar`
package, ordered by their effect on reported results. Each item says what the
package does about it.

## 1. Which model the abstract reports

* The ASCPT 2026 abstract numbers come from `best_model__(2025_09_11)_16|13.pt`
  (LeakyReLU) on the notebook's row-level seed-42 split. With the scaler re-fitted
  on that split, it reproduces the abstract model's saved predictions for all
  31,199 pool variants (max probability difference 3e-6) and every ClinVar-test
  number in the abstract (`triplevar reproduce-abstract`).
* The literature-set numbers in the abstract (80 % accuracy, 90 % precision,
  82 % recall, 75 % specificity on 15 variants) came from a run at 14:15 on
  2025-09-11 whose weights were not saved; only its summary text survives
  (`binary_predictions_summary_20250911_141539.txt`, 15/22 called pathogenic).
  The saved 16:13 checkpoint gives 80 % accuracy, 83 % precision, 91 % recall,
  50 % specificity on the same 15 variants.
* The `results/FINAL_MODEL` predictions of 2025-09-22 come from an in-memory
  model that was never saved; no random seed was set, so it cannot be recreated.

## 2. Leakage and evaluation design

| Issue | Effect | Package |
|---|---|---|
| Train/val/test split by **row**: alleles of one rsID (and its paired wild-type row) land on both sides | optimistic test metrics (see results table in README) | `split: cv` groups by rsID (`triplevar.splits`) |
| Test set used for model comparison and checkpoint naming; one threshold tuned on test | optimistic | selection uses validation folds only; external sets scored once |
| Same validation set used for grid search and early stopping of the retrain | mild optimism | per-fold validation for early stopping; configs compared by mean validation AUC |
| "Binary" metric in some cells dropped samples *predicted* Normal | inflated binary accuracy | `metrics.py` defines every rule explicitly; nothing is dropped by prediction |
| Normal class = 962 copies of the two wild-type sequences (only the imputed frequencies differ) | inflates 5-class and 3-class accuracy (52 % of the test set, recall ≈ 1.0) | P/LP vs B/LB metrics exclude it; `drop_normal_fraction` option |

## 3. Labels

* dbSNP reports clinical significance **per rsID**, and the export was expanded to
  one row per allele, so every allele inherited the rsID label. Against ClinVar
  submissions dated on or before the 2025-06-27 snapshot
  (`validation/external/training_label_audit.csv`): of the 864 trained
  P/LP/B/LB rows, 200 carried a label copied from a sibling allele and 60
  disagreed with their own allele's record. All four pathogenic/benign flips
  seen in ClinVar today (e.g. SLC19A2 p.Tyr378Ter trained as *likely benign*)
  are copied labels. `triplevar.audit` drops these rows from training on request
  (`--drop-untrusted-labels`) and reports metrics on the trusted subset.
* A stop-codon relabelling step in the preparation notebooks turned some
  pathogenic labels into likely-benign based on the pipeline's own translated
  protein length; it is not ported.

## 4. External literature set (abstract)

* 4 of the 15 labelled variants are labeled training examples: T422A, W94R and
  Y113H (trained as likely pathogenic) and E320Q (trained as *likely benign*,
  external label pathogenic). E110K, E110Q and W59R are in ClinVar as VUS.
* The labels follow thiamine-binding affinity (Gabriel et al. 2024, nanoDSF
  Kd): ≤1.5× WT → likely benign, 5–12× → likely pathogenic, ≥13× → pathogenic.
  N297A ("pathogenic", 243× Kd) retains 75 % of cellular uptake in Li et al.
  2024; L35A was only measured inside a sextuple mutant; I301A was not measured.
* The literature variants have no population data, so their frequency block is
  all zeros, which after standardisation reads as "absent from every database"
  and pushes the model towards pathogenic. The package can train with
  frequency dropout (`tabular_dropout`) so this input state is seen in training.
* Replacement sets with rules fixed before scoring are in
  `validation/external/literature_rule_labels.csv` (see LITERATURE_SOURCES.md).

## 5. Data-preparation bugs (fixed in `triplevar.prepare`, flagged where not)

| Bug | Fix |
|---|---|
| Exact duplicate lines of the dbSNP export never removed (158 / 367) | dropped on read |
| Frequency features matched on the SPDI-rewritten allele → 0 for 317 / 502 corrected rows | matched on dbSNP's own allele spelling |
| Study columns created from an unordered `set`; tabular block selected with `columns.isin` (DataFrame order) | one fixed `TABULAR_COLUMNS` list, selected by name |
| SPDI correction applies the first SPDI of an rsID to all its alleles | kept for reproducibility; mismatches are reported per row (`allele_status`) |
| Scalers never saved; checkpoints were bare state_dicts | model bundles carry weights, architecture, scaler, layout and classes |
| No random seeds for training | `set_seed` (Python, NumPy, torch, cuDNN) |
| NCBI API key and personal e-mail hard-coded in notebooks pushed to a public repository | removed from archived copies; **rotate the key** — it is still in the git history of the public repository |
