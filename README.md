# TripleVar

**Multimodal variant-effect prediction for the thiamine transporters SLC19A2 and SLC19A3.**

TripleVar fuses three views of every variant and classifies it as
pathogenic / likely pathogenic / likely benign / benign:

| Modality | Model | Dim |
|---|---|---|
| Genomic context: the mutated gene window (26–37 kb, transcript orientation), 5 chunks mean-pooled | DNABERT-2 (`zhihan1996/DNABERT-2-117M`) | 3,840 |
| Protein: the translated mutant protein, mean over residues | ESM-2 3B (`esm2_t36_3B_UR50D`, layer 36) | 2,560 |
| Population data: per-study allele frequency + presence for 32 dbSNP studies, summaries, validation flags | — | 71 |

The fused 6,471-d vector is standardised and passed to an MLP
(`Linear → BatchNorm → LeakyReLU → Dropout` blocks).

> Abo-Dahab Y, Chun J, Giacomini KM. *TripleVar: A Multimodal Deep Learning Framework for
> High-Fidelity Clinical Variant Effect Prediction in the Thiamine Transporter Genes SLC19A2
> and SLC19A3.* ASCPT Annual Meeting 2026. Abstract [10.5281/zenodo.17927676](https://doi.org/10.5281/zenodo.17927676),
> poster [10.5281/zenodo.18929630](https://doi.org/10.5281/zenodo.18929630).

## Results at a glance

Full tables, confidence intervals and the model-selection grid: [`reports/RESULTS.md`](reports/RESULTS.md).

**Abstract reproduced.** The abstract model (`models/triplevar_abstract_2025-09-11.pt`) gives,
on the published split, exactly the abstract's ClinVar numbers (P vs B: 93.3 % accuracy,
100 % precision, 84.6 % recall, AUC 0.996; with LP/LB: AUC 0.913) and the 111 / 303
prioritised candidates (`triplevar reproduce-abstract`).

**Release model `models/triplevar_v1.pt`** (5-seed ensemble of the configuration selected by
leakage-free cross-validation). For comparison, the abstract model scores 85.6 % / 91.9 % /
75.6 % / 94.2 %, AUC 0.901 on the time-split and AUC 0.773 on the literature set.

| Evaluation | n (P/B) | Accuracy | Precision | Recall | Specificity | AUC |
|---|---|---|---|---|---|---|
| ClinVar **time-split** — variants classified after the training snapshot | 97 (45/52) | 87.6 % | 94.6 % | 77.8 % | 96.2 % | 0.909 |
| Literature set of the abstract (no frequency data) | 15 (11/4) | 66.7 % | 87.5 % | 63.6 % | 75.0 % | 0.909 |
| Grouped 10-fold CV × 5, P vs B, trusted labels | 140 (72/68)* | 88.6 % | — | — | — | 0.954 |
| Grouped 10-fold CV × 5, P/LP vs B/LB, trusted labels | 565 (104/461)* | 91.4 % | 77.0 % | 75.8 % | 94.9 % | 0.904 |

\* out-of-fold predictions per repeat; values are means over 5 repeats. "Trusted" = rows whose label belongs to their own allele
(see `docs/AUDIT.md`). The abstract's own split lets alleles of one rsID fall on both
sides of the split; with rsIDs kept together the same model family scores lower,
which is why the release model is evaluated on grouped folds and external data.

**Compared with established predictors on the same variants, TripleVar is worse.** CADD v1.7
reaches AUC 0.963 vs 0.807 for TripleVar on 764 labeled single-nucleotide variants (time-split:
0.989 vs 0.850), and AlphaMissense (0.870) and REVEL (0.902) beat it on 129 missense variants
(0.635). TripleVar's only extra coverage is insertions/deletions (out-of-fold AUC 0.99), which
the CADD web service does not score. Use calibrated established predictors as clinical evidence;
see [`validation/benchmarks/BENCHMARK.md`](validation/benchmarks/BENCHMARK.md). A modality
ablation (`configs/ablation/`) shows the ESM-2 protein embedding carries almost all of the signal
and the DNABERT-2 embedding adds none.

What changed versus the abstract model: rsID-grouped evaluation, allele-frequency
dropout during training (variants without population data are scored on sequence
evidence), training only on labels that belong to the variant's own allele, seeds,
and a self-contained model file.

**Prioritisation.** Of 29,271 unclassified alleles (507 ClinVar VUS, 28,764 unlabelled dbSNP
alleles), the release model calls 1,339 pathogenic or likely pathogenic
(`reports/predictions/prioritised_pathogenic.triplevar_v1.csv`; full ranked list in
`reports/supplementary/DataS1b_unclassified_ranked.csv`).

## Install

```bash
conda env create -f environment.yml      # Python 3.8, PyTorch 1.13 + CUDA 11.7
conda activate triplevar
pip install -e ".[embeddings,dev]"
```

Scoring pre-computed features needs only the core package (CPU is fine).
Embedding new variants needs the `embeddings` extra and a GPU with ~16 GB
(ESM-2 3B in fp32).

## Pipeline

```
data/raw/dbsnp/*.txt ──prepare──▶ per-allele table (sequences + frequency features)
                                        │
        data/reference/*.window.fa ─────┤
                                        ▼
                         embed (DNABERT-2, ESM-2) ──▶ FeatureSet (.npz + .csv)
                                                          │
                       train (grouped CV, validation-only selection) ──▶ model bundle / ensemble
                                                          │
                              predict / evaluate ◀────────┘
```

| Command | Purpose |
|---|---|
| `triplevar reference` | download the wild-type genomic windows from NCBI (or `--from-legacy`) |
| `triplevar prepare --gene SLC19A3 --export data/raw/dbsnp/SLC19A3_dbsnp_export.txt --out …` | raw dbSNP export → alleles, genomic/mRNA/protein sequences, frequency features |
| `triplevar build-external --input variants.csv --out data/processed/my_set` | variant list (GRCh38 pos/ref/alt or `E320Q`-style protein changes) → embedded FeatureSet |
| `triplevar embed --input table.csv --out …` | embed a table that already has `transcript_dna` and `protein_sequence` |
| `triplevar train --config configs/production.yaml` | grouped 10-fold CV, fold-model ensemble, external evaluation |
| `triplevar predict --bundle models/triplevar_v1.pt --features data/processed/pool --out preds.csv` | per-variant calls and probabilities |
| `triplevar evaluate --model … --external NAME=PATH` | score a model on labelled external sets |
| `triplevar reproduce-abstract` | re-score the abstract model on the published data |
| `triplevar import-legacy` | export the published datasets/embeddings as FeatureSets |

Every model file is self-contained (weights, architecture, fitted scaler, feature
layout, class order), so predictions never depend on re-creating a split.

### Scoring new variants

```bash
cat > my_variants.csv <<EOF
variant_id,gene,pos,ref,alt,protein_change,label_binary
v1,SLC19A3,,,,K44E,
v2,SLC19A2,169477685,C,G,,
EOF
triplevar build-external --input my_variants.csv --out data/processed/my_variants
triplevar predict --bundle models/triplevar_v1.pt --features data/processed/my_variants --out my_predictions.csv
```

Coordinates are GRCh38, + strand; VCF-style anchored indels are accepted.
Variants without population data get an all-zero frequency block (the
production model is trained to handle this).

## Reproducing the abstract

The original working directory is needed (it holds the published embeddings and
checkpoints): link it as `data/legacy` (see `data/README.md`), then

```bash
triplevar reference --from-legacy data/legacy
triplevar reproduce-abstract          # -> reports/abstract_reproduction/
pytest -m integration                 # asserts the abstract's numbers
```

## Reproducing the manuscript

```bash
python scripts/paper_numbers.py        # every table value -> reports/paper_numbers.json
python scripts/paper_figures.py        # Figures 1-4, S1 -> reports/figures/ (vector PDF, 1,000-dpi CMYK TIFF, PNG)
python scripts/export_predictions.py --model models/triplevar_v1.pt   # -> reports/predictions/
python scripts/paper_data_s1.py        # Supplementary Data S1 + column dictionary -> reports/supplementary/
```

The benchmark against CADD, AlphaMissense and REVEL has its own scripts and provenance in
`validation/benchmarks/`; the modality ablation configurations are in `configs/ablation/`.

## Repository layout

```
src/triplevar/      package (genes, sequences, prepare, embeddings, features, model, training, metrics, …)
configs/            run configurations (production + the search/ablation grid)
scripts/            builders for external sets, experiment summaries
validation/external external validation sets with provenance (ClinVar time-split, literature, label audit)
reports/            reproduction and results reports
models/             released models (Git LFS)
notebooks/archive/  the original research notebooks (read-only provenance)
docs/AUDIT.md       issues found in the 2025 code and how they are handled
data/               raw dbSNP exports (versioned); everything else generated locally
```

## Tests

```bash
pytest                 # unit tests; integration tests run when data/legacy is linked
```

## License

MIT — see `LICENSE`.
