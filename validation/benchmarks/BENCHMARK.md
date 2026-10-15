# TripleVar vs AlphaMissense, REVEL and CADD: head-to-head benchmark

Generated 2026-10-15. Everything here is computed on the **same variants** for every tool. Files in this folder:

| file | content |
|---|---|
| `benchmark_scores.csv` | one row per variant (961 = 864 CV + 97 time-split): keys, labels, TripleVar scores/calls, AlphaMissense, REVEL, CADD, plus join provenance columns (`am_transcript`, `am_protein_variant`, `revel_transcripts`, `revel_aa`, `revel_rule`, `cadd_consequence`, `cadd_feature`, `cadd_aa`, `cadd_na_reason`) |
| `benchmark_metrics.csv` | tidy table: one row per (cohort, scope, tool, metric, rule) with value, 95% CI, n, n_pos, n_neg, confusion counts, paired-test p values |
| `fig_roc_timesplit.png` | ROC, ClinVar time-split: (a) all 78 SNVs (TripleVar v1, abstract model, CADD); (b) the 10 missense SNVs (all five scores) |
| `fig_roc_cv_missense.png` | ROC, CV missense SNVs: (a) all labels, n = 129; (b) trusted labels, n = 33. In (b) the REVEL curve is hidden under CADD's (near-identical ranking) |
| `_coverage.csv` | scored-variant counts per set and tool |
| `scripts/` | the code that produced all of the above (see "Reproduce") |

Raw score files: `data/external_raw/benchmarks/` (git-ignored; nothing here is committed).

## 1. Key results

- **ClinVar time-split, all variants.** TripleVar v1 has ROC-AUC 0.909 on all 97 variants. 19 of those are indels or an MNV, which no other tool scores. On the **78 SNVs** that CADD also scores, **CADD v1.7 reaches 0.989 [0.969-1.000] and TripleVar 0.850 [0.729-0.945]**. The paired difference is -0.139 [-0.259, -0.034] (bootstrap p = 0.006, DeLong p = 0.014). The abstract-era model scores 0.762 on the same SNVs; v1 beats it by +0.088 [+0.011, +0.175].
- **ClinVar time-split, missense.** There are only 10 missense SNVs (8 P/LP, 2 B/LB), so every AUC has a CI of roughly 0 to 1. The values are AlphaMissense 1.000, REVEL 0.812, CADD 0.688, TripleVar 0.625 and abstract model 0.500. **These are not evidence either way.** One of the two benign labels is questionable (see section 6).
- **CV set (out-of-fold), all labels.** TripleVar has AUC 0.870 on all 864 variants. On the 764 SNVs, CADD scores 0.963 [0.928-0.989] against TripleVar's 0.807 [0.747-0.865] (difference -0.157 [-0.223, -0.092], p < 0.001).
- **CV set, trusted labels** (own-allele labels only). TripleVar has AUC 0.912 on 565 variants. On the 495 SNVs, CADD scores 0.983 against TripleVar's 0.845 (difference -0.139 [-0.210, -0.069], p < 0.001).
- **CV missense (all four tools on the same variants).** With all labels (n = 129; 42 P/LP, 87 B/LB): **TripleVar 0.635 [0.522-0.747]**, AlphaMissense 0.870, CADD 0.888 and REVEL 0.902. TripleVar's differences are -0.23 to -0.27 (all p ≤ 0.001). With trusted labels (n = 33; 23:10): TripleVar 0.787, CADD 0.991, REVEL 0.991 and AlphaMissense 1.000 (differences about -0.21, bootstrap p ≤ 0.006, DeLong p 0.012-0.021).
- **Calls at the published thresholds.** TripleVar's own call is conservative on missense: sensitivity is 0.405 on CV with all labels and 0.348 with trusted labels, at specificity 0.874 and 1.000. AlphaMissense (>0.564) gives 0.738/0.839 and 0.826/1.000 on the same variants. REVEL (>0.5) and CADD (≥20) are highly sensitive but less specific with all labels (specificity 0.655 and 0.644).
- **Bottom line for the manuscript.** On SNVs, and on missense SNVs in particular, TripleVar does not match CADD v1.7, REVEL or AlphaMissense. The deficit is statistically clear on the CV set and on the time-split SNVs. What TripleVar adds is coverage: it is the only score for the 119 indels and MNV in these sets (100 CV, 19 time-split). The external tools' advantage is probably inflated by circularity (section 6), but that inflation would have to be very large to close a 0.14-0.27 AUC gap.

## 2. Score sources

Gene windows (GRCh38, as in `data/reference/*.window.fa`): SLC19A2 chr1:169,461,909-169,487,970; SLC19A3 chr2:227,682,763-227,720,028.

| tool | version / file | source URL | retrieved (UTC) | integrity | licence |
|---|---|---|---|---|---|
| AlphaMissense | `AlphaMissense_hg38.tsv.gz` (642,961,469 bytes; GCS last-modified 2023-08-03; GENCODE v32 canonical transcripts, UniProt 2021_02) | https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz (same file as Zenodo record 8208688, doi:10.5281/zenodo.8208688, published 2023-09-19) | 2026-10-15 10:43 | md5 `9fd167735f16a1b87da6eb3e4c25fcb5` = Zenodo checksum | CC BY-NC-SA 4.0 |
| REVEL | v1.3, `revel-v1.3_all_chromosomes.zip` -> `revel_with_transcript_ids` (667,102,707 bytes; file date 2021-05-03) | https://rothsj06.dmz.hpc.mssm.edu/revel-v1.3_all_chromosomes.zip (primary link on https://sites.google.com/site/revelgenomics/downloads); identical to Zenodo record 7072866 (doi:10.5281/zenodo.7072866, v1.3, 2021-05-03). The Zenodo download stalled at ~120 kB/s, so the MSSM mirror was used | 2026-10-15 10:47 | md5 `3ea2bc33e6b5455fc7e9899da863b5fe` = Zenodo checksum | free for non-commercial use |
| CADD | GRCh38-v1.7 (with annotations; VEP/Ensembl 110) | web API `https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7_inclAnno/<chrom>:<pos>_<ref>_<alt>`, one request per variant, <=1 request/s, 961 requests | 2026-10-15 10:46-11:03 | raw JSON of every response kept | CADD terms (non-commercial) |

Raw / filtered files in `data/external_raw/benchmarks/`:
`AlphaMissense_hg38.tsv.gz`, `AlphaMissense_hg38.SLC19A2_SLC19A3.tsv` (6,426 rows: 3,187 SLC19A2 ENST00000236137.10/O60779 and 3,239 SLC19A3 ENST00000258403.8/Q9BZV2), `AlphaMissense_README_zenodo8208688.md`, `alphamissense_zenodo_record_8208688.json`, `AlphaMissense_hg38.headers.txt`;
`revel-v1.3_all_chromosomes.zip`, `revel-v1.3.SLC19A2_SLC19A3.csv` (6,787 rows, filtered on `grch38_pos`), `revel_zenodo_record_7072866.json`, `revel-v1.3_all_chromosomes.source.txt`;
`cadd_GRCh38-v1.7_inclAnno_api_responses.jsonl` (query, URL, fetch time, HTTP status and full response for each variant).

TripleVar inputs (read-only): CV labels `data/processed/labeled.csv`; out-of-fold probabilities `outputs/final/production.oof.csv` (production config, 10-fold grouped CV); label audit `validation/external/training_label_audit.csv`; time-split predictions of the release model `reports/predictions/clinvar_timesplit.triplevar_v1.csv`. Abstract-era model: `triplevar predict --bundle models/triplevar_abstract_2025-09-11.pt --features data/processed/clinvar_timesplit --out /tmp/ts_abstract.csv` (97 rows, aligned with the v1 file by `variant_id`).

## 3. Variant sets

- **A. Cross-validation set.** `labeled.csv` rows with y in {-2, -1, 1, 2}: 864 variants (133 P/LP, 731 B/LB); 764 SNVs (86:678) and 100 indels (47:53). Alignment with the OOF file was asserted: its 1,826 rows are exactly the `labeled.csv` rows with y != 0, in order, with identical `snp_id` and `y`.
  - TripleVar score = (p_P + p_LP) / (p_P + p_LP + p_LB + p_B).
  - TripleVar call = grouped argmax (`triplevar.metrics.Scores.call_grouped`, the same rule as `pathogenic_call` in the prediction files).
  - `trusted` = `label_source == 'own_allele'` in the label audit. All 864 rows matched on (snp_id, pos, original, mutant); 565 are trusted (104:461).
  - These counts reproduce `production.summary.json` (n = 864 and n = 565), whose per-seed mean AUCs (0.873 all labels, 0.904 trusted) agree with the OOF AUCs here (0.870, 0.912).
- **B. ClinVar time-split.** 97 variants (45 P/LP, 52 B/LB): 78 SNVs (28:50), 18 indels and 1 MNV (ClinVar:2792939, AC>GT). All 97 rows already had GRCh38 chrom/pos/ref/alt, so no recovery from `clinvar_timesplit.csv` was needed. The UniProt row is chr1:169477685 C>G as specified. The score is `p_pathogenic` and the call is `pathogenic_call`.
- No variant occurs in both sets.

## 4. Joining rules and transcript choices

- **Coordinates.** All joins use the exact GRCh38 (chrom, pos, ref, alt), plus strand, 1-based.
  - CV alleles are dbSNP-style, with '-' marking an empty allele. SNVs are used as is. Deletions and delins start at `pos`. Insertions are placed after the base at `pos`, the convention the time-split matcher established. Each indel was anchored on the preceding reference base and vt-normalised (right-trim, left-shift, left-trim to one anchor base) against `data/reference/SLC19A{2,3}.window.fa`.
  - All 864 CV and 97 time-split REF alleles match the reference window.
- **Missense definition** (`is_missense`): an SNV that is in the AlphaMissense hg38 canonical-transcript table. The definition was cross-checked as follows:
  - It agrees with CADD/VEP `NON_SYNONYMOUS` on the canonical transcript for 763/764 CV SNVs and 78/78 time-split SNVs.
  - The one exception is rs2470581081, SLC19A3 2:227702318 T>G, p.Met1Leu (start-loss, LP, trusted). VEP calls it NON_SYNONYMOUS and REVEL scores it (0.63), but AlphaMissense has no start-codon rows. It is excluded from the missense subset and counted only in REVEL's tool-scored "all variants" n = 130.
  - It agrees with ClinVar's consequence for all 10 time-split missense variants.
  - For all 139 missense SNVs, AlphaMissense `protein_variant` matches CADD's oAA/protPos/nAA, matches REVEL's aaref/aaalt, and its reference residue matches TripleVar's `protein_sequence`.
- **Transcripts.**
  - SLC19A2: all tools use ENST00000236137 (MANE Select).
  - SLC19A3: the AlphaMissense hg38 table has only one transcript per gene, ENST00000258403.8 (GENCODE v32 canonical, UniProt Q9BZV2). REVEL rows list ENST00000258403 (with ENST00000541617 and others). CADD v1.7 annotates ENST00000644224 (MANE Select, NM_025243.4).
  - The requested ENST00000644224 is therefore not in the AlphaMissense hg38 table or in REVEL. Because the amino-acid change is identical across the three tools for every missense SNV here (139/139, including all SLC19A3 variants), the transcript difference does not affect any score used.
  - No max-over-transcripts was needed: every REVEL-scored allele had exactly one row containing the canonical transcript (`revel_rule = canonical` for 140/140). No REVEL row for these transcripts lacks a GRCh38 position.
- **CADD.** PHRED (`cadd_phred`) is the score used for ranking; RawScore (`cadd_raw`) is also stored.
  - All 842 SNVs were scored.
  - **No indel or MNV could be scored through the API.** The API returns either an empty list or, for 1-base-REF insertions, only the three SNVs at the anchor base. Exact-allele matching rejects those SNV rows, so their scores are not mistaken for the insertion's score. `cadd_na_reason` records which case applies.
  - Indels would need the offline CADD-scripts pipeline or a web-upload job.
- **Unscored variants**:
  - AlphaMissense and REVEL score missense SNVs only, plus REVEL's start-loss above. Synonymous, nonsense, splice, intronic and UTR SNVs and all indels are NA by design.
  - CADD is NA for the 100 CV indels and the 19 time-split indels/MNV.
  - TripleVar scores all 961 variants.

## 5. Metrics

- **ROC-AUC and PR-AUC.** Computed with scikit-learn; PR-AUC is average precision. The prevalence (the PR-AUC of a random score) is listed as the baseline.
- **95% CIs.** Percentile bootstrap: 2000 resamples of variants with replacement, redrawing resamples that contain only one class.
- **Paired comparisons.**
  - The same bootstrap indices are used for both tools on the common subset. `p_boot` = 2 x min(P(delta <= 0), P(delta >= 0)), floored at 1/2000.
  - A DeLong two-sided test is also reported.
  - Pairwise-common and all-tools-common subsets are identical here, because every compared tool scores every SNV or missense SNV in its cohort. Both are kept in the CSV.
- **Scopes in `benchmark_metrics.csv`.** `tool_scored` = every variant the tool scores in that cohort. `common:<tools>` = the variants scored by all listed tools: TripleVar + CADD (+ abstract model) for "all variants", and TripleVar + AlphaMissense + REVEL + CADD (+ abstract) for missense.
- **Thresholds.**
  - AlphaMissense: likely pathogenic > 0.564, likely benign < 0.34, otherwise ambiguous. This follows the `am_class` definition in the AlphaMissense README. It is reported two ways: ambiguous counted as benign, and ambiguous excluded (`n_excluded`).
  - REVEL: > 0.5, the authors' suggested cut-off.
  - CADD: PHRED >= 20 (top 1% of all possible SNVs).
  - TripleVar: its own grouped-argmax call.

## 6. Results

### 6.1 Coverage (variants with a score)

| set | tool | variants | scored | SNVs | missense SNVs | missense scored |
|---|---|---|---|---|---|---|
| ClinVar time-split | TripleVar v1 | 97 | 97 | 78 | 10 | 10 |
| ClinVar time-split | TripleVar abstract model | 97 | 97 | 78 | 10 | 10 |
| ClinVar time-split | AlphaMissense | 97 | 10 | 78 | 10 | 10 |
| ClinVar time-split | REVEL v1.3 | 97 | 10 | 78 | 10 | 10 |
| ClinVar time-split | CADD v1.7 (PHRED) | 97 | 78 | 78 | 10 | 10 |
| CV, all labels | TripleVar v1 | 864 | 864 | 764 | 129 | 129 |
| CV, all labels | AlphaMissense | 864 | 129 | 764 | 129 | 129 |
| CV, all labels | REVEL v1.3 | 864 | 130 | 764 | 129 | 129 |
| CV, all labels | CADD v1.7 (PHRED) | 864 | 764 | 764 | 129 | 129 |
| CV, trusted labels | TripleVar v1 | 565 | 565 | 495 | 33 | 33 |
| CV, trusted labels | AlphaMissense | 565 | 33 | 495 | 33 | 33 |
| CV, trusted labels | REVEL v1.3 | 565 | 34 | 495 | 33 | 33 |
| CV, trusted labels | CADD v1.7 (PHRED) | 565 | 495 | 495 | 33 | 33 |

### 6.2 ROC-AUC and PR-AUC (common subsets; plus TripleVar on all variants)

The "all SNVs" rows are the TripleVar-vs-CADD common subset. The "missense SNVs" rows are the all-tools common subset. Tool-scored rows for every tool are in `benchmark_metrics.csv`.

| set | variants | tool | n (P/LP:B/LB) | ROC-AUC [95% CI] | PR-AUC [95% CI] | PR-AUC baseline |
|---|---|---|---|---|---|---|
| ClinVar time-split | all variants (incl. indels) | TripleVar v1 | 97 (45:52) | 0.909 [0.833-0.970] | 0.934 [0.879-0.976] | 0.464 |
| ClinVar time-split | all variants (incl. indels) | TripleVar abstract model | 97 (45:52) | 0.844 [0.749-0.921] | 0.886 [0.811-0.943] | 0.464 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar v1 | 78 (28:50) | 0.850 [0.729-0.945] | 0.853 [0.737-0.942] | 0.359 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar abstract model | 78 (28:50) | 0.762 [0.628-0.881] | 0.779 [0.640-0.888] | 0.359 |
| ClinVar time-split | all SNVs (any consequence) | CADD v1.7 (PHRED) | 78 (28:50) | 0.989 [0.969-1.000] | 0.981 [0.946-1.000] | 0.359 |
| ClinVar time-split | missense SNVs | TripleVar v1 | 10 (8:2) | 0.625 [0.000-1.000] | 0.876 [0.596-1.000] | 0.800 |
| ClinVar time-split | missense SNVs | TripleVar abstract model | 10 (8:2) | 0.500 [0.143-0.857] | 0.880 [0.664-0.977] | 0.800 |
| ClinVar time-split | missense SNVs | AlphaMissense | 10 (8:2) | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.800 |
| ClinVar time-split | missense SNVs | REVEL v1.3 | 10 (8:2) | 0.812 [0.429-1.000] | 0.953 [0.814-1.000] | 0.800 |
| ClinVar time-split | missense SNVs | CADD v1.7 (PHRED) | 10 (8:2) | 0.688 [0.143-1.000] | 0.907 [0.672-1.000] | 0.800 |
| CV, all labels | all variants (incl. indels) | TripleVar v1 | 864 (133:731) | 0.870 [0.828-0.908] | 0.681 [0.592-0.768] | 0.154 |
| CV, all labels | all SNVs (any consequence) | TripleVar v1 | 764 (86:678) | 0.807 [0.747-0.865] | 0.499 [0.383-0.618] | 0.113 |
| CV, all labels | all SNVs (any consequence) | CADD v1.7 (PHRED) | 764 (86:678) | 0.963 [0.928-0.989] | 0.846 [0.771-0.913] | 0.113 |
| CV, all labels | missense SNVs | TripleVar v1 | 129 (42:87) | 0.635 [0.522-0.747] | 0.574 [0.431-0.715] | 0.326 |
| CV, all labels | missense SNVs | AlphaMissense | 129 (42:87) | 0.870 [0.807-0.927] | 0.658 [0.516-0.837] | 0.326 |
| CV, all labels | missense SNVs | REVEL v1.3 | 129 (42:87) | 0.902 [0.848-0.951] | 0.770 [0.628-0.900] | 0.326 |
| CV, all labels | missense SNVs | CADD v1.7 (PHRED) | 129 (42:87) | 0.888 [0.830-0.938] | 0.738 [0.603-0.878] | 0.326 |
| CV, trusted labels | all variants (incl. indels) | TripleVar v1 | 565 (104:461) | 0.912 [0.869-0.948] | 0.858 [0.797-0.909] | 0.184 |
| CV, trusted labels | all SNVs (any consequence) | TripleVar v1 | 495 (61:434) | 0.845 [0.778-0.905] | 0.707 [0.600-0.803] | 0.123 |
| CV, trusted labels | all SNVs (any consequence) | CADD v1.7 (PHRED) | 495 (61:434) | 0.983 [0.944-1.000] | 0.984 [0.950-1.000] | 0.123 |
| CV, trusted labels | missense SNVs | TripleVar v1 | 33 (23:10) | 0.787 [0.611-0.934] | 0.907 [0.800-0.980] | 0.697 |
| CV, trusted labels | missense SNVs | AlphaMissense | 33 (23:10) | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.697 |
| CV, trusted labels | missense SNVs | REVEL v1.3 | 33 (23:10) | 0.991 [0.960-1.000] | 0.996 [0.982-1.000] | 0.697 |
| CV, trusted labels | missense SNVs | CADD v1.7 (PHRED) | 33 (23:10) | 0.991 [0.957-1.000] | 0.996 [0.982-1.000] | 0.697 |

### 6.3 Paired AUC differences (same variants)

| set | variants | comparison | n (P/LP:B/LB) | AUC (first, second) | delta AUC [95% CI] | p bootstrap | p DeLong |
|---|---|---|---|---|---|---|---|
| ClinVar time-split | all SNVs (any consequence) | TripleVar v1 minus CADD v1.7 (PHRED) | 78 (28:50) | 0.850, 0.989 | -0.139 [-0.259, -0.034] | 0.0060 | 0.014 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar abstract model minus CADD v1.7 (PHRED) | 78 (28:50) | 0.762, 0.989 | -0.227 [-0.365, -0.103] | 0.0005 | 0.00082 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar v1 minus TripleVar abstract model | 78 (28:50) | 0.850, 0.762 | +0.088 [+0.011, +0.175] | 0.0240 | 0.042 |
| ClinVar time-split | missense SNVs | TripleVar v1 minus TripleVar abstract model | 10 (8:2) | 0.625, 0.500 | +0.125 [-0.444, +0.714] | 0.8100 | 0.76 |
| ClinVar time-split | missense SNVs | TripleVar abstract model minus REVEL v1.3 | 10 (8:2) | 0.500, 0.812 | -0.312 [-0.810, +0.333] | 0.3780 | 0.34 |
| ClinVar time-split | missense SNVs | TripleVar abstract model minus AlphaMissense | 10 (8:2) | 0.500, 1.000 | -0.500 [-0.857, -0.143] | 0.0160 | 0.0082 |
| ClinVar time-split | missense SNVs | TripleVar abstract model minus CADD v1.7 (PHRED) | 10 (8:2) | 0.500, 0.688 | -0.188 [-0.762, +0.556] | 0.6430 | 0.65 |
| ClinVar time-split | missense SNVs | TripleVar v1 minus REVEL v1.3 | 10 (8:2) | 0.625, 0.812 | -0.188 [-1.000, +0.572] | 0.7600 | 0.75 |
| ClinVar time-split | missense SNVs | TripleVar v1 minus AlphaMissense | 10 (8:2) | 0.625, 1.000 | -0.375 [-1.000, +0.000] | 0.5590 | 0.33 |
| ClinVar time-split | missense SNVs | TripleVar v1 minus CADD v1.7 (PHRED) | 10 (8:2) | 0.625, 0.688 | -0.062 [-0.889, +0.875] | 0.9170 | 0.93 |
| CV, all labels | all SNVs (any consequence) | TripleVar v1 minus CADD v1.7 (PHRED) | 764 (86:678) | 0.807, 0.963 | -0.157 [-0.223, -0.092] | 0.0005 | 2.4e-06 |
| CV, all labels | missense SNVs | TripleVar v1 minus AlphaMissense | 129 (42:87) | 0.635, 0.870 | -0.235 [-0.374, -0.101] | 0.0010 | 0.0007 |
| CV, all labels | missense SNVs | TripleVar v1 minus REVEL v1.3 | 129 (42:87) | 0.635, 0.902 | -0.267 [-0.395, -0.137] | 0.0005 | 4.2e-05 |
| CV, all labels | missense SNVs | TripleVar v1 minus CADD v1.7 (PHRED) | 129 (42:87) | 0.635, 0.888 | -0.252 [-0.395, -0.115] | 0.0005 | 0.00019 |
| CV, trusted labels | all SNVs (any consequence) | TripleVar v1 minus CADD v1.7 (PHRED) | 495 (61:434) | 0.845, 0.983 | -0.139 [-0.210, -0.069] | 0.0005 | 0.00012 |
| CV, trusted labels | missense SNVs | TripleVar v1 minus REVEL v1.3 | 33 (23:10) | 0.787, 0.991 | -0.204 [-0.398, -0.050] | 0.0060 | 0.021 |
| CV, trusted labels | missense SNVs | TripleVar v1 minus AlphaMissense | 33 (23:10) | 0.787, 1.000 | -0.213 [-0.405, -0.070] | 0.0005 | 0.012 |
| CV, trusted labels | missense SNVs | TripleVar v1 minus CADD v1.7 (PHRED) | 33 (23:10) | 0.787, 0.991 | -0.204 [-0.404, -0.048] | 0.0060 | 0.021 |

### 6.4 Calls at published thresholds (same subsets)

| set | variants | tool | rule | n (P:B) | excl. | TP/FN/FP/TN | accuracy | sensitivity | specificity | PPV | NPV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ClinVar time-split | all variants (incl. indels) | TripleVar v1 | own_call(grouped_argmax) | 97 (45:52) | 0 | 35/10/2/50 | 0.876 | 0.778 | 0.962 | 0.946 | 0.833 |
| ClinVar time-split | all variants (incl. indels) | TripleVar abstract model | own_call(grouped_argmax) | 97 (45:52) | 0 | 30/15/1/51 | 0.835 | 0.667 | 0.981 | 0.968 | 0.773 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar v1 | own_call(grouped_argmax) | 78 (28:50) | 0 | 18/10/2/48 | 0.846 | 0.643 | 0.960 | 0.900 | 0.828 |
| ClinVar time-split | all SNVs (any consequence) | TripleVar abstract model | own_call(grouped_argmax) | 78 (28:50) | 0 | 15/13/1/49 | 0.821 | 0.536 | 0.980 | 0.938 | 0.790 |
| ClinVar time-split | all SNVs (any consequence) | CADD v1.7 (PHRED) | cadd_phred>=20 | 78 (28:50) | 0 | 26/2/1/49 | 0.962 | 0.929 | 0.980 | 0.963 | 0.961 |
| ClinVar time-split | missense SNVs | TripleVar v1 | own_call(grouped_argmax) | 10 (8:2) | 0 | 5/3/1/1 | 0.600 | 0.625 | 0.500 | 0.833 | 0.250 |
| ClinVar time-split | missense SNVs | TripleVar abstract model | own_call(grouped_argmax) | 10 (8:2) | 0 | 2/6/0/2 | 0.400 | 0.250 | 1.000 | 1.000 | 0.250 |
| ClinVar time-split | missense SNVs | AlphaMissense | am>0.564;ambiguous_as_benign | 10 (8:2) | 0 | 7/1/0/2 | 0.900 | 0.875 | 1.000 | 1.000 | 0.667 |
| ClinVar time-split | missense SNVs | AlphaMissense | am>0.564_vs_<0.34;ambiguous_excluded | 9 (7:2) | 1 | 7/0/0/2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ClinVar time-split | missense SNVs | REVEL v1.3 | revel>0.5 | 10 (8:2) | 0 | 8/0/1/1 | 0.900 | 1.000 | 0.500 | 0.889 | 1.000 |
| ClinVar time-split | missense SNVs | CADD v1.7 (PHRED) | cadd_phred>=20 | 10 (8:2) | 0 | 7/1/1/1 | 0.800 | 0.875 | 0.500 | 0.875 | 0.500 |
| CV, all labels | all variants (incl. indels) | TripleVar v1 | own_call(grouped_argmax) | 864 (133:731) | 0 | 96/37/61/670 | 0.887 | 0.722 | 0.917 | 0.611 | 0.948 |
| CV, all labels | all SNVs (any consequence) | TripleVar v1 | own_call(grouped_argmax) | 764 (86:678) | 0 | 50/36/56/622 | 0.880 | 0.581 | 0.917 | 0.472 | 0.945 |
| CV, all labels | all SNVs (any consequence) | CADD v1.7 (PHRED) | cadd_phred>=20 | 764 (86:678) | 0 | 83/3/39/639 | 0.945 | 0.965 | 0.942 | 0.680 | 0.995 |
| CV, all labels | missense SNVs | TripleVar v1 | own_call(grouped_argmax) | 129 (42:87) | 0 | 17/25/11/76 | 0.721 | 0.405 | 0.874 | 0.607 | 0.752 |
| CV, all labels | missense SNVs | AlphaMissense | am>0.564;ambiguous_as_benign | 129 (42:87) | 0 | 31/11/14/73 | 0.806 | 0.738 | 0.839 | 0.689 | 0.869 |
| CV, all labels | missense SNVs | AlphaMissense | am>0.564_vs_<0.34;ambiguous_excluded | 120 (37:83) | 9 | 31/6/14/69 | 0.833 | 0.838 | 0.831 | 0.689 | 0.920 |
| CV, all labels | missense SNVs | REVEL v1.3 | revel>0.5 | 129 (42:87) | 0 | 40/2/30/57 | 0.752 | 0.952 | 0.655 | 0.571 | 0.966 |
| CV, all labels | missense SNVs | CADD v1.7 (PHRED) | cadd_phred>=20 | 129 (42:87) | 0 | 41/1/31/56 | 0.752 | 0.976 | 0.644 | 0.569 | 0.982 |
| CV, trusted labels | all variants (incl. indels) | TripleVar v1 | own_call(grouped_argmax) | 565 (104:461) | 0 | 81/23/23/438 | 0.919 | 0.779 | 0.950 | 0.779 | 0.950 |
| CV, trusted labels | all SNVs (any consequence) | TripleVar v1 | own_call(grouped_argmax) | 495 (61:434) | 0 | 38/23/22/412 | 0.909 | 0.623 | 0.949 | 0.633 | 0.947 |
| CV, trusted labels | all SNVs (any consequence) | CADD v1.7 (PHRED) | cadd_phred>=20 | 495 (61:434) | 0 | 60/1/1/433 | 0.996 | 0.984 | 0.998 | 0.984 | 0.998 |
| CV, trusted labels | missense SNVs | TripleVar v1 | own_call(grouped_argmax) | 33 (23:10) | 0 | 8/15/0/10 | 0.545 | 0.348 | 1.000 | 1.000 | 0.400 |
| CV, trusted labels | missense SNVs | AlphaMissense | am>0.564;ambiguous_as_benign | 33 (23:10) | 0 | 19/4/0/10 | 0.879 | 0.826 | 1.000 | 1.000 | 0.714 |
| CV, trusted labels | missense SNVs | AlphaMissense | am>0.564_vs_<0.34;ambiguous_excluded | 30 (20:10) | 3 | 19/1/0/10 | 0.967 | 0.950 | 1.000 | 1.000 | 0.909 |
| CV, trusted labels | missense SNVs | REVEL v1.3 | revel>0.5 | 33 (23:10) | 0 | 23/0/1/9 | 0.970 | 1.000 | 0.900 | 0.958 | 1.000 |
| CV, trusted labels | missense SNVs | CADD v1.7 (PHRED) | cadd_phred>=20 | 33 (23:10) | 0 | 23/0/1/9 | 0.970 | 1.000 | 0.900 | 0.958 | 1.000 |

## 7. Caveats: circularity and label issues

1. **AlphaMissense** (Cheng et al. 2023, *Science* 381:eadg7492, doi:10.1126/science.adg7492; DeepMind blog, 19 Sep 2023).
   - It is fine-tuned on weak labels from population frequency: variants seen in human and primate populations are treated as benign, and unseen variants as pathogenic. No clinical labels are used in training.
   - Its class thresholds were set to reach 90% precision on known disease variants (ClinVar in the paper).
   - Consequences here:
     - Common benign variants in both our sets were effectively benign training examples for AlphaMissense. The same frequency circularity applies to TripleVar, whose ~70 frequency features come from dbSNP.
     - ClinVar labels that existed in 2023 may have influenced the threshold calibration, though not the per-variant scores.
2. **REVEL** (Ioannidis et al. 2016, *AJHG* 99:877-885, doi:10.1016/j.ajhg.2016.08.016).
   - It is a random forest over 13 component predictors. Training used HGMD 2015.2 missense mutations added after 2012-08-01 against rare ESP/ARIC/1000 Genomes variants, excluding variants used to train the components.
   - The components (MutPred, VEST, FATHMM, MutationTaster, PolyPhen-2 and others) were trained on HGMD, UniProt or SwissVar disease variants.
   - Classic TRMA and BTBGD missense mutations (for example SLC19A2 p.Asp93His, SLC19A3 p.Gly23Val and p.Thr422Ala, all in these sets) were published before 2015 and are very likely in HGMD. REVEL, or its components, has therefore probably seen many CV P/LP missense labels. This cannot be checked without an HGMD licence.
3. **CADD v1.7** (Schubach et al. 2024, *NAR* 52:D1143-D1154, doi:10.1093/nar/gkad989).
   - It is a logistic regression separating about 14 M human-derived (proxy-neutral) alleles from size-matched simulated (proxy-deleterious) variants. No clinical labels are used in training; ClinVar, gnomAD, ProteinGym and MPRA data are used for evaluation only.
   - Features include label-trained predictors (PolyPhen-2), plus ESM-1v, SpliceAI/MMSplice, conservation and consequence class.
   - Its near-perfect ranking of all-consequence SNVs mostly reflects variant class: nonsense and canonical-splice variants score high, synonymous and intronic variants low. That is real signal, not leakage.
4. **ClinVar label circularity affects every tool.**
   - ACMG/AMP classification uses computational evidence (PP3/BP4) and population frequency (BA1/BS1/PM2).
   - REVEL and CADD, and since 2023 AlphaMissense, are standard PP3/BP4 tools (ClinGen SVI calibration: Pejaver et al. 2022, *AJHG* 109:2163-2177, doi:10.1016/j.ajhg.2022.10.013).
   - Most time-split labels are 1-star, single-submitter clinical-lab calls made in 2023-2026, so the external scores may have contributed to the labels themselves. TripleVar benefits in the same way through frequency.
   - This inflation cannot be removed without SCV-level evidence codes.
5. **Temporal leakage in the time-split.** Labels postdate all three external tools (REVEL training 2015 / v1.3 2021, AlphaMissense Aug 2023, CADD v1.7 2023), with these exceptions:
   - Three variants had a concordant ClinVar SCV before AlphaMissense's release: ClinVar:372581 (SLC19A3 p.Gln418Glu, B/LB, SCV 2019), ClinVar:872654 (SLC19A3 c.894T>G nonsense, P, 2020) and ClinVar:2186483 (SLC19A3 c.151-17A>G, LB, 2023-02).
   - UniProt:VAR_010249 (SLC19A2 p.Asp93His) is an old literature TRMA variant, likely in HGMD.
6. **Time-split missense labels.** Only 2 of the 10 are B/LB.
   - One of them, ClinVar:372581 (p.Gln418Glu), is recorded as Benign/Likely benign, but the stored classification history shows "Likely pathogenic" (2019) and "Conflicting classifications" (2026-10-02).
   - REVEL (0.823) and CADD (27.6) call it pathogenic; AlphaMissense (0.168) and TripleVar (0.169) call it benign.
   - The time-split missense AUCs hinge on this one label, so re-check it before quoting them.
7. **CV is not like-for-like.**
   - TripleVar's scores are out-of-fold, but TripleVar was trained, in the other folds, on the same label source: dbSNP clinical significance, which mirrors ClinVar. Production training drops copied labels; OOF scores exist for all 864 rows.
   - The external tools never saw our folds but may have seen the same labels (points 1-4).
   - In the "all labels" missense set, 96 of the 129 labels are not own-allele labels (copied from a sibling allele, disagreeing with the allele's own ClinVar record, or unknown at the cutoff). Restricted to trusted labels (n = 33), all three external tools reach AUC 0.99-1.00.
8. **Small n.** The time-split missense set (n = 10) and the trusted CV missense set (n = 33, 10 B/LB) give wide or degenerate CIs. AUC = 1.000 with CI [1.000-1.000] means perfect separation in every resample, not certainty.

## 8. Reproduce

The scripts are in `scripts/`; the paths inside them point at this repo.

```bash
# 1. downloads (approved): AlphaMissense_hg38.tsv.gz (GCS) and revel-v1.3_all_chromosomes.zip (MSSM / Zenodo 7072866)
bash scripts/bm_am_filter.sh        # zcat + awk -> AlphaMissense_hg38.SLC19A2_SLC19A3.tsv
bash scripts/bm_revel_filter.sh     # unzip -p + awk -> revel-v1.3.SLC19A2_SLC19A3.csv
# 2. abstract-model predictions
triplevar predict --bundle models/triplevar_abstract_2025-09-11.pt --features data/processed/clinvar_timesplit --out /tmp/ts_abstract.csv
# 3. variant table, CADD API (resumable, <=1 req/s), metrics, figures, markdown tables
python scripts/bm_prep.py variants.csv
python -I scripts/bm_cadd_fetch.py variants.csv data/external_raw/benchmarks/cadd_GRCh38-v1.7_inclAnno_api_responses.jsonl
python scripts/bm_analysis.py variants.csv roc.pkl
python scripts/bm_figs.py roc.pkl
python scripts/bm_tables.py tables_fragment.md
```
