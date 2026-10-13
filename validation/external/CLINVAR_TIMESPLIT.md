# TripleVar external validation: ClinVar time-split + UniProt

Generated 2026-10-12 by `validation/external/_work/` scripts (`download_clinvar.py`, `parse_clinvar.py`, `prep_training.py`, `build_timesplit.py`, `map_uniprot.py`, `make_report.py`).

## 1. Sources and query

- **ClinVar** via NCBI E-utilities (`tool=triplevar`, no email/api_key, <=2 req/s). Query time (UTC): `2026-10-13T04:11:51.197835+00:00`. `esearch db=clinvar term="SLC19A2[gene] OR SLC19A3[gene]"` -> **1347 VariationIDs**; all fetched with `esummary` (JSON) and `efetch rettype=vcv` (VCV XML: VCF alleles, DateCreated, MostRecentSubmission, aggregate DescriptionHistory, and every SCV with its creation date and classification). Raw files: `data/external_raw/clinvar/` (esearch_ids.json, clinvar_esummary_all.json, esummary_batches/, vcv_xml_batches/). Tidy table: `validation/external/clinvar_current.csv` (one row per VariationID).
- **UniProt** release 2026_03: `rest.uniprot.org/uniprotkb/{O60779,Q9BZV2}.json|.txt` (Natural variant features), the UniProt variation service (`ebi.ac.uk/proteins/api/variation/<acc>`, for GRCh38 coordinates and cross-refs) and `humsavar.txt` (2026_03 of 02-Sep-2026). Raw files: `data/external_raw/uniprot/`.
- **Training data** (read-only): `DNA_bert_GIT_Modified/Both Genes/final_dataset/final_dataset_{SLC19A2,SLC19A3}_model_5.pkl` (22,133 + 38,892 rows; 25,321 synthetic `normal` WT rows dropped -> 35,704 allele rows). The training labels were cleaned with the exact `replacement_mapping` from `Training & Prediction Both.ipynb` (composite strings such as `pathogenic;likely-benign` -> `uncertain-significance`). Note: the notebook drops label 0 (VUS) before training, so VUS rows were never training examples, and they were also excluded from the unlabeled prediction pool.
- **Time-split cutoff** used for the strict subset: **2025-06-27** (build date of `final_dataset_20250627_*.pkl`, the dbSNP export the model_5 sets derive from).
- **Predictions**: Sep-11 2025 `results_abstract_sumbititon/predictions_with_details_new_model.csv` (multiclass label + max-softmax confidence), Sep-22 2025 `results/FINAL_MODEL/predictions_with_details_full_reduced_normal.csv` (multiclass) and `binary_predictions_20250922_180758.csv` (`Predicted_Class`, `Pathogenic_Probability`; class == P(path)>0.5 for all 31,199 rows). All three cover the same 31,199-row unlabeled pool, joined on (snp_id, original, mutant).

## 2. Matching and categories

Each ClinVar VariationID was matched to the training export on GRCh38 + strand: (1) exact `(chr, pos, ref, alt)` against `#chr, pos, original, mutant` (VCF alleles from the VCV XML); (2) after trimming shared prefix/suffix on both sides (training `-` = empty allele; dbSNP-export insertions are anchored one base earlier, which is corrected); (3) rsID + equivalent allele (same indel sequence up to repeat rotation); (4) for indels without an rsID hit, an equivalent placement inside ClinVar's fully-justified canonical SPDI span. Gene-spanning CNVs (genes other than SLC19A2/3) are `MULTI_GENE_CNV` and ignored. Sanity check: for 440/441 matched missense variants the ClinVar p. change equals the protein change derived from the training row's `protein_sequence` (the exception is the start-loss c.1A>C).

| match_method | n |
|---|---|
| pos_ref_alt | 1074 |
| none | 179 |
| pos_ref_alt_trimmed | 89 |
| rsid+equivalent_allele | 4 |
| rsid_only(allele_not_in_training) | 1 |

Categories (allele level; the training label is per rsID and was copied to every alt allele):

- `TRAIN_LABELED`: training label P/LP/B/LB -> was a training example, **excluded** from the external set.
- `TRAIN_VUS`: training label VUS (incl. composites mapped to VUS) and ClinVar now P/LP/B/LB -> time-split (never a training example, but no stored predictions).
- `TRAIN_UNLABELED`: in the export with NaN label and ClinVar now P/LP/B/LB -> time-split, **scorable with the stored predictions**.
- `ABSENT`: allele not in the export (none of these has an rsID) and ClinVar now P/LP/B/LB -> new variants, need fresh inference.
- `external_eligible` = ClinVar P/LP/B/LB and category in {TRAIN_VUS, TRAIN_UNLABELED, ABSENT}. `strict_temporal` = no ClinVar SCV with the current direction existed on/before the cutoff (otherwise the label existed in ClinVar but had not reached the dbSNP export: label lag rather than new evidence).

### All 1,347 ClinVar records by training category and current ClinVar direction

| training_category | B/LB | P/LP | VUS/conflicting/none | Total |
|---|---|---|---|---|
| ABSENT | 23 | 47 | 56 | 126 |
| MULTI_GENE_CNV | 0 | 45 | 8 | 53 |
| TRAIN_LABELED | 499 | 106 | 66 | 671 |
| TRAIN_UNLABELED | 26 | 11 | 87 | 124 |
| TRAIN_VUS | 3 | 3 | 367 | 373 |
| Total | 551 | 212 | 584 | 1347 |

## 3. External time-split set (`clinvar_timesplit.csv`, rows with `external_eligible=True`)

**113 variants: 61 P/LP, 52 B/LB.**

### By category x gene

| training_category | gene | B/LB | P/LP | Total |
|---|---|---|---|---|
| ABSENT | SLC19A2 | 12 | 13 | 25 |
| ABSENT | SLC19A3 | 11 | 34 | 45 |
| TRAIN_UNLABELED | SLC19A2 | 11 | 6 | 17 |
| TRAIN_UNLABELED | SLC19A3 | 15 | 5 | 20 |
| TRAIN_VUS | SLC19A3 | 3 | 3 | 6 |
| Total |  | 52 | 61 | 113 |

### Missense only

| training_category | gene | B/LB | P/LP | Total |
|---|---|---|---|---|
| ABSENT | SLC19A2 | 0 | 1 | 1 |
| ABSENT | SLC19A3 | 0 | 4 | 4 |
| TRAIN_UNLABELED | SLC19A2 | 0 | 1 | 1 |
| TRAIN_UNLABELED | SLC19A3 | 0 | 1 | 1 |
| TRAIN_VUS | SLC19A3 | 2 | 0 | 2 |
| Total |  | 2 | 7 | 9 |

| training_category | gene | hgvs_c | hgvs_p | clinvar_classification | review_stars |
|---|---|---|---|---|---|
| ABSENT | SLC19A2 | c.958T>C | p.Trp320Arg | Pathogenic | 1 |
| TRAIN_UNLABELED | SLC19A2 | c.205G>T | p.Val69Phe | Likely pathogenic | 2 |
| TRAIN_VUS | SLC19A3 | c.1252C>G | p.Gln418Glu | Benign/Likely benign | 2 |
| ABSENT | SLC19A3 | c.950G>A | p.Gly317Glu | Likely pathogenic | 1 |
| TRAIN_VUS | SLC19A3 | c.861C>A | p.Phe287Leu | Likely benign | 1 |
| TRAIN_UNLABELED | SLC19A3 | c.517A>G | p.Asn173Asp | Likely pathogenic | 1 |
| ABSENT | SLC19A3 | c.479T>C | p.Leu160Pro | Likely pathogenic | 1 |
| ABSENT | SLC19A3 | c.449C>T | p.Ala150Val | Likely pathogenic | 1 |
| ABSENT | SLC19A3 | c.170C>G | p.Pro57Arg | Likely pathogenic | 1 |

### By review stars

| training_category | stars | B/LB | P/LP | Total |
|---|---|---|---|---|
| ABSENT | 0 | 0 | 3 | 3 |
| ABSENT | 1 | 23 | 43 | 66 |
| ABSENT | 2 | 0 | 1 | 1 |
| TRAIN_UNLABELED | 1 | 24 | 9 | 33 |
| TRAIN_UNLABELED | 2 | 2 | 2 | 4 |
| TRAIN_VUS | 1 | 2 | 1 | 3 |
| TRAIN_VUS | 2 | 1 | 2 | 3 |
| Total |  | 52 | 61 | 113 |

### By consequence

| csq | B/LB | P/LP | Total |
|---|---|---|---|
| 5'UTR | 2 | 0 | 2 |
| frameshift | 0 | 15 | 15 |
| genic upstream transcript variant | 0 | 1 | 1 |
| intronic | 16 | 1 | 17 |
| missense | 2 | 7 | 9 |
| none (exon-level del / CNV / no annotation) | 0 | 16 | 16 |
| nonsense | 0 | 15 | 15 |
| splice acceptor | 0 | 4 | 4 |
| splice donor | 0 | 2 | 2 |
| synonymous | 32 | 0 | 32 |
| Total | 52 | 61 | 113 |

### Strict temporal vs label lag

| training_category | strict_temporal | B/LB | P/LP | Total |
|---|---|---|---|---|
| ABSENT | False | 13 | 28 | 41 |
| ABSENT | True | 10 | 19 | 29 |
| TRAIN_UNLABELED | False | 12 | 6 | 18 |
| TRAIN_UNLABELED | True | 14 | 5 | 19 |
| TRAIN_VUS | False | 3 | 3 | 6 |
| Total |  | 52 | 61 | 113 |

### Scorable by a sequence-window model (VCF alleles, <=50 bp)

| training_category | small_variant | B/LB | P/LP | Total |
|---|---|---|---|---|
| ABSENT | False | 0 | 17 | 17 |
| ABSENT | True | 23 | 30 | 53 |
| TRAIN_UNLABELED | True | 26 | 11 | 37 |
| TRAIN_VUS | True | 3 | 3 | 6 |
| Total |  | 52 | 61 | 113 |

### Submitters behind the eligible labels (concordant SCVs, top 8)

| index | n_concordant_SCVs |
|---|---|
| B: Labcorp Genetics (formerly Invitae), Labcorp | 39 |
| P: Labcorp Genetics (formerly Invitae), Labcorp | 25 |
| P: Myriad Genetics, Inc. | 14 |
| B: Blueprint Genetics | 10 |
| P: Women's Health and Genetics/Laboratory Corporation of America, LabCorp | 4 |
| P: Fulgent Genetics, Fulgent Genetics | 3 |
| B: CeGaT Center for Human Genetics Tuebingen | 2 |
| P: Bionano Laboratories | 2 |

## 4. Label changes vs the training labels

Allele-level comparison of the training label with ClinVar today, over all training P/LP/B/LB rows:

| index | rows |
|---|---|
| B: ClinVar VUS/conflicting | 64 |
| B: ClinVar agrees | 564 |
| B: ClinVar opposite | 3 |
| B: allele not in ClinVar today | 276 |
| B: allele not in ClinVar today, rsID multi-allelic in export | 276 |
| P: ClinVar VUS/conflicting | 13 |
| P: ClinVar agrees | 103 |
| P: ClinVar opposite | 1 |
| P: allele not in ClinVar today | 16 |
| P: allele not in ClinVar today, rsID multi-allelic in export | 16 |

ClinVar records matched to a TRAIN_LABELED/TRAIN_VUS allele, by change:

| label_change_vs_training | n |
|---|---|
| concordant | 593 |
| B_now_Conflicting_classifications_of_pathogenicity | 42 |
| B_now_Uncertain_significance | 11 |
| P_now_Conflicting_classifications_of_pathogenicity | 9 |
| concordant_level_changed | 8 |
| P_now_Uncertain_significance | 4 |
| FLIP_B_to_P | 3 |
| VUS_resolved_to_B | 3 |
| VUS_resolved_to_P | 3 |
| FLIP_P_to_B | 1 |

### P<->B flips and resolved VUS (all explained below)

| gene | rsids | hgvs_c | hgvs_p | training_label_raw | training_label | clinvar_classification | review_stars | label_change_vs_training | other ClinVar alleles at the same training rsID |
|---|---|---|---|---|---|---|---|---|---|
| SLC19A2 |  | c.1134T>G | p.Tyr378Ter | likely-benign | likely-benign | Pathogenic | 1 | FLIP_B_to_P | c.1134T>C p.Tyr378= = Likely benign |
| SLC19A3 | 368364131 | c.1252C>G | p.Gln418Glu | likely-pathogenic;benign-likely-benign;uncertain-significance | uncertain-significance | Benign/Likely benign | 2 | VUS_resolved_to_B |  |
| SLC19A3 | 200542114 | c.980-14A>T |  | pathogenic-likely-pathogenic;likely-benign;pathogenic | likely-pathogenic | Likely benign | 1 | FLIP_P_to_B | c.980-14A>G = Pathogenic/Likely pathogenic |
| SLC19A3 | 137852958 | c.958G>C | p.Glu320Gln | likely-benign | likely-benign | Pathogenic | 0 | FLIP_B_to_P | c.958G>T p.Glu320Ter = Likely pathogenic |
| SLC19A3 | 1474448985 | c.894T>G | p.Tyr298Ter | pathogenic;likely-benign | uncertain-significance | Pathogenic | 2 | VUS_resolved_to_P | c.894T>C p.Tyr298= = Likely benign |
| SLC19A3 |  | c.861C>A | p.Phe287Leu | uncertain-significance | uncertain-significance | Likely benign | 1 | VUS_resolved_to_B | c.861C>T p.Phe287= = Conflicting classifications of pathogenicity |
| SLC19A3 | 2106328711 | c.816C>A | p.Cys272Ter | pathogenic;likely-benign | uncertain-significance | Pathogenic | 1 | VUS_resolved_to_P | c.816C>T p.Cys272= = Likely benign |
| SLC19A3 | 1353757275 | c.513C>A | p.Tyr171Ter | likely-benign;pathogenic | uncertain-significance | Pathogenic/Likely pathogenic | 2 | VUS_resolved_to_P | c.513C>T p.Tyr171= = Likely benign |
| SLC19A3 |  | c.183C>G | p.Tyr61Ter | likely-benign | likely-benign | Pathogenic | 1 | FLIP_B_to_P | c.183C>T p.Tyr61= = Likely benign |
| SLC19A3 | 1272289806 | c.151-17A>G |  | uncertain-significance | uncertain-significance | Likely benign | 1 | VUS_resolved_to_B |  |

All four P<->B flips are artifacts of the per-rsID label copy: the training row inherited the label of a sibling allele at the same rsID (e.g. the nonsense c.1134T>G p.Tyr378Ter was trained as likely-benign because the synonymous c.1134T>C is LB; the intronic c.980-14A>T was trained as likely-pathogenic because c.980-14A>G is P/LP). The same mechanism produced the `pathogenic;likely-benign` composites that were mapped to VUS. c.958G>C p.Glu320Gln is a 0-star 2009 Pathogenic record; no ClinVar allele at rs137852958 supports the training LB label. Downgrades (P/LP or B/LB at training, now VUS/conflicting): see `clinvar_label_changes_vs_training.csv`.

## 5. Scoring the stored Sep-11 and Sep-22 predictions on the time-split

Only `TRAIN_UNLABELED` variants have stored predictions (VUS rows and ABSENT variants were never in the prediction pool): **37 variants (11 P/LP, 26 B/LB)**, all >=1 star. Positive class = P/LP. Multiclass outputs: predicted P or LP -> 1; LB/B/Normal -> 0. The multiclass files store only the top-class confidence, so their AUC is a *proxy* (score = 0.5 +/- 0.5*confidence by predicted side); the Sep-22 binary file has a real `Pathogenic_Probability`.

| model | subset | n | n_pos | n_neg | TN | FP | FN | TP | accuracy | precision | recall | specificity | balanced_acc | MCC | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sep-11 abstract (multiclass, proxy-AUC) | all | 37 | 11 | 26 | 25 | 1 | 6 | 5 | 0.811 | 0.833 | 0.455 | 0.962 | 0.708 | 0.516 | 0.626 |
| Sep-22 FINAL multiclass (proxy-AUC) | all | 37 | 11 | 26 | 23 | 3 | 4 | 7 | 0.811 | 0.700 | 0.636 | 0.885 | 0.760 | 0.536 | 0.720 |
| Sep-22 FINAL binary (Pathogenic_Probability) | all | 37 | 11 | 26 | 23 | 3 | 1 | 10 | 0.892 | 0.769 | 0.909 | 0.885 | 0.897 | 0.760 | 0.930 |
| Sep-11 abstract (multiclass, proxy-AUC) | missense | 2 | 2 | 0 | 0 | 0 | 1 | 1 | 0.500 | 1.000 | 0.500 |  | 0.500 |  |  |
| Sep-22 FINAL multiclass (proxy-AUC) | missense | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 1.000 | 1.000 | 1.000 |  | 1.000 |  |  |
| Sep-22 FINAL binary (Pathogenic_Probability) | missense | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 1.000 | 1.000 | 1.000 |  | 1.000 |  |  |
| Sep-11 abstract (multiclass, proxy-AUC) | >=1 star | 37 | 11 | 26 | 25 | 1 | 6 | 5 | 0.811 | 0.833 | 0.455 | 0.962 | 0.708 | 0.516 | 0.626 |
| Sep-22 FINAL multiclass (proxy-AUC) | >=1 star | 37 | 11 | 26 | 23 | 3 | 4 | 7 | 0.811 | 0.700 | 0.636 | 0.885 | 0.760 | 0.536 | 0.720 |
| Sep-22 FINAL binary (Pathogenic_Probability) | >=1 star | 37 | 11 | 26 | 23 | 3 | 1 | 10 | 0.892 | 0.769 | 0.909 | 0.885 | 0.897 | 0.760 | 0.930 |
| Sep-11 abstract (multiclass, proxy-AUC) | strict temporal (no concordant SCV before 2025-06-27) | 19 | 5 | 14 | 14 | 0 | 3 | 2 | 0.842 | 1.000 | 0.400 | 1.000 | 0.700 | 0.574 | 0.557 |
| Sep-22 FINAL multiclass (proxy-AUC) | strict temporal (no concordant SCV before 2025-06-27) | 19 | 5 | 14 | 14 | 0 | 1 | 4 | 0.947 | 1.000 | 0.800 | 1.000 | 0.900 | 0.864 | 0.800 |
| Sep-22 FINAL binary (Pathogenic_Probability) | strict temporal (no concordant SCV before 2025-06-27) | 19 | 5 | 14 | 12 | 2 | 0 | 5 | 0.895 | 0.714 | 1.000 | 0.857 | 0.929 | 0.782 | 1.000 |
| Sep-11 abstract (multiclass, proxy-AUC) | non-missense | 35 | 9 | 26 | 25 | 1 | 5 | 4 | 0.829 | 0.800 | 0.444 | 0.962 | 0.703 | 0.507 | 0.620 |
| Sep-22 FINAL multiclass (proxy-AUC) | non-missense | 35 | 9 | 26 | 23 | 3 | 4 | 5 | 0.800 | 0.625 | 0.556 | 0.885 | 0.720 | 0.458 | 0.671 |
| Sep-22 FINAL binary (Pathogenic_Probability) | non-missense | 35 | 9 | 26 | 23 | 3 | 1 | 8 | 0.886 | 0.727 | 0.889 | 0.885 | 0.887 | 0.728 | 0.919 |
| Sep-11 abstract (multiclass, proxy-AUC) | SLC19A2 | 17 | 6 | 11 | 11 | 0 | 4 | 2 | 0.765 | 1.000 | 0.333 | 1.000 | 0.667 | 0.494 | 0.561 |
| Sep-22 FINAL multiclass (proxy-AUC) | SLC19A2 | 17 | 6 | 11 | 10 | 1 | 2 | 4 | 0.824 | 0.800 | 0.667 | 0.909 | 0.788 | 0.604 | 0.667 |
| Sep-22 FINAL binary (Pathogenic_Probability) | SLC19A2 | 17 | 6 | 11 | 9 | 2 | 0 | 6 | 0.882 | 0.750 | 1.000 | 0.818 | 0.909 | 0.783 | 1.000 |
| Sep-11 abstract (multiclass, proxy-AUC) | SLC19A3 | 20 | 5 | 15 | 14 | 1 | 2 | 3 | 0.850 | 0.750 | 0.600 | 0.933 | 0.767 | 0.577 | 0.760 |
| Sep-22 FINAL multiclass (proxy-AUC) | SLC19A3 | 20 | 5 | 15 | 13 | 2 | 2 | 3 | 0.800 | 0.600 | 0.600 | 0.867 | 0.733 | 0.467 | 0.680 |
| Sep-22 FINAL binary (Pathogenic_Probability) | SLC19A3 | 20 | 5 | 15 | 14 | 1 | 1 | 4 | 0.900 | 0.800 | 0.800 | 0.933 | 0.867 | 0.733 | 0.893 |

Bootstrap 95% CIs (stratified, 4,000 resamples):

| model | subset | n | AUC_95CI | balanced_acc_95CI |
|---|---|---|---|---|
| Sep-11 abstract (multiclass) | all | 37 | 0.38-0.85 | 0.55-0.86 |
| Sep-22 FINAL multiclass | all | 37 | 0.48-0.92 | 0.60-0.91 |
| Sep-22 FINAL binary | all | 37 | 0.79-1.00 | 0.78-0.98 |
| Sep-11 abstract (multiclass) | strict temporal | 19 | 0.20-0.89 | 0.50-0.90 |
| Sep-22 FINAL multiclass | strict temporal | 19 | 0.40-1.00 | 0.70-1.00 |
| Sep-22 FINAL binary | strict temporal | 19 | 1.00-1.00 | 0.82-1.00 |

Confusion matrices (rows = ClinVar truth, cols = prediction):

- Sep-11 abstract (multiclass, proxy-AUC), all: `[[TN=25, FP=1], [FN=6, TP=5]]`
- Sep-22 FINAL multiclass (proxy-AUC), all: `[[TN=23, FP=3], [FN=4, TP=7]]`
- Sep-22 FINAL binary (Pathogenic_Probability), all: `[[TN=23, FP=3], [FN=1, TP=10]]`
- Sep-11 abstract (multiclass, proxy-AUC), missense: `[[TN=0, FP=0], [FN=1, TP=1]]`
- Sep-22 FINAL multiclass (proxy-AUC), missense: `[[TN=0, FP=0], [FN=0, TP=2]]`
- Sep-22 FINAL binary (Pathogenic_Probability), missense: `[[TN=0, FP=0], [FN=0, TP=2]]`
- Sep-11 abstract (multiclass, proxy-AUC), >=1 star: `[[TN=25, FP=1], [FN=6, TP=5]]`
- Sep-22 FINAL multiclass (proxy-AUC), >=1 star: `[[TN=23, FP=3], [FN=4, TP=7]]`
- Sep-22 FINAL binary (Pathogenic_Probability), >=1 star: `[[TN=23, FP=3], [FN=1, TP=10]]`
- Sep-11 abstract (multiclass, proxy-AUC), strict temporal (no concordant SCV before 2025-06-27): `[[TN=14, FP=0], [FN=3, TP=2]]`
- Sep-22 FINAL multiclass (proxy-AUC), strict temporal (no concordant SCV before 2025-06-27): `[[TN=14, FP=0], [FN=1, TP=4]]`
- Sep-22 FINAL binary (Pathogenic_Probability), strict temporal (no concordant SCV before 2025-06-27): `[[TN=12, FP=2], [FN=0, TP=5]]`

### Per-variant predictions (all scored variants)

| gene | hgvs_c | hgvs_p | csq | clinvar_classification | review_stars | strict_temporal | sep11_label | sep22m_label | sep22b_class | sep22b_prob |
|---|---|---|---|---|---|---|---|---|---|---|
| SLC19A3 | c.-2-4020G>A |  | 5'UTR | Benign | 1 | False | Normal | B | Benign | 0.000 |
| SLC19A2 | c.1224-15_1224-14del |  | intronic | Likely benign | 1 | False | LB | LB | Benign | 0.121 |
| SLC19A2 | c.246G>A | p.Leu82= | synonymous | Likely benign | 1 | False | LB | LP | Benign | 0.322 |
| SLC19A2 | c.27G>A | p.Arg9= | synonymous | Likely benign | 1 | True | LB | LB | Pathogenic | 0.525 |
| SLC19A2 | c.426C>T | p.Tyr142= | synonymous | Likely benign | 1 | False | LB | LB | Benign | 0.325 |
| SLC19A2 | c.807+12C>A |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.226 |
| SLC19A2 | c.807+12_807+13insA |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.381 |
| SLC19A2 | c.807+19A>G |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.236 |
| SLC19A2 | c.807+9A>G |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.202 |
| SLC19A2 | c.918C>T | p.Thr306= | synonymous | Likely benign | 1 | True | LB | LB | Benign | 0.246 |
| SLC19A2 | c.99G>T | p.Pro33= | synonymous | Likely benign | 1 | True | LB | LB | Benign | 0.163 |
| SLC19A2 | c.9G>C | p.Val3= | synonymous | Likely benign | 1 | True | LB | LB | Pathogenic | 0.550 |
| SLC19A3 | c.1011G>A | p.Val337= | synonymous | Likely benign | 1 | False | LP | LP | Pathogenic | 0.756 |
| SLC19A3 | c.1050G>A | p.Val350= | synonymous | Likely benign | 1 | True | LB | LB | Benign | 0.029 |
| SLC19A3 | c.1068T>C | p.Asn356= | synonymous | Likely benign | 1 | False | LB | LB | Benign | 0.127 |
| SLC19A3 | c.1227C>T | p.Ile409= | synonymous | Likely benign | 1 | False | LB | LB | Benign | 0.251 |
| SLC19A3 | c.1314+8G>A |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.332 |
| SLC19A3 | c.150+12T>C |  | intronic | Likely benign | 1 | True | LB | LB | Benign | 0.077 |
| SLC19A3 | c.151-6T>C |  | intronic | Likely benign | 1 | False | LB | LB | Benign | 0.105 |
| SLC19A3 | c.151-8C>T |  | intronic | Likely benign | 1 | False | LB | LB | Benign | 0.058 |
| SLC19A3 | c.363G>A | p.Val121= | synonymous | Likely benign | 1 | False | LB | LB | Benign | 0.187 |
| SLC19A3 | c.465G>A | p.Ser155= | synonymous | Likely benign | 2 | False | LB | LB | Benign | 0.231 |
| SLC19A3 | c.480C>T | p.Leu160= | synonymous | Likely benign | 1 | True | LB | LB | Benign | 0.326 |
| SLC19A3 | c.495G>A | p.Ala165= | synonymous | Likely benign | 2 | True | LB | B | Benign | 0.014 |
| SLC19A3 | c.669T>G | p.Gly223= | synonymous | Likely benign | 1 | True | LB | LB | Benign | 0.109 |
| SLC19A3 | c.980-7T>C |  | intronic | Likely benign | 1 | False | LB | LP | Benign | 0.302 |
| SLC19A2 | c.1224-2A>T |  | splice acceptor | Likely pathogenic | 1 | False | LB | LB | Pathogenic | 0.792 |
| SLC19A2 | c.199dup | p.Arg67fs | frameshift | Likely pathogenic | 1 | True | LB | P | Pathogenic | 0.998 |
| SLC19A2 | c.205G>T | p.Val69Phe | missense | Likely pathogenic | 2 | True | LB | LP | Pathogenic | 0.552 |
| SLC19A3 | c.1172+5G>C |  | intronic | Likely pathogenic | 1 | False | LB | LB | Benign | 0.144 |
| SLC19A3 | c.151-2A>G |  | splice acceptor | Likely pathogenic | 1 | False | LB | LB | Pathogenic | 0.744 |
| SLC19A3 | c.517A>G | p.Asn173Asp | missense | Likely pathogenic | 1 | False | LP | LP | Pathogenic | 0.860 |
| SLC19A2 | c.1031-1G>A |  | splice acceptor | Pathogenic | 1 | True | LB | LB | Pathogenic | 0.734 |
| SLC19A2 | c.1168del | p.Tyr390fs | frameshift | Pathogenic | 1 | False | P | P | Pathogenic | 1.000 |
| SLC19A2 | c.1201_1202del | p.Met401fs | frameshift | Pathogenic | 1 | True | P | P | Pathogenic | 0.999 |
| SLC19A3 | c.191_192insA | p.Val65fs | frameshift | Pathogenic | 1 | True | P | P | Pathogenic | 1.000 |
| SLC19A3 | c.35G>A | p.Trp12Ter | nonsense | Pathogenic/Likely pathogenic | 2 | False | P | P | Pathogenic | 1.000 |

## 6. UniProt (`uniprot_variants.csv`)

UniProtKB/Swiss-Prot has only 7 curated natural variants for these two proteins (3 for O60779, 4 for Q9BZV2; identical to humsavar). 6/7 are already labeled training examples (and concordant with ClinVar). The only external one is **SLC19A2 p.Asp93His (VAR_010249, TRMA, LP/P, PMID 10874303)**: no rsID, no UniProt genomic mapping, not in ClinVar and not in the training export. Its GRCh38 allele was derived from the codon (training reference bases + WT protein): chr1:169477685 C>G = NM_006996 c.277G>C (unique SNV giving Asp->His). It needs fresh model inference. The rest of the UniProt variation service (~1,400 large-scale variants) carries clinical significance imported from ClinVar/Ensembl only, so it adds nothing independent.

| gene | feature_id | hgvs_p | humsavar_category | uniprot_disease | rsid | chrom | pos | ref | alt | genomic_source | training_label | training_category | clinvar_classification | external_eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SLC19A2 | VAR_010249 | p.Asp93His | LP/P | TRMA |  | 1 | 169477685 | C | G | derived from codon (training ref bases + WT protein); single possible SNV |  | ABSENT |  | True |
| SLC19A2 | VAR_010250 | p.Ser143Phe | LP/P | TRMA | rs761957186 | 1 | 169477534 | G | A | UniProt variation service | pathogenic | TRAIN_LABELED | Pathogenic | False |
| SLC19A2 | VAR_010248 | p.Gly172Asp | LP/P | TRMA | rs28937595 | 1 | 169477447 | C | T | UniProt variation service | pathogenic | TRAIN_LABELED | Pathogenic/Likely pathogenic | False |
| SLC19A3 | VAR_025992 | p.Gly23Val | LP/P | BTBGD | rs121917882 | 2 | 227702251 | C | A | UniProt variation service | likely-pathogenic | TRAIN_LABELED | Pathogenic/Likely pathogenic | False |
| SLC19A3 | VAR_061864 | p.Val174Ile | LB/B |  | rs59736804 | 2 | 227699195 | C | T | UniProt variation service | benign | TRAIN_LABELED | Benign | False |
| SLC19A3 | VAR_052405 | p.Val350Ala | LB/B |  | rs34507036 | 2 | 227696012 | A | G | UniProt variation service | benign | TRAIN_LABELED | Benign | False |
| SLC19A3 | VAR_025993 | p.Thr422Ala | LP/P | BTBGD | rs121917884 | 2 | 227688216 | T | C | UniProt variation service | likely-pathogenic | TRAIN_LABELED | Pathogenic | False |

## 7. Caveats

- **Tiny, imbalanced, easy set.** Only 37 variants are scorable with stored predictions; 15/26 B/LB are synonymous and 10/26 intronic, and the P/LP side is mostly frameshift/canonical-splice/nonsense. Missense: 9 eligible in total (7 P/LP, 2 B/LB), of which only 2 (both P/LP) are scorable, so missense specificity/AUC cannot be estimated. CIs are wide (see bootstrap table). These numbers do not test the hard case (missense VUS reclassification).
- **Allele-frequency circularity.** ClinVar submitters use population frequency (ACMG BA1/BS1/PM2) and the model ingests ~70 frequency columns, so agreement on B/LB (and on absent-from-gnomAD P/LP) is partly circular. Most eligible labels are single-submitter 1-star calls from a few clinical labs (table above) that weigh frequency and in silico predictors heavily.
- **Label lag vs true time-split.** 18/37 scorable and 41/70 ABSENT variants already had a concordant ClinVar SCV before 2025-06-27; the model never saw these labels (they were missing from the dbSNP export), but they are not "future" knowledge. Use the `strict_temporal` subset (19 scorable) for a true prospective claim.
- **Per-rsID label copy in the training data.** Labels were applied to every alt allele of an rsID; this created the 4 P<->B flips and the `pathogenic;likely-benign` composites above, and 276 B/LB + 16 P/LP training rows are alleles with no ClinVar record at all today - every one of them belongs to a multi-allelic rsID, i.e. the label was inherited from a sibling allele (276/907 = 30% of the B/LB and 16/133 = 12% of the P/LP training rows). The training export also contains duplicated (snp_id, alleles) rows for multi-allelic indels (e.g. rs776444312, rs11334205).
- **ABSENT and TRAIN_VUS need new inference.** 53 ABSENT small variants (23 B/LB, 30 P/LP), 6 TRAIN_VUS and UniProt D93H have no stored prediction; 17 ABSENT P/LP are exon-level deletions/CNVs that a sequence-window model cannot score.
- **Thresholds.** The binary file calls 12% of the unlabeled pool pathogenic at P>0.5; the multiclass files call 6% (Sep-22) and 1% (Sep-11) P/LP, which explains the higher recall / lower specificity of the binary output. Proxy AUCs for the multiclass files are not comparable to a true probability AUC.
- ClinVar records whose current classification is VUS/conflicting are kept in `clinvar_current.csv` / `clinvar_vs_training_all.csv` but are not in the evaluation.

## 8. Files

- `clinvar_timesplit.csv`: all 718 gene-specific ClinVar P/LP/B/LB variants with `training_category`, `external_eligible`, `strict_temporal`, `is_missense`, `small_variant`, match details, ClinVar dates/history and the stored predictions (`pred_*`). Filter `external_eligible==True` for the external set (113 rows).
- `clinvar_vs_training_all.csv`: same columns for all 1,347 ClinVar records (incl. VUS/conflicting and CNVs).
- `clinvar_label_changes_vs_training.csv`: flips, downgrades, level changes, resolved VUS.
- `clinvar_current.csv`: tidy ClinVar table (incl. per-SCV summary).
- `uniprot_variants.csv`: 7 curated UniProt variants with training/ClinVar status and predictions where available.
- `timesplit_metrics_existing_predictions.csv`: all metric rows (incl. per-gene and non-missense subsets).
- `_work/`: scripts, logs, reduced training table (`train_reduced_both.pkl`), bootstrap CIs.
