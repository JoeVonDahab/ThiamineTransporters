# External validation sets

None of these variants were training examples. Each set has its own provenance
document; the raw downloads stay in `data/external_raw/` (not redistributed).

| Set | Variants | Labels from | Files |
|---|---|---|---|
| **ClinVar time-split** | 97 sequence-level variants (45 P/LP, 52 B/LB) in SLC19A2/SLC19A3 that ClinVar classifies today (query 2026-10-13) but that were unlabeled, VUS, or absent in the June-2025 dbSNP export used for training. Exon-level deletions/CNVs (17) are excluded; the UniProt-only TRMA variant SLC19A2 p.Asp93His is included | ClinVar germline classification | `clinvar_timesplit.csv` (`external_eligible == True`), `CLINVAR_TIMESPLIT.md`, `uniprot_variants.csv` |
| Literature, abstract labels | the 22 SLC19A3 variants of the abstract, 15 labelled; 4 of those are training examples (T422A, W94R, Y113H, E320Q) so a clean 11-variant version is also used | hand labels used for the abstract | `abstract_literature_labels.csv` |
| Literature, uptake rule | 13 single alanine substitutions (SLC19A2 + SLC19A3) | Li et al. 2024 [3H]-thiamine uptake: < 50 % of WT = loss of function, ≥ 75 % = tolerated | `literature_rule_labels.csv`, `literature_variants.csv`, `LITERATURE_SOURCES.md` |
| Literature, binding rule | 10 SLC19A3 substitutions | Gabriel et al. 2024 thiamine Kd: ≥ 5× WT or no binding = loss of function, ≤ 2× = tolerated (K44E excluded: trafficking defect) | `literature_binding_gabriel2024.csv`, `literature_rule_labels.csv` |

`training_label_audit.csv` is not a validation set: it records, for each of the
1,371 labeled training rows, whether its label belongs to its own allele as of
the 2025-06-27 snapshot (see `docs/AUDIT.md` §3).

Feature sets are built with

```bash
python scripts/build_clinvar_timesplit.py     # reuses published features where the variant was in the training export
python scripts/build_literature_sets.py       # embeds the protein changes; no frequency data
```

Caveats: ClinVar submitters use population frequency, so agreement on benign
calls is partly circular with the frequency features; most time-split benign
variants are synonymous or intronic and most pathogenic ones are truncating, so
the set is easier than the missense-only question (9 missense variants). The
literature sets are small (10–15) and measure biochemical function of
engineered substitutions, not clinical pathogenicity.
