# External literature validation set — SLC19A3 / SLC19A2 functional variants

Provenance record for `validation/external/literature_variants.csv` (58 rows; 29 with a measured
% -of-WT activity, 29 `NOT_DETERMINED`).
Built 2026-10-12. Every number traced to a Source Data cell or a figure legend; nothing read off a bar chart.

**Quick use:** filter `activity_pct_wt` non-empty → 29 transport rows (4 of them WT/EV controls) →
25 scorable alanine-scan/combined variants. Everything else is present for provenance and to document
what the papers did *not* measure.

---

## 1. Papers

| # | Citation | DOI | PMCID |
|---|---|---|---|
| **Gabriel 2024** | Gabriel F, Spriestersbach L, Fuhrmann A, Jungnickel KEJ, Mostafavi S, Pardon E, Steyaert J, Löw C. **Structural basis of thiamine transport and drug recognition by SLC19A3.** *Nat Commun* 2024;15:8542 | `10.1038/s41467-024-52872-8` | PMC11447181 |
| **Li 2024** | Li P, Zhu Z, Wang Y, Zhang X, Yang C, Zhu Y, Zhou Z, Chao Y, Long Y, Gao Y, Liu S, Zhang L, Gao P, Qu Q. **Substrate transport and drug interaction of human thiamine transporters SLC19A2/A3.** *Nat Commun* 2024;15:10924 | `10.1038/s41467-024-55359-8` | PMC11686366 |

Both CC-BY open access. Files in `data/external_raw/literature/`:

```
gabriel_2024/  gabriel_2024_NatCommun_SLC19A3.pdf        article (13 pp)
               gabriel_SupplementaryInformation.pdf      Suppl. Figs 1-26 + Table 1 (32 pp)
               g_MOESM9_SourceData.xlsx                  SOURCE DATA (103 MB; sheets Fig. 1-3, S1-S5, S14, S19-S21, S26)
               gabriel_SupplementaryData1.xlsx           constructs / primers / MS raw data
               gabriel_PeerReviewFile.pdf, gabriel_ReportingSummary.pdf,
               gabriel_DescriptionOfAdditionalSuppFiles.docx, g_MOESM4-6_movie.mp4
li_2024/       li_2024_NatCommun_SLC19A2_A3.pdf          article (13 pp)
               li_SupplementaryInformation.pdf           Suppl. Figs 1-13 (15 pp)
               li_SourceData_bundle.zip                  MOESM4 (86 MB; Source Data + MD simulation inputs)
               source_data/li_SourceData.xlsx            SOURCE DATA (sheets Fig.2, Fig.3, Fig.4, Suppl. Fig.1/9/11)
               source_data/li_SupplementaryData1_constructs.xlsx
               li_PeerReviewFile.pdf, li_ReportingSummary.pdf
               Q9BZV2.fasta (SLC19A3), O60779.fasta (SLC19A2)   UniProt, for residue verification
```

Gabriel's Source Data is **MOESM9**, which is not linked from the rendered article HTML; it was found by
probing `static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-52872-8/MediaObjects/41467_2024_52872_MOESM9_ESM.xlsx`.

---

## 2. The two assays are **not** the same measurement

This is the single most important thing about this dataset.

| | **Li 2024** | **Gabriel 2024** |
|---|---|---|
| What is measured | **transport** — `[3H]`-thiamine accumulation | **binding** — thiamine affinity |
| Method | 5 nM `[3H]`-thiamine, 3 min, HEK293T stable lines, lysed in 0.2 M NaOH, scintillation | nanoDSF thermal shift (F350/F330), purified protein in detergent, thiamine titration 1.95 µM → 2 mM, global fit of ΔTm/Tm,apo |
| Readout | % of WT | Kd,app (µM) |
| Statistics | mean ± **SEM**, n = 3 biologically independent experiments | mean ± **s.d.** from a global fit, n = 3 independent measurements |
| Expression control | lines mCherry-sorted for matched expression (Suppl. Fig. 1e/1g) | SEC + SDS-PAGE of every mutant (Suppl. Fig. 20) |
| Figures used | **Fig. 2g** (SLC19A3), **Fig. 3k** (SLC19A2) | **Fig. 2f**, Suppl. Figs 4, 5b, 19, 20 |

`activity_pct_wt` is populated **only** from Li's uptake data. Gabriel rows carry no `activity_pct_wt`; their
Kd,app, the WT reference (12.7 ± 1.2 µM) and the fold-change are in `notes` as
`Kd_app_uM=…; WT=12.7+/-1.2; fold_vs_WT=…x`. Converting a Kd into a "% activity" would be fabrication —
affinity and turnover are independent, and the papers themselves disagree (see §5, N297A).

### Source-data cell provenance

* **SLC19A3 uptake** → `li_SourceData.xlsx`, sheet **`Fig.2`**, header row 2, replicate rows 3-5.
  The sheet's internal label is `Fig.2h` but the published legend calls the panel **Fig. 2g** — a
  sheet-labelling slip in the deposited file. `figure` records both.
* **SLC19A2 uptake** → sheet **`Fig.3`**, header row 45 (`EV, WT, E50A, …`), replicate rows 46-48, panel **Fig. 3k**.
* **SLC19A3 binding** → `g_MOESM9_SourceData.xlsx`, sheet **`Fig. 2`**, `Panel f`,
  "Apparent dissociation constants…", columns `mean` / `s.d.`, rows `wt` … `Q300A`.
* **Suppl. Fig. 1 / 9 / 11** sheets in Li's Source Data are labelled "Extended Data Fig. 1/8/12" internally
  (preprint numbering); they hold construct controls and drug dose-responses, no point-variant data, so
  they are not used here.
* Li Suppl. Fig. 1e/1g (per-mutant mCherry expression) exists only as a figure image — the per-mutant
  values are **not** in the Source Data. `expression_note` says so rather than inventing numbers.

---

## 3. Residue-identity verification

Every `ref_aa`/`position` was checked against the UniProt canonical sequences
**Q9BZV2** (SLC19A3, 496 aa) and **O60779** (SLC19A2, 497 aa). All 30 SLC19A3 and 13 SLC19A2 positions match.

The WT `protein_sequence` column in the user's `literature.csv` is byte-identical to Q9BZV2 except at the
mutated residue, so the user's numbering is already UniProt-canonical — no offset correction needed.

**SLC19A2 → SLC19A3 equivalence.** Li states explicitly: A2 Tyr74 ≡ A3 Phe56, A2 Leu127 ≡ A3 Val109,
A2 Phe169 ≡ A3 Tyr151, A2 Val313 ≡ A3 Leu296, A2 Asn314 ≡ A3 Asn297. That implies a −18 offset up to
residue ~131 and −17 from ~296 onward, which reproduces all five stated pairs against the UniProt
sequences. The equivalences in `notes` for the other A2 rows (E50≡E32, L53≡L35, W77≡W59, T111≡T93,
W112≡W94, L115≡L97, E128≡E110, Y131≡Y113, Q317≡Q300, I124≡M106) are **inferred from that offset**, not
stated in the paper. `I124A ≡ M106A` is the least safe — the residue identity differs (Ile vs Met).

---

## 4. Judgement calls

1. **One row per (variant, assay, paper).** F56A, N297A and Q300A appear twice (Gabriel binding +
   Li uptake). Do not average them; pick one assay per evaluation. **De-duplicate on
   `activity_pct_wt != ''` to get a clean 27-row transport-only set** (incl. WT/EV controls).
2. **Combined mutants are kept as one row** with the full name in `protein_change`
   (`L35A/I36A/T93A/W94A/L97A/V109A`, `N45Q/N166Q`) and blank `ref_aa`/`position`/`alt_aa`. The six
   individual cage residues and N45Q/N166Q *also* get their own rows with blank activity and a note
   pointing at the combined measurement, so **all 22 of the user's variants resolve to a row**.
   Don't count a combined row and its member rows as independent observations.
3. **`label_functional = NOT_DETERMINED`** wherever `activity_pct_wt` is blank — i.e. for every
   Gabriel binding row, every clinical-only row, every individually-unmeasured cage residue, and I301A.
   The spec's LOF/TOLERATED/AMBIGUOUS rule needs a % activity; it is not applied to Kd data.
4. **`patient_variant`** is TRUE only for alleles in Gabriel Suppl. Fig. 18a (the BTBGD/WLE mutation
   table): K44E, E320Q, T422A, W59R, W94R, Y113H, E320K, I301V. **E32K is FALSE** — it is a designed
   mutation by analogy to SLC19A1-Glu45Lys (methotrexate resistance), not a reported SLC19A3 allele.
   All alanine-scan positions are FALSE.
5. **N297A's Kd (3083 ± 2344 µM) is an extrapolation.** The s.d. is 76 % of the mean because the
   titration never saturates. Treat "~250-fold" as a lower bound on affinity loss, not a point estimate.
6. **E110K's Kd is `N/A` in the Source Data, not zero.** Suppl. Fig. 5b shows two melting transitions,
   neither responding to thiamine up to 2 mM. Recorded as "no detectable binding", `activity_pct_wt`
   blank. It is the most severe binding phenotype in the panel but has no numeric value.
7. **K44E is a trafficking defect, not a binding defect** — it binds thiamine ~6× *tighter* than WT
   (2.1 µM) but is destabilised (ΔTm = −15 °C), unglycosylated and ER-retained. A classifier scored on
   binding alone would call it benign; it is a confirmed WLE allele. Good adversarial case.
8. **WT rows have `activity_err = 0` by construction**, because Li normalises each replicate to its own
   WT. The real assay noise floor is the EV row (7.69 % for SLC19A3, 6.75 % for SLC19A2) — use that,
   not 0 %, when judging whether a mutant is "dead".
9. **Secondary functional claims** are cited in `notes`, not given numbers: T422A impaired uptake
   (Subramanian, Marchant & Said, *Am J Physiol Cell Physiol* 2006;291:851-859) and E320Q impaired
   transport (Dang et al., *Cell Res* 2024;34:458-461). Neither primary paper reproduces those values,
   so no activity figure is recorded for them.

---

## 5. Threshold analysis vs the user's hand labels

### 5.1 The user's labels encode Gabriel's **binding fold-change**, not Li's uptake

Sorting the 10 user-labelled variants that have Gabriel Kd data by fold-change gives perfect,
monotone, gap-separated agreement:

| Variant | Kd,app (µM) | fold vs WT | user label |
|---|---|---|---|
| F56A | 4.0 ± 0.3 | 0.32× | Likely benign |
| N45Q / N166Q (as "gf") | 10.4 ± 0.9 | 0.82× | Likely benign (both) |
| *WT* | *12.7 ± 1.2* | *1.00×* | — |
| Y151F | 17.8 ± 1.8 | 1.41× | Likely benign |
| **— boundary ≈ 2–6× —** | | | |
| E110Q | 85.2 ± 13.1 | 6.71× | Likely Pathogenic |
| E32K | 145.5 ± 26.8 | 11.46× | Likely Pathogenic |
| **— boundary ≈ 12–13× —** | | | |
| T422A | 172.6 ± 35.3 | 13.60× | Pathogenic |
| E320Q | 284.4 ± 44.9 | 22.40× | Pathogenic |
| N297A | 3083 ± 2344 | 242.8× | Pathogenic |
| E110K | no binding | ∞ | Pathogenic |

**Implied rule: `Likely benign` ≤ 1.5× · `Likely Pathogenic` ≈ 5–12× · `Pathogenic` ≥ 13×.**
Zero inversions across 10 variants. This is not a transport-activity cutoff at all.

If you insist on reading affinity as a pseudo-activity (1/fold, which the papers do **not** support):
benign ≥ 71 %, Likely Pathogenic 8.7–14.9 %, Pathogenic ≤ 7.4 % — i.e. an implied cutoff near
**~15–20 %**, far below the 50 %/75 % in the CSV spec.

The remaining 5 labelled variants come from elsewhere:

* **L35A → Pathogenic** from Li's sextuple cage mutant (11.96 % of WT).
* **W59R, W94R, Y113H, E320K → Likely Pathogenic** from BTBGD clinical status plus Gabriel's structural
  argument (an Arg/Lys inserts a positive charge into the thiamine pocket). **No functional measurement
  exists for any of these four in either paper.**

### 5.2 Cutoff sensitivity on the 27 measured transport rows

Labels at the spec rule (LOF <50 / TOLERATED ≥75 / else AMBIGUOUS) vs single cutoffs at 30/50/70 %:

| gene | variant | % WT | ±SEM | spec | c30 | c50 | c70 |
|---|---|---|---|---|---|---|---|
| SLC19A3 | WT | 100.00 | 0.00 | TOLERATED | TOL | TOL | TOL |
| SLC19A3 | EV | 7.69 | 0.79 | LOF | LOF | LOF | LOF |
| SLC19A3 | R29A | 39.10 | 0.89 | LOF | TOL | LOF | LOF |
| SLC19A3 | E32A | 36.45 | 4.75 | LOF | TOL | LOF | LOF |
| SLC19A3 | F56A | 76.77 | 1.34 | TOLERATED | TOL | TOL | TOL |
| SLC19A3 | W59A | 78.25 | 1.22 | TOLERATED | TOL | TOL | TOL |
| SLC19A3 | E110A | 28.53 | 0.69 | LOF | LOF | LOF | LOF |
| SLC19A3 | Y113A | 59.36 | 0.43 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A3 | Y151A | 53.96 | 3.45 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A3 | N297A | 74.62 | 0.36 | AMBIGUOUS | TOL | TOL | TOL |
| SLC19A3 | Q300A | 66.20 | 0.79 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A3 | E320A | 51.56 | 1.07 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A3 | K380A | 36.72 | 0.86 | LOF | TOL | LOF | LOF |
| SLC19A3 | cage ×6 | 11.96 | 0.82 | LOF | LOF | LOF | LOF |
| SLC19A2 | WT | 100.00 | 0.00 | TOLERATED | TOL | TOL | TOL |
| SLC19A2 | EV | 6.75 | 0.46 | LOF | LOF | LOF | LOF |
| SLC19A2 | E50A | 27.46 | 2.32 | LOF | LOF | LOF | LOF |
| SLC19A2 | L53A | 76.29 | 3.14 | TOLERATED | TOL | TOL | TOL |
| SLC19A2 | Y74A | 56.73 | 1.17 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | W77A | 56.16 | 1.16 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | T111A | 69.64 | 1.40 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | W112A | 49.30 | 2.69 | LOF | TOL | LOF | LOF |
| SLC19A2 | L115A | 67.87 | 3.56 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | I124A | 81.92 | 2.27 | TOLERATED | TOL | TOL | TOL |
| SLC19A2 | L127A | 82.75 | 3.91 | TOLERATED | TOL | TOL | TOL |
| SLC19A2 | E128A | 31.59 | 0.89 | LOF | TOL | LOF | LOF |
| SLC19A2 | Y131A | 58.09 | 0.31 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | N314A | 66.01 | 2.88 | AMBIGUOUS | TOL | TOL | LOF |
| SLC19A2 | Q317A | 78.47 | 3.13 | TOLERATED | TOL | TOL | TOL |

Counts (excluding WT/EV controls, n = 25): **spec** 7 LOF / 7 TOLERATED / 11 AMBIGUOUS ·
**c30** 2 LOF / 23 TOLERATED · **c50** 7 LOF / 18 TOLERATED · **c70** 16 LOF / 9 TOLERATED.

The alanine scans cluster in **50–80 %**, exactly where every cutoff is unstable. Nine of 25 variants
(Y113A, Y151A, Q300A, E320A, Y74A, W77A, T111A, L115A, Y131A, N314A) flip between c50 and c70.
**Any single cutoff in this range is close to arbitrary** — these are deliberate binding-site alanine
substitutions, i.e. partial-activity engineering mutants, not a natural benign/pathogenic mixture.
Don't use them as a balanced benchmark. The only unambiguous calls are the EV floor (~7 %), the
sextuple cage mutant (12 %), and the ≥75 % group (F56A, W59A, L53A, I124A, L127A, Q317A).

### 5.3 Rows that look mislabelled

1. **N297A — "Pathogenic" is contradicted by the transport data.** Gabriel: 243× affinity loss
   (the most extreme in the panel). Li: **74.62 ± 0.36 % of WT uptake** — essentially functional.
   The same substitution, measured two ways, gives opposite answers. Reducing thiamine affinity
   ~250-fold barely dents transport at 5 nM substrate, which is itself informative, but it means
   N297A should be **excluded or flagged AMBIGUOUS**, not used as a positive.
2. **E32K — "Likely Pathogenic" for a variant that is not a disease allele.** Functionally defensible
   (11.5× affinity loss; the E32A proxy gives 36.5 % uptake = LOF), but it is a designed mutation.
   Fine as a functional datapoint, wrong as a clinical one.
3. **L35A — "Pathogenic" attributes a sextuple mutant's phenotype to one residue.** The 12 % figure
   belongs to L35A+I36A+T93A+W94A+L97A+V109A together. L35A alone was never measured. By the same
   logic the 6 unlabelled cage residues would also be "Pathogenic", which is why they were presumably
   left blank — the label is inconsistent with that (correct) caution.
4. **W59R / W94R / Y113H / E320K — "Likely Pathogenic" rests on ClinVar-type evidence, not function.**
   These are real BTBGD alleles, so the label is almost certainly right, but there is **no measured
   activity** for any of them. If TripleVar saw ClinVar during training, scoring on these four is
   circular. Note too that the alanine proxies are *mild* (W59A 78.3 %, Y113A 59.4 %) — the pathogenic
   mechanism is the inserted positive charge, which an Ala scan cannot model.
5. **F56A, Y151F, N45Q, N166Q — "Likely benign" is well supported.** F56A: 0.32× Kd *and* 76.8 % uptake.
   Y151F: 1.41× Kd (the Y151A proxy drops to 54 % uptake, so the conservative Phe substitution matters).
   N45Q/N166Q: 0.82× Kd, WT-like SEC — though only ever measured as the double mutant.
6. **K44E (not in the user's 22) is the best adversarial case in the set** — WLE allele, 6× *tighter*
   binding, defect is ER retention. Worth adding to the user's list.

### 5.4 Measurements for the 7 previously unlabelled variants

| variant | what exists | suggested label |
|---|---|---|
| **Q300A** | Li uptake **66.20 ± 0.79 %** (SEM, n=3); Gabriel Kd **5.64 ± 1.36 µM = 0.44× WT** (tighter than WT) | **AMBIGUOUS / lean tolerated.** Only variant of the 7 with a real individual measurement, and the two assays agree it is not badly broken. |
| **I36A** | sextuple only (11.96 %) | NOT_DETERMINED |
| **T93A** | sextuple only | NOT_DETERMINED |
| **W94A** | sextuple only | NOT_DETERMINED |
| **L97A** | sextuple only | NOT_DETERMINED |
| **V109A** | sextuple only; A2 proxy **L127A = 82.75 %** (tolerated) | NOT_DETERMINED (A2 proxy suggests tolerated) |
| **I301A** | **nothing — not made in either paper** | NOT_DETERMINED |

**I301A could not be verified at all.** Neither paper substitutes Ile301. Both describe the residue
(Gabriel: the Ile36–Ile301 apolar contact seals the extracellular gate in the inward-open state,
Suppl. Fig. 17; Li: Asn297/Tyr298/Ile301 on TM7 contact fedratinib). The patient allele there is
**I301V** (infantile-onset BTBGD, Gabriel Suppl. Fig. 18a), also unmeasured. If the user wants I301A
scored, the measurement does not exist in these two papers.

---

## 6. What could not be verified

* **I301A** — not tested anywhere in either paper (see above).
* **W59R, W94R, Y113H, E320K, I301V** — named as BTBGD alleles, never functionally assayed. No activity
  value exists to verify.
* **L35A, I36A, T93A, W94A, L97A, V109A individually** — only the sextuple was measured.
* **N45Q and N166Q individually** — only the `gf` double mutant was measured.
* **E110K numeric Kd** — reported as `N/A`; qualitative "no binding" only.
* **Per-mutant expression levels** — Li's Suppl. Fig. 1e/1g values are not in the Source Data; only the
  statement that lines were mCherry-sorted for matched expression. Gabriel reports SEC/SDS-PAGE per
  mutant (Suppl. Fig. 20) but no quantified expression.
* **Secondary functional numbers** for T422A (Subramanian 2006) and E320Q (Dang 2024) — those papers
  were not downloaded; only the citing sentences were read.
* **SLC19A2 patient variants** — Li's SLC19A2 panel (Fig. 3k) is a pure alanine scan. Neither paper
  tabulates SLC19A2 (thiamine-responsive megaloblastic anaemia) disease alleles, so this set has
  **no SLC19A2 patient variants at all**.
