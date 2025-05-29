# Variant Dataset – Final Version Used for Model Training

**Purpose**

This markdown describes the exact content of the finalized DataFrame that was fed into the SLC19A3 variant-effect model.
**This is the final dataset that has been used to train the model.**

## 1. Model Input Features

The following columns were used to train the model, encompassing both sequence-based information and quantitative variant characteristics:

* **`generated_dna`**: str 
    * **Brief Description**: Simulated DNA icnldues the whole gene dna + 2000 bp upstream and 1000 bp downstream. If the row label `clinical_significance` is 'Normal', the sequence represents the normal sequence to provide contrast-learning negatives.
* **`protein_sequence`**: str 
    * **Brief Description**: In-silico translated protein segment reflecting the amino-acid change introduced by the variant. For synonymous or intronic changes this is identical to reference protein.

### Additional Model Input Features: Allele Frequencies & Validation Flags

These columns provide quantitative data about variant frequencies and validation status, also used as model inputs:

**Cohort-Specific Allele Frequencies:**
Each cohort contributes two columns:
* `<STUDY>_freq` — Minor-allele frequency (a number between 0 and 1, usually very low, representing the frequency of the variation).
* `<STUDY>_is_mentioned` — Boolean indicating presence of an entry in the published dataset (`1` if mentioned, `0` if not).

**Included Cohorts:**
`PRJEB37584`, `GoNL`, `GnomAD_genomes`, `TOPMED`, `TWINSUK`, `ALSPAC`, `Siberian`, `SGDP_PRJ`, `GENOME_DK`, `GoESP`, `Chileans`, `PharmGKB`, `HGDP_Stanford`, `HapMap`, `NorthernSweden`, `Korea4K`, `GnomAD_exomes`, `Vietnamese`, `ALFA`, `Daghestan`, `GENOGRAPHIC`, `Korea1K`, `PAGE_STUDY`, `Estonian`, `TOMMO`, `PRJEB36033`, `1000Genomes_30X`, `1000Genomes`, `Qatari`, `KOREAN`, `FINRISK`, `ExAC`, `MGP`

**Aggregated Frequency Statistics:**

| Column          | Description                                    |
| :-------------- | :--------------------------------------------- |
| `Avg_Frequency` | Mean of available cohort frequencies (non-zero). |
| `Max_Frequency` | Maximum observed MAF across cohorts.           |
| `Min_Frequency` | Minimum non-zero MAF across cohorts.           |
| `Total_Studies` | Count of cohorts in which the variant was seen with a non-zero frequency. |

**Convenience Flags:**

| Column         | Logic                                                                                                                                              |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| `by-frequency` | `1` if `Avg_Frequency` ≥ 0.01, else `0`.                                                                                                           |
| `by-alfa`      | `1` if `ALFA_is_mentioned` is True, else `0`.                                                                                                    |
| `by-cluster`   | `1` if assigned a cluster label from unsupervised embedding of sequence features (k-means). |

## 2. Variant Identifiers & Basic Annotations (metadata only)

These columns contain fundamental identifying and descriptive information about each variant, but **were not used as direct model inputs**; they are included for reference and downstream analysis.

| Column                  | Meaning                                                                    |
| :---------------------- | :------------------------------------------------------------------------- |
| `#chr`                  | Genomic coordinate on GRCh38.                                              |
| `pos`                   | Genomic coordinate on GRCh38.                                              |
| `variation`             | HGVS-like gDNA notation (e.g., `c.123A>G`).                                |
| `variant_type`          | SNV / InDel / MNV.                                                         |
| `snp_id`                | dbSNP                                                 |
| `clinical_significance` | ClinVar label (Pathogenic / Likely-Pathogenic / Likely-Benign/ Benign / Uncertain / Normal). |
| `validation_status`     | Original experimental confirmation flag string (e.g., Sanger-validated).   |
| `function_class`        | Effect relative to transcript: missense, nonsense, splice-site, synonymous, intronic, UTR etc. |
| `gene`                  | Always SLC19A3 but kept for join consistency.                              |
| `frequency`             | Original raw string of cohort-specific allele frequencies (e.g., G:0.039736:199:1000Genomes). |
| `original`              | Reference nucleotide (1-letter).                                           |
| `mutant`                | Alternate nucleotide (1-letter).                                           |
| `generated_mRNA`        | In-silico generated mRNA sequence.                                         |

## 3. Reproducibility Notes

* the dataset needs to be loaded using the Training & Prediciton.ipynb file. Some preparation for the dataset needs to be done. please refere to the notebook