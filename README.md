# Project: High Accuracy Prediction of DNA Variants Effects using a Multi-Modal Deep Learning Approach

**Author:** Youssef Abo-Dahab, Pharm.D, M.S Candidate
**Lab:** Giacomini Lab
**Date (from presentation):** May 2, 2025 (Note: Project details updated based on recent information)

## 1. Motivation

* A vast number of human missense variants (over 4 million) are cataloged, yet only a small fraction (~2%) have been clinically classified.
* Specifically for genes like SLC19A2 and SLC19A3, a significant number of known variants lack clinical annotation (e.g., for SLC19A3, ~490 labeled out of 16,000+).
* Traditional prediction tools (e.g., SIFT, PolyPhen-2) often underperform, particularly for rare variants, and may lack a deep contextual understanding of the genetic and protein sequences.

## 2. Research Objective

The primary objectives of this research are to:

* Develop customized prediction models tailored for specific genes of interest (e.g., SLC19A2, SLC19A3) to achieve improved predictability of variant effects.
* Leverage pretrained deep learning models (DNABERT-2 for DNA sequences and ESM-2 for protein sequences) to generate rich, informative embeddings.
* Integrate these multi-modal embeddings (DNA and protein) to capture comprehensive sequence information and potentially extract gene-specific patterns related to clinical significance.
* Build a classifier to accurately categorize variants into classes such as Pathogenic, Likely Pathogenic, Benign, Likely Benign, and Normal.

## 3. Dataset and Preprocessing

* **Data Source:** Initial variant data is sourced from files like `SLC19A2_dSNP.txt` (downloaded from NCBI's dbSNP database), which contains columns such as `#chr`, `pos`, `variation`, `variant_type`, `snp_id`, `clinical_significance`, `validation_status`, `function_class`, `gene`, and `frequency`. Additional data is sourced from ClinVar.
* **Gene Sequence Construction:**
    * For each gene (e.g., SLC19A2, SLC19A3), the complete reference gene sequence is obtained.
    * To capture regulatory regions, 2000 base pairs upstream and 1000 base pairs downstream of the gene are included.
* **Variant Simulation:**
    * For each SNP ID listed in the source file (e.g., `SLC19A2_dSNP.txt`), the specific variation (e.g., A>C) is mapped to its precise location within the constructed gene sequence (reference + flanking regions).
    * A "mutated" version of the gene sequence is generated, incorporating the specific SNP.
    * A "normal" or reference version of the sequence (at that specific variant site, within the context of the full gene+flanking regions) is also maintained for comparison or as a baseline.
* **Protein Sequence Generation:**
    * From both the "mutated" and "normal" DNA sequences, corresponding mRNA sequences are transcribed, and then protein sequences are translated.
    * This process accounts for how the DNA variation might affect the coding region, potentially leading to missense mutations, frameshifts, premature stop codons, or no change in the protein sequence (silent mutations).
* **Input Data for Models:**
    * `generated_dna`: The full DNA sequence (gene + flanking regions) incorporating the specific variant.
    * `protein_sequence`: The translated protein sequence derived from the `generated_dna`.
* **Labels:**
    * Clinical significance of variants is obtained from sources like ClinVar and the input `dSNP.txt` file.
    * These are mapped to numerical values for model training. The primary classes used for the 5-label classification are: Pathogenic (-2), Likely Pathogenic (-1), Likely Benign (1), Benign (2), and Normal (3). "Uncertain Significance" (0) is typically filtered out before training or handled as a separate category if needed.
* **Final Dataset:** Prepared datasets (e.g., `final_ready20250505_171511.pkl`) contain the processed DNA sequences, protein sequences, and their corresponding numerical labels, ready for model training.

## 4. Methodology: Multi-Modal Deep Learning Architecture

The proposed model architecture involves the following key steps:

1.  **DNA Embeddings:**
    * The `generated_dna` sequences are fed into a pretrained **DNABERT-2** model.
    * This model generates DNA embeddings (representations of the DNA sequence). The typical dimension for these embeddings is 3840 (achieved through methods like splitting sequences into parts, embedding each part, and concatenating, or using specific pooling over the sequence tokens).

2.  **Protein Embeddings:**
    * The `protein_sequence` (derived from the variant DNA) is processed by a pretrained **ESM-2** model (e.g., `esm2_t36_3B_UR50D`).
    * This model generates protein embeddings, typically with a dimension of 2560.

3.  **Embedding Scaling and Fusion:**
    * Both DNA and protein embeddings are independently scaled (e.g., using `StandardScaler` fitted on the training set).
    * The scaled DNA embeddings (3840-dimensional) and scaled protein embeddings (2560-dimensional) are concatenated.
    * This results in a fused, multi-modal embedding vector of 6400 dimensions (3840 + 2560).

4.  **Feed-Forward Neural Network (FCNN) Classifier:**
    * The 6400-dimensional fused embedding is passed as input to an FCNN.
    * **Input Layer:** Takes the 6400-dimensional fused embedding.
    * **Hidden Layers:** Consist of multiple layers with linear transformations, activation functions (e.g., ReLU), Batch Normalization, and Dropout for regularization. A common configuration explored is `[Input -> 2048 -> 1024 -> 512 -> Output]`, though other configurations like `[Input -> 512 -> 256 -> Output]` were also considered.
    * **Output Layer:** Transforms the final hidden layer representation to the number of output classes (5 for the distinct categories), followed by a Softmax activation function to produce probabilities for each class.
    * **Output Classes:** Pathogenic, Likely Pathogenic, Benign, Likely Benign, Normal.

5.  **Training and Evaluation:**
    * The model is trained to predict the clinical significance of DNA variants using the fused embeddings.
    * Strategies to handle class imbalance (e.g., ClassBalancedLoss, weighted samplers) are employed during training.
    * Performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrices, for both individual classes and grouped categories.
        * **Grouped Categories:**
            1.  Benign/Likely Benign (original labels: 1, 2)
            2.  Pathogenic/Likely Pathogenic (original labels: -1, -2)
            3.  Others (e.g., Normal, original label: 3)

## 5. Achieved Outcomes

* The multi-modal model achieved **90% accuracy** across the 5 distinct clinical significance labels (Pathogenic, Likely Pathogenic, Benign, Likely Benign, Normal).
* When predictions were grouped into 3 broader categories (Benign/Likely Benign, Pathogenic/Likely Pathogenic, Others/Normal), the model achieved **92% accuracy**.
* These results indicate strong performance in differentiating various classes of DNA variant effects, surpassing preliminary results and demonstrating the effectiveness of the multi-modal approach.

## 6. Project Repository Structure (Example for SLC19A2)

The project is organized with main folders for each gene (e.g., SLC19A2, SLC19A3). An example structure for the `SLC19A2` folder includes:

* `SLC19A2/`
    * `SLC19A2_dSNP.txt`: Original raw variant data downloaded from NCBI's dbSNP database.
    * `exons_data_20250505_171511.csv`: Contains exon coordinates for SLC19A2 (used in data preparation, e.g., for reverse complementing if gene is on negative strand).
    * `Prepating_Data_SLC19A2.ipynb`: Jupyter notebook with Python code for data preprocessing. This script takes the raw `SLC19A2_dSNP.txt` and `exons_data`, generates simulated DNA, RNA, and protein sequences for each SNP, and produces the final dataset.
    * `final_dataset/`
        * `final_ready20250505_171511.csv`: The final prepared dataset in CSV format, containing sequences and labels.
        * `final_ready20250505_171511.pkl`: The final prepared dataset in Python pickle format, used for training.
    * `df_labeled.csv`: A subset of the final dataset containing only labeled data used for training and validation.
    * `csv_labeled_data_cleaned.pkl`: Cleaned version of the labeled data in pickle format.
    * `Training & Prediction.ipynb`: Jupyter notebook containing Python code for model training, evaluation, and prediction on unlabeled data.
    * `models/`
        * `final_model_slc19a21747199044.pt`: The saved, trained PyTorch model file for SLC19A2, achieving the reported accuracies.
    * `results/`
        * `predictions_with_details.csv`: Output file containing predicted labels and probabilities for unlabeled data.

