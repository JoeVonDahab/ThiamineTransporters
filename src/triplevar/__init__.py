"""TripleVar: multimodal variant effect prediction for SLC19A2 and SLC19A3.

Three input modalities are fused for every variant:
DNABERT-2 embeddings of the mutated genomic window, ESM-2 embeddings of the
translated protein, and population allele-frequency features from dbSNP.
"""

__version__ = "1.0.0"
