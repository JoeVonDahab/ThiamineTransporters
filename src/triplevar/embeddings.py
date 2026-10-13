"""DNABERT-2 and ESM-2 embeddings, computed exactly as for the published model.

DNA: the transcript-orientation genomic window (26-37 kb) is cut into five
contiguous chunks (the last takes the remainder); each chunk is embedded by
DNABERT-2 and mean-pooled over all tokens (incl. [CLS]/[SEP]); the five 768-d
vectors are concatenated -> 3840-d.

Protein: ESM-2 3B (esm2_t36_3B_UR50D), layer 36, mean over residues
(BOS/EOS excluded) -> 2560-d.

Both models are needed only to featurise *new* variants; they are imported
lazily so the rest of the package runs without transformers/fair-esm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .dataset import FeatureSet
from .features import DNA_DIM, PROTEIN_DIM, tabular_matrix

log = logging.getLogger(__name__)

DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
DNABERT_MAX_POSITIONS = 38000
DNA_CHUNKS = 5
ESM_LAYER = 36


def _device(device: Optional[str]) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class DnaEmbedder:
    def __init__(self, device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(DNABERT_MODEL, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(
                DNABERT_MODEL, trust_remote_code=True, max_position_embeddings=DNABERT_MAX_POSITIONS
            )
        except TypeError:  # newer transformers reject the kwarg; pass it via the config
            from transformers.models.bert.configuration_bert import BertConfig

            config = BertConfig.from_pretrained(DNABERT_MODEL)
            config.max_position_embeddings = DNABERT_MAX_POSITIONS
            model = AutoModel.from_pretrained(DNABERT_MODEL, trust_remote_code=True, config=config)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def _embed_chunk(self, seq: str) -> np.ndarray:
        ids = self.tokenizer(seq, return_tensors="pt")["input_ids"].to(self.device)
        hidden = self.model(ids)[0]
        return hidden[0].mean(dim=0).float().cpu().numpy()

    def embed(self, sequence: str) -> np.ndarray:
        n = len(sequence)
        part = n // DNA_CHUNKS
        if part < 10:
            raise ValueError(f"DNA sequence too short to split into {DNA_CHUNKS} chunks ({n} nt)")
        bounds = [i * part for i in range(DNA_CHUNKS)] + [n]
        return np.concatenate([self._embed_chunk(sequence[bounds[i] : bounds[i + 1]]) for i in range(DNA_CHUNKS)])


class ProteinEmbedder:
    def __init__(self, device: Optional[str] = None):
        import esm

        self.device = _device(device)
        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self._cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed(self, sequence: str) -> np.ndarray:
        if sequence not in self._cache:
            _, _, tokens = self.batch_converter([("seq", sequence)])
            tokens = tokens.to(self.device)
            reps = self.model(tokens, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
            length = int((tokens != self.alphabet.padding_idx).sum(1)[0])
            self._cache[sequence] = reps[0, 1 : length - 1].mean(0).float().cpu().numpy()
        return self._cache[sequence]


def embed_sequences(
    dna: Iterable[str], protein: Iterable[str], device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    dna, protein = list(dna), list(protein)
    dna_model = DnaEmbedder(device)
    X_dna = np.stack([dna_model.embed(s) for s in tqdm(dna, desc="DNABERT-2")]).astype(np.float32)
    del dna_model
    torch.cuda.empty_cache()
    prot_model = ProteinEmbedder(device)
    X_prot = np.stack([prot_model.embed(s) for s in tqdm(protein, desc="ESM-2")]).astype(np.float32)
    assert X_dna.shape[1] == DNA_DIM and X_prot.shape[1] == PROTEIN_DIM
    return X_dna, X_prot


def embed_variant_table(
    input_csv: Union[str, Path], out_prefix: Union[str, Path], device: Optional[str] = None, limit: Optional[int] = None
) -> FeatureSet:
    """CSV with ``transcript_dna`` + ``protein_sequence`` (+ optional tabular columns) -> FeatureSet."""
    table = pd.read_csv(input_csv)
    if limit:
        table = table.head(limit)
    missing = {"transcript_dna", "protein_sequence"} - set(table.columns)
    if missing:
        raise ValueError(f"{input_csv} lacks columns {sorted(missing)}")
    X_dna, X_prot = embed_sequences(table["transcript_dna"], table["protein_sequence"], device)
    meta = table.drop(columns=["transcript_dna"])
    fs = FeatureSet(meta=meta, dna=X_dna, protein=X_prot, tabular=tabular_matrix(table))
    fs.save(out_prefix)
    log.info("embedded %d variants -> %s", len(fs), out_prefix)
    return fs
