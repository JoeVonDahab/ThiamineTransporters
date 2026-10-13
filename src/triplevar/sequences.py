"""Build the three sequence views of a variant: genomic window, spliced mRNA, protein.

This is a direct, globals-free port of the logic in the original preparation
notebooks so that the published datasets can be regenerated exactly:

* the mutated genomic window is returned in transcript orientation
  (reverse complement of the + strand window) because both genes are on the
  - strand; this is the string DNABERT-2 embeds,
* the mRNA is the reverse complement of the fused (+ strand) exons, with the
  variant applied when it falls inside an exon,
* the protein is translated from the reference CDS start, falling back to a
  Kozak-context search when a variant destroys or shifts that AUG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from Bio.Seq import Seq

from .genes import GeneConfig

KOZAK_FLANK_NT = 9

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Variant:
    """A variant in dbSNP + strand notation; '-' marks the empty side of an indel."""

    pos: int
    ref: str
    alt: str

    @property
    def is_insertion(self) -> bool:
        return self.ref == "-"

    @property
    def is_deletion(self) -> bool:
        return self.alt == "-"


class GeneReference:
    """Wild-type sequences of one gene, plus helpers that apply variants to them."""

    def __init__(self, gene: GeneConfig, window_plus_strand: str):
        window = window_plus_strand.upper()
        if len(window) != gene.window_length:
            raise ValueError(
                f"{gene.name}: reference window has {len(window)} bp, expected {gene.window_length}"
            )
        self.gene = gene
        self.window = window
        self.exon_sequences: List[str] = [
            window[start - gene.window_start : end - gene.window_start + 1] for start, end in gene.exons
        ]
        self.wt_mrna = self._mrna_from_exons(self.exon_sequences)
        cds = self.wt_mrna[gene.cds_start : gene.cds_end]
        if not cds.startswith("AUG"):
            raise ValueError(f"{gene.name}: reference CDS does not start with AUG")
        self.wt_protein = str(Seq(cds).translate(to_stop=True))
        start = max(0, gene.cds_start - KOZAK_FLANK_NT)
        self.wt_kozak = self.wt_mrna[start : gene.cds_start + 3 + KOZAK_FLANK_NT]
        self.wt_kozak_anchor = min(gene.cds_start, KOZAK_FLANK_NT)

    # -- genomic window -----------------------------------------------------
    def mutate_window(self, variant: Variant) -> Optional[str]:
        """Return the + strand window with the variant applied, or None if the
        reference allele does not match the genome."""
        window = self.window
        i = variant.pos - self.gene.window_start
        if variant.is_insertion:
            if 0 <= i <= len(window):
                return window[:i] + variant.alt + window[i:]
            return None
        ref = variant.ref
        if not ref or not (0 <= i < len(window) and i + len(ref) <= len(window)):
            return None
        if window[i : i + len(ref)] != ref:
            return None
        alt = "" if variant.is_deletion else variant.alt
        return window[:i] + alt + window[i + len(ref) :]

    def transcript_dna(self, variant: Optional[Variant]) -> Optional[str]:
        """Mutated window in transcript orientation (the DNABERT-2 input)."""
        plus = self.window if variant is None else self.mutate_window(variant)
        return None if plus is None else reverse_complement(plus)

    # -- mRNA ---------------------------------------------------------------
    @staticmethod
    def _mrna_from_exons(exon_sequences: List[str]) -> str:
        return reverse_complement("".join(exon_sequences)).replace("T", "U")

    def mrna(self, variant: Optional[Variant]) -> str:
        """Spliced mRNA; unchanged when the variant is intronic/flanking or its
        reference allele does not match the exon."""
        if variant is None:
            return self.wt_mrna
        for k, (start, end) in enumerate(self.gene.exons):
            if start <= variant.pos <= end:
                break
        else:
            return self.wt_mrna
        exon = self.exon_sequences[k]
        rel = variant.pos - start
        if variant.is_insertion:
            mutated = exon[:rel] + variant.alt + exon[rel:]
        elif exon[rel : rel + len(variant.ref)] == variant.ref:
            alt = "" if variant.is_deletion else variant.alt
            mutated = exon[:rel] + alt + exon[rel + len(variant.ref) :]
        else:
            return self.wt_mrna
        exons = list(self.exon_sequences)
        exons[k] = mutated
        return self._mrna_from_exons(exons)

    # -- protein ------------------------------------------------------------
    def protein(self, mrna: str) -> Optional[str]:
        start = find_start_codon(mrna, self.gene.cds_start, self.wt_kozak, self.wt_kozak_anchor)
        if start is None:
            return None
        orf = mrna[start:]
        return str(Seq(orf[: len(orf) - len(orf) % 3]).translate(to_stop=True))

    def sequences(self, variant: Optional[Variant]) -> Tuple[Optional[str], str, Optional[str]]:
        """(transcript-orientation DNA, mRNA, protein) for a variant (None = wild type)."""
        mrna = self.mrna(variant)
        return self.transcript_dna(variant), mrna, self.protein(mrna)


# -- start codon selection ----------------------------------------------------
def score_kozak_context(
    mrna: str, cand_idx: int, wt_kozak: str, wt_anchor: int, reference_idx: int
) -> dict:
    base_m3 = mrna[cand_idx - 3] if cand_idx >= 3 else "N"
    base_p4 = mrna[cand_idx + 3] if cand_idx + 3 < len(mrna) else "N"
    if base_m3 in "GA" and base_p4 == "G":
        label = "Strong"
    elif base_m3 in "GA" or base_p4 == "G":
        label = "Moderate"
    else:
        label = "Weak"

    cand_kozak = mrna[max(0, cand_idx - KOZAK_FLANK_NT) : cand_idx + 3 + KOZAK_FLANK_NT]
    cand_anchor = min(cand_idx, KOZAK_FLANK_NT)
    score = max_score = 0
    for i in range(-KOZAK_FLANK_NT, KOZAK_FLANK_NT + 1):
        weight = 3 if i in (-3, 3) else 2 if i in (-2, -1, 1, 2) else 1
        wt_i, cand_i = wt_anchor + i, cand_anchor + i
        if 0 <= wt_i < len(wt_kozak) and 0 <= cand_i < len(cand_kozak):
            max_score += weight
            if wt_kozak[wt_i] == cand_kozak[cand_i]:
                score += weight
    norm = score / max_score if max_score > 0 else 0.0
    norm += 0.03 if label == "Strong" else 0.01 if label == "Moderate" else 0.0
    dist = cand_idx - reference_idx
    penalty = abs(dist) * (0.002 if dist < 0 else 0.001)
    return {"idx": cand_idx, "score": norm - penalty, "kozak": label}


def find_start_codon(mrna: str, cds_start: int, wt_kozak: str, wt_anchor: int) -> Optional[int]:
    """Reference AUG if intact; otherwise the best nearby AUG by Kozak context."""
    mrna = mrna.upper().replace("T", "U")
    n = len(mrna)
    if cds_start + 3 <= n and mrna[cds_start : cds_start + 3] == "AUG":
        return cds_start

    def score(idx: int) -> dict:
        return score_kozak_context(mrna, idx, wt_kozak, wt_anchor, cds_start)

    # 1) a strong-context AUG within +/-15 nt, in frame-step offsets
    for offset in (-15, -12, -9, -6, -3, 3, 6, 9, 12, 15):
        idx = cds_start + offset
        if 0 <= idx < n - 3 and mrna[idx : idx + 3] == "AUG" and score(idx)["kozak"] == "Strong":
            return idx

    # 2) best-scoring AUG within a +/-90 nt band, if clearly good
    lo, hi = max(0, cds_start - 90), min(n, cds_start + 90 + 3)
    candidates = [
        score(lo + m.start())
        for m in re.finditer("AUG", mrna[lo:hi])
        if lo + m.start() >= 3 and lo + m.start() + 4 < n
    ]
    if candidates:
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        second = candidates[1]["score"] if len(candidates) > 1 else 0
        if best["score"] >= 0.70 or (best["score"] >= 0.60 and best["score"] - second >= 0.05):
            return best["idx"]

    # 3) first downstream AUG with at least moderate context, else best weak one
    weak = []
    pos = cds_start + 1
    while pos < n:
        idx = mrna.find("AUG", pos)
        if idx == -1:
            break
        if idx >= 3 and idx + 4 < n:
            info = score(idx)
            if info["kozak"] in ("Strong", "Moderate"):
                return idx
            weak.append(info)
        pos = idx + 1
    if weak:
        weak.sort(key=lambda c: c["score"], reverse=True)
        return weak[0]["idx"]
    return None


# -- protein-level variants (literature) ---------------------------------------
_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
    "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
    "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter",
}
_PROTEIN_CHANGE = re.compile(r"^(?:p\.)?([A-Z])(\d+)([A-Z*])$")


def parse_protein_change(change: str) -> Tuple[str, int, str]:
    m = _PROTEIN_CHANGE.match(change.strip())
    if not m:
        raise ValueError(f"Not a one-letter missense change: {change!r}")
    return m.group(1), int(m.group(2)), m.group(3)


def codon_genomic_positions(ref: GeneReference, aa_position: int) -> List[int]:
    """+ strand genomic positions of the 3 codon bases (in mRNA 5'->3' order)."""
    offsets = []
    for exon_seq, (start, end) in zip(ref.exon_sequences, ref.gene.exons):
        offsets.append((start, end))
    # mRNA index -> genomic position: walk exons from the highest coordinate (5' end).
    mrna_to_genomic: List[int] = []
    for start, end in sorted(offsets, reverse=True):
        mrna_to_genomic.extend(range(end, start - 1, -1))
    first = ref.gene.cds_start + 3 * (aa_position - 1)
    return [mrna_to_genomic[first + k] for k in range(3)]


def protein_change_to_variant(ref: GeneReference, change: str) -> Variant:
    """Translate a missense change (e.g. 'E320Q') into a genomic variant.

    The single-nucleotide change producing the new amino acid is preferred
    (first codon position first); when none exists the codon is replaced by the
    alternative codon with the fewest differences. Multi-base changes are
    returned as one MNV on the + strand.
    """
    wt_aa, aa_pos, alt_aa = parse_protein_change(change)
    if ref.wt_protein[aa_pos - 1] != wt_aa:
        raise ValueError(
            f"{ref.gene.name} {change}: reference residue {aa_pos} is {ref.wt_protein[aa_pos - 1]}, not {wt_aa}"
        )
    i = ref.gene.cds_start + 3 * (aa_pos - 1)
    wt_codon = ref.wt_mrna[i : i + 3].replace("U", "T")
    best = None
    for b1 in "ACGT":
        for b2 in "ACGT":
            for b3 in "ACGT":
                codon = b1 + b2 + b3
                if str(Seq(codon).translate()) != alt_aa:
                    continue
                diff = [k for k in range(3) if codon[k] != wt_codon[k]]
                key = (len(diff), diff)
                if best is None or key < best[0]:
                    best = (key, codon)
    if best is None:
        raise ValueError(f"No codon encodes {alt_aa!r}")
    codon = best[1]
    positions = codon_genomic_positions(ref, aa_pos)  # descending genomic order
    changed = [k for k in range(3) if codon[k] != wt_codon[k]]
    k0, k1 = min(changed), max(changed)
    # mRNA bases k0..k1 map to genomic positions positions[k1]..positions[k0] (ascending).
    pos = positions[k1]
    ref_plus = reverse_complement(wt_codon[k0 : k1 + 1])
    alt_plus = reverse_complement(codon[k0 : k1 + 1])
    return Variant(pos=pos, ref=ref_plus, alt=alt_plus)
