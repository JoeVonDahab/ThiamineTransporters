"""Wild-type genomic windows (+ strand FASTA, one per gene)."""

from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

from .genes import GENES, GeneConfig, get_gene
from .sequences import GeneReference, reverse_complement

log = logging.getLogger(__name__)

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def window_path(gene: GeneConfig, reference_dir: Union[str, Path]) -> Path:
    return Path(reference_dir) / f"{gene.name}.window.fa"


def write_fasta(path: Path, header: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [seq[i : i + 80] for i in range(0, len(seq), 80)]
    path.write_text(f">{header}\n" + "\n".join(lines) + "\n")


def read_fasta(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">")).upper()


def fetch_window(gene: GeneConfig, reference_dir: Union[str, Path]) -> Path:
    """Download the + strand window from NCBI nuccore (set NCBI_EMAIL to identify yourself)."""
    params = {
        "db": "nuccore", "id": gene.chromosome_accession, "rettype": "fasta", "retmode": "text",
        "seq_start": gene.window_start, "seq_stop": gene.window_end, "strand": 1, "tool": "triplevar",
    }
    if os.environ.get("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    with urllib.request.urlopen(f"{EFETCH}?{urllib.parse.urlencode(params)}", timeout=120) as r:
        text = r.read().decode()
    seq = "".join(l.strip() for l in text.splitlines() if not l.startswith(">")).upper()
    path = window_path(gene, reference_dir)
    write_fasta(path, f"{gene.chromosome_accession}:{gene.window_start}-{gene.window_end} {gene.name} (+)", seq)
    return path


def export_windows_from_legacy(legacy_root: Union[str, Path], reference_dir: Union[str, Path]) -> None:
    """Recover each window from the published dataset ('normal' rows hold the WT window)."""
    from .legacy import LegacyArtifacts

    df = LegacyArtifacts(legacy_root).combined()
    for gene in GENES.values():
        rows = df[(df["gene"].astype(str).str.contains(gene.name)) & (df["clinical_significance"] == "normal")]
        seqs = rows["generated_dna"].dropna().unique()
        if len(seqs) != 1:
            raise RuntimeError(f"{gene.name}: expected one wild-type window, found {len(seqs)}")
        window = reverse_complement(seqs[0])
        write_fasta(
            window_path(gene, reference_dir),
            f"{gene.chromosome_accession}:{gene.window_start}-{gene.window_end} {gene.name} (+)",
            window,
        )
        log.info("%s window: %d bp", gene.name, len(window))


def load_reference(gene: Union[str, GeneConfig], reference_dir: Union[str, Path]) -> GeneReference:
    gene = get_gene(gene) if isinstance(gene, str) else gene
    path = window_path(gene, reference_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `triplevar reference` first")
    return GeneReference(gene, read_fasta(path))
