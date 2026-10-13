"""Gene-specific constants (GRCh38).

All genomic coordinates are 1-based, inclusive, on the + strand, exactly as in
the dbSNP tabular export. Both genes are on the - strand; the "window" is the
gene plus flanking sequence that every DNA embedding is computed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class GeneConfig:
    name: str
    chromosome: str
    chromosome_accession: str
    gene_start: int
    gene_end: int
    strand: int
    flank_below: int  # bp added below gene_start (3' side for a - strand gene)
    flank_above: int  # bp added above gene_end (5'/upstream side for a - strand gene)
    ensembl_transcript: str
    refseq_mrna: str
    cds_start: int  # 0-based index of the A of AUG in the spliced mRNA
    cds_end: int  # 0-based, exclusive (end of stop codon)
    exons: Tuple[Tuple[int, int], ...]  # (start, end), ascending genomic order
    uniprot: str
    dbsnp_export: str  # file name of the raw dbSNP export

    @property
    def window_start(self) -> int:
        return self.gene_start - self.flank_below

    @property
    def window_end(self) -> int:
        return self.gene_end + self.flank_above

    @property
    def window_length(self) -> int:
        return self.window_end - self.window_start + 1


SLC19A2 = GeneConfig(
    name="SLC19A2",
    chromosome="1",
    chromosome_accession="NC_000001.11",
    gene_start=169463909,
    gene_end=169485970,
    strand=-1,
    flank_below=2000,
    flank_above=2000,
    ensembl_transcript="ENST00000236137",
    refseq_mrna="NM_006996.3",
    cds_start=178,
    cds_end=1672,
    exons=(
        (169463909, 169465977),
        (169468111, 169468252),
        (169468644, 169468836),
        (169469964, 169470186),
        (169477155, 169477757),
        (169485563, 169485944),
    ),
    uniprot="O60779",
    dbsnp_export="SLC19A2_dSNP.txt",
)

SLC19A3 = GeneConfig(
    name="SLC19A3",
    chromosome="2",
    chromosome_accession="NC_000002.12",
    gene_start=227683763,
    gene_end=227718028,
    strand=-1,
    flank_below=1000,
    flank_above=2000,
    ensembl_transcript="ENST00000644224",
    refseq_mrna="NM_025243.4",
    cds_start=88,
    cds_end=1579,
    exons=(
        (227683763, 227687573),
        (227688166, 227688307),
        (227695889, 227696081),
        (227698736, 227699564),
        (227702169, 227702320),
        (227717943, 227718028),
    ),
    uniprot="Q9BZV2",
    dbsnp_export="SLC19A3.txt",
)

GENES: Dict[str, GeneConfig] = {g.name: g for g in (SLC19A2, SLC19A3)}


def get_gene(name: str) -> GeneConfig:
    try:
        return GENES[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown gene {name!r}; expected one of {sorted(GENES)}") from None
