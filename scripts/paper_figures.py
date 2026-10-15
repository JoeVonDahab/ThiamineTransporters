"""Figures for the TripleVar manuscript -> reports/figures/ (PDF + 1,000-dpi CMYK TIFF + PNG).

Sized for Clinical and Translational Science: 178 mm (two-column) wide, Arial (or its
metric twin Liberation Sans) at 7-10 pt.

    python scripts/paper_figures.py [--only 1 2 3 4 S1]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from triplevar.audit import masks  # noqa: E402
from triplevar.dataset import FeatureSet  # noqa: E402
from triplevar.ensemble import load_model  # noqa: E402
from triplevar.labels import VUS  # noqa: E402
from triplevar.metrics import Scores  # noqa: E402

OUT = Path("reports/figures")
MM = 1 / 25.4
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TOOL_COLOR = {"TripleVar": BLUE, "CADD": ORANGE, "AlphaMissense": AQUA, "REVEL": YELLOW, "Abstract model": MAGENTA}
TOOL_STYLE = {"TripleVar": "-", "CADD": "--", "AlphaMissense": "-.", "REVEL": ":", "Abstract model": (0, (5, 1, 1, 1))}
LABEL_COLOR = {0: BLUE, 1: ORANGE}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    """Vector PDF, 300-dpi PNG preview and a CMYK TIFF at 1,000 dpi (CTS line-art requirement)."""
    from PIL import Image

    # Exact canvas size (no tight bbox), so the printed width is what the figure was laid out for.
    ink = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if ink.x0 < -0.01 or ink.y0 < -0.01 or ink.x1 > w + 0.01 or ink.y1 > h + 0.01:
        print(f"WARNING {name}: content {ink.bounds} extends beyond the {w:.2f} x {h:.2f} in canvas")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    rgb_tiff = OUT / f"{name}.rgb.tiff"
    fig.savefig(rgb_tiff, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(rgb_tiff) as im:
        im.convert("RGB").convert("CMYK").save(OUT / f"{name}.tiff", compression="tiff_lzw", dpi=(1000, 1000))
    rgb_tiff.unlink()
    plt.close(fig)
    print("wrote", OUT / name)


def panel(ax, letter, x=-0.02):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def roc(ax, y, s, label, color, ls="-"):
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.3, label=f"{label} (AUC {roc_auc_score(y, s):.2f})")


def roc_axes(ax):
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_xlim(-0.01, 1.01), ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_aspect("equal")
    clean(ax)


# ---------------------------------------------------------------------------- Figure 1
def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.8, edgecolor=color, facecolor="white"))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.055), w, 0.055, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0, facecolor=color, alpha=0.16))
    ax.text(x + 0.010, y + h - 0.028, title, fontsize=8, fontweight="bold", va="center", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + 0.010, y + h - 0.080 - i * 0.047, line, fontsize=7, va="top", color=INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=MUTED))


def figure1():
    fig = plt.figure(figsize=(178 * MM, 100 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    for x, letter, title in ((0.0, "a", "Data and labels"), (0.335, "b", "Featurisation and model"), (0.69, "c", "Evaluation")):
        ax.text(x + 0.005, 0.978, letter, fontsize=10, fontweight="bold", va="top")
        ax.text(x + 0.03, 0.975, title, fontsize=9, fontweight="bold", va="top")
    # (a)
    box(ax, 0.005, 0.66, 0.305, 0.27, "dbSNP exports (GRCh38, June 2025)",
        ["SLC19A2 and SLC19A3, all alleles", "one row per alternative allele",
         "+ one wild-type reference row per rsID"], BLUE)
    box(ax, 0.005, 0.33, 0.305, 0.30, "ClinVar-derived labels",
        ["864 labelled alleles of 962 rsIDs", "P 85 · LP 48 · LB 618 · B 113", "507 VUS (not trained on)",
         "audit: 565 labels belong to the allele"], AQUA)
    box(ax, 0.005, 0.03, 0.305, 0.27, "Variant sequences",
        ["mutated genomic window (26–37 kb)", "spliced mRNA from canonical exons", "translated protein (Kozak-aware start)"],
        YELLOW)
    arrow(ax, 0.157, 0.66, 0.157, 0.632)
    arrow(ax, 0.157, 0.33, 0.157, 0.302)
    # (b): three parallel feature blocks joined into the fused vector
    feats = ((0.745, "DNABERT-2  (3,840-d)", ["mutated window, 5 chunks mean-pooled"], BLUE),
             (0.545, "ESM-2 3B  (2,560-d)", ["mutant protein, layer 36, mean-pooled"], ORANGE),
             (0.345, "Population data  (71-d)", ["32 dbSNP studies: frequency, presence,", "summaries, validation flags"], AQUA))
    for y, title, lines, color in feats:
        box(ax, 0.335, y, 0.305, 0.185, title, lines, color)
        ax.plot([0.64, 0.655], [y + 0.0925, y + 0.0925], color=MUTED, linewidth=0.8)
    ax.plot([0.655, 0.655], [0.345 + 0.0925, 0.745 + 0.0925], color=MUTED, linewidth=0.8)
    arrow(ax, 0.655, 0.437, 0.655, 0.268)
    box(ax, 0.335, 0.03, 0.335, 0.235, "Fused 6,471-d → MLP → 5 classes",
        ["2048-1024-512-256, BatchNorm, LeakyReLU", "class-balanced loss; frequency dropout 0.3", "P · LP · LB · B · wild type"], MAGENTA)
    # (c)
    box(ax, 0.69, 0.73, 0.305, 0.20, "Grouped cross-validation",
        ["10 folds by rsID × 5 repeats", "selection on validation folds only"], BLUE)
    box(ax, 0.69, 0.50, 0.305, 0.20, "ClinVar time-split (external)",
        ["97 variants classified after the snapshot", "45 P/LP · 52 B/LB"], ORANGE)
    box(ax, 0.69, 0.27, 0.305, 0.20, "Benchmark and functional data",
        ["CADD · AlphaMissense · REVEL", "uptake (13) and binding (10) assays"], AQUA)
    box(ax, 0.69, 0.03, 0.305, 0.21, "Prioritisation",
        ["507 ClinVar VUS", "28,764 unlabelled dbSNP alleles", "ranked P/LP candidates"], YELLOW)
    save(fig, "Figure1_overview")


# ---------------------------------------------------------------------------- Figure 2
ABLATIONS = [
    ("abl_dna_protein_freq", "DNA + protein + population", BLUE, "-"),
    ("abl_protein_freq", "Protein + population", ORANGE, "--"),
    ("abl_dna_protein", "DNA + protein", AQUA, "-."),
    ("abl_protein_only", "Protein", YELLOW, ":"),
    ("abl_dna_freq", "DNA + population", MAGENTA, (0, (5, 1, 1, 1))),
    ("abl_freq_only", "Population", MUTED, (0, (1, 1))),
    ("abl_dna_only", "DNA", INK2, (0, (3, 1))),
]


def figure2():
    labeled = FeatureSet.load("data/processed/labeled")
    _, trusted = masks(labeled, "validation/external/training_label_audit.csv")
    keep = labeled.y != VUS
    tr = trusted[keep]
    fig = plt.figure(figsize=(178 * MM, 80 * MM))
    ax1 = fig.add_axes([0.075, 0.14, 0.36, 0.80])
    ax2 = fig.add_axes([0.665, 0.14, 0.29, 0.80])
    for name, label, color, ls in ABLATIONS[:5]:
        oof = pd.read_csv(f"outputs/ablation/{name}.oof.csv")
        assert len(oof) == keep.sum() and np.array_equal(oof["y"].to_numpy(), labeled.y[keep])
        s = Scores(oof[["p_P", "p_LP", "p_LB", "p_B", "p_normal"]].to_numpy())
        m = tr & np.isin(oof["y"].to_numpy(), [-2, -1, 1, 2])
        roc(ax1, np.isin(oof["y"].to_numpy()[m], [-2, -1]), s.pathogenic_given_variant[m], label, color, ls)
    roc_axes(ax1)
    ax1.legend(loc="lower right", frameon=False, handlelength=2.0, handletextpad=0.4, labelspacing=0.3,
               borderaxespad=0.1, fontsize=7)
    panel(ax1, "a")
    rows = []
    for name, label, color, _ in ABLATIONS:
        per = json.loads(Path(f"outputs/ablation/{name}.per_seed.json").read_text())
        vals = [r["test_clean"]["PLP_vs_BLB"]["auc_renormalised"] for r in per]
        rows.append((label, vals, color))
    rows.sort(key=lambda r: np.mean(r[1]))
    for i, (label, vals, color) in enumerate(rows):
        ax2.scatter(vals, [i] * len(vals), s=14, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax2.plot([np.mean(vals)] * 2, [i - 0.28, i + 0.28], color=INK, linewidth=1.0, zorder=4)
        ax2.text(max(vals) + 0.008, i, f"{np.mean(vals):.3f}", va="center", fontsize=7, color=INK2)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlabel("ROC-AUC, P/LP vs B/LB (trusted labels)")
    ax2.set_xlim(0.5, 0.97)
    clean(ax2)
    ax2.grid(False, axis="y")
    panel(ax2, "b")
    save(fig, "Figure2_cross_validation")


# ---------------------------------------------------------------------------- Figure 3
CONSEQUENCE_GROUPS = [
    ("frameshift", "Frameshift"), ("nonsense", "Nonsense"), ("splice", "Splice site"),
    ("missense", "Missense"), ("synonymous", "Synonymous"), ("intron|UTR|upstream", "Intronic / UTR"),
]


def consequence_group(text):
    import re

    for pattern, name in CONSEQUENCE_GROUPS:
        if re.search(pattern, str(text)):
            return name
    return "Other"


def figure3():
    ts = pd.read_csv("reports/predictions/clinvar_timesplit.triplevar_v1.csv")
    y = ts["label_binary"].astype(int).to_numpy() == 1
    bench = pd.read_csv("validation/benchmarks/benchmark_scores.csv")
    fig = plt.figure(figsize=(178 * MM, 165 * MM))
    ax1 = fig.add_axes([0.075, 0.565, 0.36, 0.385])
    ax2 = fig.add_axes([0.62, 0.565, 0.24, 0.35])
    ax3 = fig.add_axes([0.075, 0.065, 0.36, 0.385])
    ax4 = fig.add_axes([0.585, 0.065, 0.36, 0.385])
    abstract = load_model("models/triplevar_abstract_2025-09-11.pt")
    fs = FeatureSet.load("data/processed/clinvar_timesplit")
    p_abs = Scores(abstract.predict_proba(fs.X(abstract.layout))).pathogenic_given_variant
    roc(ax1, y, ts["p_pathogenic"], f"TripleVar, n={len(y)}", BLUE, "-")
    roc(ax1, y, p_abs, f"Abstract model, n={len(y)}", MAGENTA, TOOL_STYLE["Abstract model"])
    roc_axes(ax1)
    ax1.legend(loc="lower right", frameon=False, handlelength=2.6)
    ax1.set_title("ClinVar time-split", loc="left")
    panel(ax1, "a")

    cv = bench[bench["set"] == "cv"]
    c = cv.dropna(subset=["cadd_phred"])
    yc = c["label_binary"].astype(int) == 1
    roc(ax3, yc, c["triplevar_score"], "TripleVar (out of fold)", BLUE, "-")
    roc(ax3, yc, c["cadd_phred"], "CADD v1.7", ORANGE, "--")
    roc_axes(ax3)
    ax3.legend(loc="lower right", frameon=False, handlelength=2.6)
    ax3.set_title(f"Labeled alleles scored by CADD (n={len(c)}; {int(yc.sum())} P/LP)", loc="left")
    panel(ax3, "c")

    mis = cv[cv["is_missense"].astype(str).str.lower().eq("true")].dropna(subset=["alphamissense", "revel"])
    ym = mis["label_binary"].astype(int) == 1
    roc(ax4, ym, mis["triplevar_score"], "TripleVar (out of fold)", BLUE, "-")
    roc(ax4, ym, mis["alphamissense"], "AlphaMissense", AQUA, "-.")
    roc(ax4, ym, mis["revel"], "REVEL", YELLOW, ":")
    roc_axes(ax4)
    ax4.legend(loc="lower right", frameon=False, handlelength=2.6)
    ax4.set_title(f"Labeled missense alleles (n={len(mis)}; {int(ym.sum())} P/LP)", loc="left")
    panel(ax4, "d")

    ts["group"] = ts["consequence"].map(consequence_group)
    order = [g for _, g in CONSEQUENCE_GROUPS if (ts["group"] == g).any()]
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        sub = ts[ts["group"] == g]
        for lab in (0, 1):
            s = sub[sub["label_binary"] == lab]
            if s.empty:
                continue
            off = -0.17 if lab == 0 else 0.17
            jitter = rng.uniform(-0.09, 0.09, len(s))
            correct = s["pathogenic_call"].astype(str).str.lower().eq("true").to_numpy() == (lab == 1)
            ax2.scatter(s["p_pathogenic"][correct], (i + off + jitter)[correct], s=13, color=LABEL_COLOR[lab],
                        edgecolor="white", linewidth=0.5, zorder=3)
            ax2.scatter(s["p_pathogenic"][~correct], (i + off + jitter)[~correct], s=16, facecolor="white",
                        edgecolor=LABEL_COLOR[lab], linewidth=0.9, zorder=3)
        n_p, n_b = int((sub.label_binary == 1).sum()), int((sub.label_binary == 0).sum())
        ax2.text(1.03, i, f"{n_p} P/LP\n{n_b} B/LB", va="center", fontsize=7, color=INK2, linespacing=1.1,
                 transform=ax2.get_yaxis_transform())
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.set_xlabel("TripleVar P(pathogenic or likely pathogenic)")
    clean(ax2)
    ax2.grid(False, axis="y")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=ORANGE, label="P/LP, called pathogenic"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white", markeredgecolor=ORANGE, label="P/LP, missed"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=BLUE, label="B/LB, called benign"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white", markeredgecolor=BLUE, label="B/LB, called pathogenic"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.525, 0.985), ncol=2, frameon=False,
               columnspacing=1.0, handletextpad=0.2, borderaxespad=0.0, fontsize=7)
    ax1.apply_aspect()
    a_box = ax1.get_position()
    fig.text(0.505, a_box.y1 + 0.02 * a_box.height, "b", fontsize=10, fontweight="bold", va="bottom", ha="right")
    save(fig, "Figure3_clinvar_timesplit")


# ---------------------------------------------------------------------------- Figure 4
def protein_class(meta, wt):
    out = []
    for prot, gene, fc in zip(meta["protein_sequence"].astype(str), meta["gene"].astype(str), meta["function_class"].astype(str)):
        w = wt["SLC19A2"] if "SLC19A2" in gene else wt["SLC19A3"]
        if prot == w:
            if "synonymous" in fc:
                out.append("Synonymous")
            elif "UTR" in fc:
                out.append("UTR")
            elif "upstream" in fc or "downstream" in fc:
                out.append("Flanking")
            else:
                out.append("Intronic")
        elif len(prot) < 0.95 * len(w):
            out.append("Protein-truncating")
        elif len(prot) == len(w):
            out.append("Missense")
        else:
            out.append("Other protein-altering")
    return np.array(out)


def figure4():
    from triplevar.reference import load_reference

    wt = {g: load_reference(g, "data/reference").wt_protein for g in ("SLC19A2", "SLC19A3")}
    model = load_model("models/triplevar_v1.pt")
    labeled = FeatureSet.load("data/processed/labeled")
    vus = labeled.subset(labeled.y == VUS)
    pool = FeatureSet.load("data/processed/pool")
    pool = pool.subset((pool.meta["clinical_significance"].astype(str) != "normal").to_numpy())
    rows = []
    for source, fs in (("ClinVar VUS", vus), ("Unlabelled dbSNP", pool)):
        s = Scores(model.predict_proba(fs.X(model.layout)))
        # Same call rule as every reported metric: argmax, with a reference-class argmax resolved by P vs B mass.
        rows.append(pd.DataFrame({"source": source, "p": s.pathogenic_given_variant,
                                  "call": s.call_grouped(), "cls": protein_class(fs.meta, wt)}))
    df = pd.concat(rows, ignore_index=True)
    order = ["Protein-truncating", "Missense", "Other protein-altering", "Synonymous", "Intronic", "UTR", "Flanking"]
    order = [o for o in order if (df.cls == o).any()]
    fig = plt.figure(figsize=(178 * MM, 75 * MM))
    ax = fig.add_axes([0.185, 0.155, 0.505, 0.815])
    data = [df.loc[df.cls == o, "p"].to_numpy() for o in order]
    parts = ax.violinplot(data, positions=range(len(order)), vert=False, widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE), b.set_edgecolor("none"), b.set_alpha(0.35)
    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK2, linewidth=1.6)
        ax.scatter([med], [i], s=10, color="white", edgecolor=INK, linewidth=0.7, zorder=3)
        sub = df[df.cls == order[i]]
        ax.text(1.03, i, f"n = {len(sub):,}   P/LP calls {int(sub.call.sum()):,} ({100 * sub.call.mean():.1f}%)",
                va="center", fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("TripleVar P(pathogenic or likely pathogenic)")
    clean(ax)
    ax.grid(False, axis="y")
    save(fig, "Figure4_prioritisation")
    df.groupby(["source", "cls"]).agg(n=("p", "size"), plp_calls=("call", "sum"), median_p=("p", "median")).to_csv(OUT / "Figure4_counts.csv")


# ---------------------------------------------------------------------------- Figure S1
def figureS1():
    """Evaluation protocol: row-level split (abstract) vs rsID-grouped CV, same model family."""
    fig = plt.figure(figsize=(120 * MM, 62 * MM))
    ax = fig.add_axes([0.30, 0.2, 0.62, 0.76])
    entries = []
    ap = json.loads(Path("reports/experiments/abstract_protocol.summary.json").read_text())["metrics"]
    entries.append(("Row-level split\n(abstract protocol)", ap["test.PLP_vs_BLB.auc_renormalised"]["mean"], ap["test.PLP_vs_BLB.auc_renormalised"]["sd"], MAGENTA))
    for name, label, color in (("s01_big_legacy", "Grouped CV, abstract\nhyperparameters", ORANGE),
                               ("s15_big_legacy_freqdrop_cleanlabels", "Grouped CV,\nTripleVar v1 settings", BLUE)):
        m = json.loads(Path(f"reports/experiments/search/{name}.summary.json").read_text())["metrics"]
        entries.append((label, m["test.PLP_vs_BLB.auc_renormalised"]["mean"], m["test.PLP_vs_BLB.auc_renormalised"]["sd"], color))
    for i, (label, mean, sd, color) in enumerate(entries):
        ax.errorbar(mean, i, xerr=sd, fmt="o", color=color, markersize=4, capsize=2, linewidth=0.9)
        ax.text(mean + sd + 0.004, i, f"{mean:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.set_xlim(0.83, 0.94)
    ax.set_xlabel("ROC-AUC, P/LP vs B/LB, all labels (mean ± SD over repeats)")
    clean(ax)
    ax.grid(False, axis="y")
    save(fig, "FigureS1_protocol")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    figs = {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "S1": figureS1}
    for k, f in figs.items():
        if not args.only or k in args.only:
            f()
