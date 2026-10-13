"""Print a mean +/- sd table of experiment summaries (outputs/experiments/*.summary.json)."""

import argparse
import json
from pathlib import Path

ROWS = [
    ("ClinVar test P vs B: accuracy", "test.strict_P_vs_B.accuracy"),
    ("ClinVar test P vs B: AUC", "test.strict_P_vs_B.auc"),
    ("ClinVar test P/LP vs B/LB: accuracy", "test.PLP_vs_BLB.accuracy"),
    ("ClinVar test P/LP vs B/LB: precision", "test.PLP_vs_BLB.precision"),
    ("ClinVar test P/LP vs B/LB: recall", "test.PLP_vs_BLB.recall"),
    ("ClinVar test P/LP vs B/LB: specificity", "test.PLP_vs_BLB.specificity"),
    ("ClinVar test P/LP vs B/LB: AUC", "test.PLP_vs_BLB.auc_renormalised"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in args.summaries]
    keys = list(ROWS)
    for k in runs[0]["metrics"]:
        if k.startswith("external:"):
            keys.append((k.replace("external:", "ext ").replace(".", " | "), k))
    header = ["metric"] + [r["config"] for r in runs]
    lines = []
    if args.markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
    else:
        lines.append("\t".join(header))
    for label, key in keys:
        cells = []
        for r in runs:
            m = r["metrics"].get(key)
            cells.append("n/a" if not m or m["n"] == 0 else f"{m['mean']:.3f} ± {m['sd']:.3f}")
        lines.append(("| " + " | ".join([label] + cells) + " |") if args.markdown else "\t".join([label] + cells))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
