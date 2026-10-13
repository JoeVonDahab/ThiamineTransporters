"""Compact comparison of search runs (outputs/search/*.summary.json).

The first column (mean validation AUC) is the selection criterion; the other
columns are out-of-fold test and external results, shown for information.
"""

import argparse
import json
from pathlib import Path

COLS = [
    ("val AUC", "val_auc"),
    ("CV P/LP-B/LB AUC", "test.PLP_vs_BLB.auc_renormalised"),
    ("CV acc", "test.PLP_vs_BLB.accuracy"),
    ("CV rec", "test.PLP_vs_BLB.recall"),
    ("CV spec", "test.PLP_vs_BLB.specificity"),
    ("CV P-vs-B AUC", "test.strict_P_vs_B.auc"),
    ("trusted AUC", "test_clean.PLP_vs_BLB.auc_renormalised"),
    ("time-split AUC", "external:clinvar_timesplit.renormalised_gt_0.5.auc"),
    ("time-split acc", "external:clinvar_timesplit.renormalised_gt_0.5.accuracy"),
    ("lit(abstract) AUC", "external:literature.renormalised_gt_0.5.auc"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="outputs/search")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.dir).glob("*.summary.json")):
        s = json.loads(p.read_text())
        m = s["metrics"]
        rows.append([s["config"]] + [m[k]["mean"] if k in m else float("nan") for _, k in COLS])
    head = ["config"] + [c for c, _ in COLS]
    if args.markdown:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join([r[0]] + [f"{v:.3f}" for v in r[1:]]) + " |")
    else:
        print(f"{'config':36s}" + "".join(f"{c:>18s}" for c in head[1:]))
        for r in rows:
            print(f"{r[0]:36s}" + "".join(f"{v:18.3f}" for v in r[1:]))


if __name__ == "__main__":
    main()
