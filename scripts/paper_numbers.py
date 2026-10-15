"""Every number and table cell quoted in the manuscript -> reports/paper_numbers.json.

    python scripts/paper_numbers.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triplevar.audit import label_source
from triplevar.dataset import FeatureSet
from triplevar.ensemble import load_model
from triplevar.metrics import Scores, binary_report, bootstrap_ci

OUT = Path("reports/paper_numbers.json")


def pct(x):
    return f"{100 * x:.1f}"


def ms(m, key, p=False, nd=3):
    v = m[key]
    return f"{100 * v['mean']:.1f} ± {100 * v['sd']:.1f}" if p else f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f}"


def ci_str(ci):
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


def gene_of(series):
    return np.where(series.astype(str).str.contains("SLC19A2"), "SLC19A2", "SLC19A3")


def binary_row(y, call, score, label, n_boot_seed=0):
    r = binary_report(y, call, score)
    ci = bootstrap_ci(y, score, seed=n_boot_seed) if r.auc is not None else [np.nan, np.nan]
    return {
        "label": label, "n": f"{r.n} ({r.tp + r.fn}:{r.fp + r.tn})",
        "auc": f"{r.auc:.3f} ({ci_str(ci)})" if r.auc is not None else "—",
        "accuracy": pct(r.accuracy), "sensitivity": pct(r.recall), "specificity": pct(r.specificity),
        "precision": pct(r.precision) if (r.tp + r.fp) else "—",
        "_raw": r.as_dict(), "_ci": ci,
    }


def main() -> None:
    nums = {}
    labeled = FeatureSet.load("data/processed/labeled")
    meta = labeled.meta.assign(y=labeled.y, gene2=gene_of(labeled.meta["gene"]))
    src = label_source(labeled, "validation/external/training_label_audit.csv")
    meta["src"] = src

    # ---------------- Table 1
    t1 = []
    def count(mask):
        return [int((mask & (meta.gene2 == g)).sum()) for g in ("SLC19A2", "SLC19A3")] + [int(mask.sum())]
    rec = meta[meta.y != 3].groupby("gene2").snp_id.nunique()
    t1.append(["Labeled dbSNP records (rsIDs)", str(rec["SLC19A2"]), str(rec["SLC19A3"]), str(rec.sum()), "Training and cross-validation"])
    for code, name in ((-2, "Pathogenic alleles"), (-1, "Likely pathogenic alleles"), (1, "Likely benign alleles"), (2, "Benign alleles")):
        t1.append([name] + [str(c) for c in count(meta.y == code)] + [""])
    t1.append(["  of which label belongs to the allele (trusted)"] + [str(c) for c in count((meta.src == "own_allele") & meta.y.isin([-2, -1, 1, 2]))] + ["Trusted-label evaluation"])
    t1.append(["  copied from another allele of the record"] + [str(c) for c in count((meta.src == "copied_from_sibling_allele") & meta.y.isin([-2, -1, 1, 2]))] + ["Excluded from training"])
    t1.append(["  discordant with the allele's own record"] + [str(c) for c in count((meta.src == "disagrees_with_own_allele") & meta.y.isin([-2, -1, 1, 2]))] + ["Excluded from training"])
    t1.append(["VUS alleles"] + [str(c) for c in count(meta.y == 0)] + ["Prioritization"])
    pool = FeatureSet.load("data/processed/pool").meta
    pool = pool[pool.clinical_significance.astype(str) != "normal"]
    pg = pd.Series(gene_of(pool["gene"])).value_counts()
    t1.append(["Unclassified alleles", f"{pg['SLC19A2']:,}", f"{pg['SLC19A3']:,}", f"{len(pool):,}", "Prioritization"])
    ts = pd.read_csv("reports/predictions/clinvar_timesplit.triplevar_v1.csv")
    for lab, name in ((1, "ClinVar time-split, P/LP"), (0, "ClinVar time-split, B/LB")):
        c = ts[ts.label_binary == lab].gene.value_counts()
        t1.append([name, str(c.get("SLC19A2", 0)), str(c.get("SLC19A3", 0)), str(int(c.sum())), "External validation"])
    for setname, name in (("lit_transport", "Thiamine uptake set, loss of function / tolerated"), ("lit_binding", "Thiamine binding set, loss of function / tolerated")):
        d = FeatureSet.load(f"data/processed/{setname}").meta
        cells = []
        for g in ("SLC19A2", "SLC19A3"):
            s = d[d.gene == g]
            cells.append(f"{int((s.label_binary == 1).sum())} / {int((s.label_binary == 0).sum())}" if len(s) else "—")
        cells.append(f"{int((d.label_binary == 1).sum())} / {int((d.label_binary == 0).sum())}")
        t1.append([name] + cells + ["External validation"])
    nums["table1"] = t1

    # ---------------- Table 2
    fm = json.loads(Path("reports/experiments/production.summary.json").read_text())["metrics"]
    per = json.loads(Path("reports/experiments/production.per_seed.json").read_text())
    t2 = []
    for key, label in (("test", "Grouped CV (5 × 10-fold), all labels"), ("test_clean", "Grouped CV, trusted labels")):
        n = per[0][key]["PLP_vs_BLB"]
        t2.append({"label": label, "n": f"{n['n']} ({n['tp'] + n['fn']}:{n['fp'] + n['tn']})",
                   "auc": ms(fm, f"{key}.PLP_vs_BLB.auc_renormalised"), "accuracy": ms(fm, f"{key}.PLP_vs_BLB.accuracy", True),
                   "sensitivity": ms(fm, f"{key}.PLP_vs_BLB.recall", True), "specificity": ms(fm, f"{key}.PLP_vs_BLB.specificity", True),
                   "precision": ms(fm, f"{key}.PLP_vs_BLB.precision", True)})
    n = per[0]["test_clean"]["strict_P_vs_B"]
    t2.append({"label": "Grouped CV, trusted labels, P vs B only", "n": f"{n['n']} ({n['tp'] + n['fn']}:{n['fp'] + n['tn']})",
               "auc": ms(fm, "test_clean.strict_P_vs_B.auc"), "accuracy": "—", "sensitivity": "—", "specificity": "—", "precision": "—"})
    release = load_model("models/triplevar_v1.pt")
    tsf = FeatureSet.load("data/processed/clinvar_timesplit")
    s = Scores(release.predict_proba(tsf.X(release.layout)))
    y = tsf.meta.label_binary.astype(int).to_numpy() == 1
    call, score = s.call_grouped(), s.pathogenic_given_variant
    sub = {
        "ClinVar time-split, all": np.ones(len(y), bool),
        "  no ClinVar submission before the snapshot": tsf.meta.get("strict_temporal", pd.Series(False, index=tsf.meta.index)).fillna(False).astype(bool).to_numpy(),
        "  absent from training export (no population data)": (tsf.meta.feature_source != "published_features").to_numpy(),
        "  missense": tsf.meta.is_missense.fillna(False).astype(bool).to_numpy(),
    }
    for label, m in sub.items():
        t2.append(binary_row(y[m], call[m], score[m], label))
    nums["timesplit_v1"] = t2[3]["_raw"]
    for setname, label in (("lit_transport", "Thiamine uptake (Li et al. 2024)"), ("lit_binding", "Thiamine binding (Gabriel et al. 2024)")):
        f = FeatureSet.load(f"data/processed/{setname}")
        sc = Scores(release.predict_proba(f.X(release.layout)))
        yy = f.meta.label_binary.astype(int).to_numpy() == 1
        t2.append(binary_row(yy, sc.call_grouped(), sc.pathogenic_given_variant, label))
    nums["table2"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in t2]

    # abstract model on the time-split, same scoring rule
    ab = load_model("models/triplevar_abstract_2025-09-11.pt")
    sa = Scores(ab.predict_proba(tsf.X(ab.layout)))
    nums["timesplit_abstract"] = binary_row(y, sa.call_grouped(), sa.pathogenic_given_variant, "abstract model")["_raw"]

    # ---------------- ablation
    abl = {}
    for p in sorted(Path("outputs/ablation").glob("*.summary.json")):
        m = json.loads(p.read_text())["metrics"]
        abl[p.stem.replace(".summary", "")] = {
            "trusted_auc": m["test_clean.PLP_vs_BLB.auc_renormalised"]["mean"], "trusted_sd": m["test_clean.PLP_vs_BLB.auc_renormalised"]["sd"],
            "all_auc": m["test.PLP_vs_BLB.auc_renormalised"]["mean"],
            "timesplit_auc": m.get("external:clinvar_timesplit.grouped_argmax.auc", {}).get("mean"),
        }
    nums["ablation"] = abl

    # ---------------- benchmark
    bm = Path("validation/benchmarks/benchmark_metrics.csv")
    if bm.exists():
        nums["benchmark_metrics_csv"] = str(bm)

    # ---------------- supplementary tables
    desc = {
        "s01_big_legacy": "Abstract settings (4 layers, early stop on accuracy + pathogenic recall)",
        "s02_big_auc": "s01, early stop on validation AUC",
        "s03_big_auc_nonormal": "s02, reference class removed",
        "s04_small_auc": "Small network (512-128, dropout 0.5, AdamW), early stop on AUC",
        "s05_small_auc_nonormal": "s04, reference class removed",
        "s06_small_auc_freqdrop": "s04 + frequency dropout 0.3",
        "s07_small_auc_nonormal_freqdrop": "s05 + frequency dropout 0.3",
        "s08_big_auc_cleanlabels": "s02, trained on trusted labels only",
        "s09_small_auc_cleanlabels": "s04, trained on trusted labels only",
        "s10_small_auc_freqdrop_cleanlabels": "s06, trained on trusted labels only",
        "s11_big_auc_freqdrop": "s02 + frequency dropout 0.3",
        "s12_big_auc_freqdrop_cleanlabels": "s11, trained on trusted labels only",
        "s13_big_legacy_cleanlabels": "s01, trained on trusted labels only",
        "s14_big_legacy_freqdrop": "s01 + frequency dropout 0.3",
        "s15_big_legacy_freqdrop_cleanlabels": "s01 + frequency dropout 0.3, trusted labels only (selected)",
    }
    s1 = []
    for name, d in desc.items():
        m = json.loads(Path(f"reports/experiments/search/{name}.summary.json").read_text())["metrics"]
        s1.append([name.split("_")[0].upper(), d, ms(m, "test.PLP_vs_BLB.auc_renormalised"),
                   ms(m, "test_clean.PLP_vs_BLB.auc_renormalised"), ms(m, "test.strict_P_vs_B.auc")])
    nums["tableS1"] = s1

    s2 = []
    cats = [("own_allele", "Trusted"), ("copied_from_sibling_allele", "Copied"), ("disagrees_with_own_allele", "Discordant"),
            ("own_allele_classification_unknown_at_cutoff", "Unresolved")]
    for code, name in ((-2, "Pathogenic"), (-1, "Likely pathogenic"), (1, "Likely benign"), (2, "Benign"), (0, "VUS")):
        for g in ("SLC19A2", "SLC19A3"):
            mm = (meta.y == code) & (meta.gene2 == g)
            s2.append([name, g] + [str(int((mm & (meta.src == c)).sum())) for c, _ in cats] + [str(int(mm.sum()))])
    nums["tableS2"] = s2

    s3 = []
    for setname, label in (("lit_transport", "Uptake"), ("lit_binding", "Binding"), ("literature_abstract", "Abstract labels")):
        p = pd.read_csv(f"reports/predictions/{setname}.triplevar_v1.csv")
        f = FeatureSet.load(f"data/processed/{setname}").meta
        for i, r in p.iterrows():
            ev = r.get("evidence", "")
            lab = r.get("label_abstract") if setname == "literature_abstract" else ("Loss of function" if r["label_binary"] == 1 else "Tolerated")
            if setname == "literature_abstract" and pd.isna(lab):
                lab = "Unlabeled"
            note = "training allele" if setname == "literature_abstract" and bool(f.loc[i, "in_training"]) else ""
            s3.append([label, r["gene"], r["protein_change"], "" if pd.isna(ev) else ev, lab, f"{r['p_pathogenic']:.2f}", r["predicted_class"], note])
    nums["tableS3"] = s3

    rep = json.loads(Path("reports/abstract_reproduction/abstract_reproduction.json").read_text())
    st, pl = rep["clinvar_test"]["strict_P_vs_B"], rep["clinvar_test"]["PLP_vs_BLB"]
    lit = rep["literature_external"]["rule_strict_ratio_gt_0.5"]
    ap_ = json.loads(Path("reports/experiments/abstract_protocol.summary.json").read_text())["metrics"]
    s01 = json.loads(Path("reports/experiments/search/s01_big_legacy.summary.json").read_text())["metrics"]
    nums["tableS4"] = [
        ["Held-out P vs B: accuracy / precision / recall / specificity (%)", "93.3 / 100 / 84.6 / 100",
         f"{pct(st['accuracy'])} / {pct(st['precision'])} / {pct(st['recall'])} / {pct(st['specificity'])}"],
        ["Held-out P vs B: AUC", "1.0", f"{st['auc']:.3f}"],
        ["Held-out P/LP vs B/LB: accuracy (%) / AUC", "93.7 / 0.913", f"{pct(pl['accuracy'])} / {pl['auc_renormalised']:.3f}"],
        ["Literature set (15): accuracy / precision / recall / specificity (%)", "80 / 90 / 82 / 75",
         f"{pct(lit['accuracy'])} / {pct(lit['precision'])} / {pct(lit['recall'])} / {pct(lit['specificity'])}"],
        ["Pathogenic / likely pathogenic calls among VUS and unclassified alleles", "111 / 303",
         f"{rep['prioritised']['pathogenic_high_confidence']} / {rep['prioritised']['likely_pathogenic_lower_confidence']}"],
        ["Row-level split, abstract protocol, 5 seeds: P/LP vs B/LB AUC", "—", ms(ap_, "test.PLP_vs_BLB.auc_renormalised")],
        ["Row-level split, abstract protocol, 5 seeds: P vs B AUC", "—", ms(ap_, "test.strict_P_vs_B.auc")],
        ["Grouped CV, abstract settings, 2 repeats: P/LP vs B/LB AUC", "—", ms(s01, "test.PLP_vs_BLB.auc_renormalised")],
        ["Grouped CV, abstract settings, 2 repeats: P vs B AUC", "—", ms(s01, "test.strict_P_vs_B.auc")],
    ]

    OUT.write_text(json.dumps(nums, indent=1, default=float))
    print(json.dumps({k: v for k, v in nums.items() if k != "table1"}, indent=1, default=float)[:6000])
    for row in t1:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
