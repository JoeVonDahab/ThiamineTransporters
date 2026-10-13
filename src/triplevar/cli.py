"""Command-line interface: ``triplevar <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("triplevar")

DEFAULT_LEGACY_ROOT = "data/legacy"


def _load_external(spec: str):
    """NAME=PATH where PATH is a FeatureSet prefix whose metadata has a 'label_binary' column."""
    from .dataset import FeatureSet
    from .experiment import ExternalSet

    name, path = spec.split("=", 1)
    fs = FeatureSet.load(path)
    if "label_binary" not in fs.meta.columns:
        raise SystemExit(f"{path}.csv needs a 'label_binary' column (1 = pathogenic, 0 = benign)")
    keep = fs.meta["label_binary"].notna().to_numpy()
    fs = fs.subset(keep)
    return ExternalSet(name, fs, fs.meta["label_binary"].astype(int).to_numpy() == 1)


def cmd_reproduce_abstract(args) -> None:
    from .reproduce import ABSTRACT_CLAIMS, reproduce_abstract

    report = reproduce_abstract(args.legacy_root, args.out, args.labels)
    print(json.dumps({"reproduced": {k: report[k] for k in report if k != "claims"}, "claimed": ABSTRACT_CLAIMS}, indent=2))


def cmd_import_legacy(args) -> None:
    from .legacy import LegacyArtifacts

    art = LegacyArtifacts(args.legacy_root)
    out = Path(args.out)
    for name, getter in (("labeled", art.labeled), ("pool", art.pool), ("literature", art.literature)):
        fs = getter()
        fs.save(out / name)
        log.info("%s: %d rows -> %s.npz/.csv", name, len(fs), out / name)


def cmd_train(args) -> None:
    from .config import RunConfig
    from .dataset import FeatureSet
    from .experiment import run

    cfg = RunConfig.from_yaml(args.config)
    if args.seeds:
        cfg.seeds = args.seeds
    labeled = FeatureSet.load(args.data)
    externals = [_load_external(s) for s in args.external or []]
    drop, clean = None, None
    if args.label_audit:
        from .audit import masks

        untrusted, clean = masks(labeled, args.label_audit)
        drop = untrusted if args.drop_untrusted_labels else None
        log.info("label audit: %d labeled rows have copied/disagreeing labels%s", int(untrusted.sum()),
                 " (excluded from training)" if drop is not None else "")
    if args.final:
        from .ensemble import Ensemble
        from .experiment import fit_final

        members = [fit_final(cfg, labeled, seed=s, device=args.device, drop_from_training=drop) for s in cfg.seeds]
        path = Path(args.out) / f"{cfg.name}.pt"
        if len(members) == 1:
            members[0].save(path)
        else:
            Ensemble(members, {"run_config": cfg.name, "seeds": cfg.seeds, "release": True}).save(path, half=args.half)
        info = [{k: v for k, v in m.metadata.items() if k != "train_config"} for m in members]
        print(json.dumps({"saved": str(path), "members": info}, indent=2))
        return
    summary = run(cfg, labeled, Path(args.out), externals, device=args.device,
                  drop_from_training=drop, clean_eval_mask=clean, save_model=not args.no_save)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))


def cmd_predict(args) -> None:
    from .dataset import FeatureSet
    from .ensemble import load_model
    from .predict import prediction_table

    bundle = load_model(args.bundle)
    fs = FeatureSet.load(args.features)
    table = prediction_table(bundle, fs.X(bundle.layout), fs.meta, device=args.device or "cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    counts = table["predicted_class"].value_counts().to_dict()
    log.info("wrote %d predictions to %s: %s", len(table), args.out, counts)


def cmd_evaluate(args) -> None:
    from .ensemble import load_model
    from .metrics import evaluate_external

    model = load_model(args.model)
    report = {}
    for spec in args.external:
        ext = _load_external(spec)
        report[ext.name] = evaluate_external(model.predict_proba(ext.features.X(model.layout)), ext.y_pathogenic)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


def cmd_embed(args) -> None:
    from .embeddings import embed_variant_table

    embed_variant_table(args.input, args.out, device=args.device, limit=args.limit)


def cmd_build_external(args) -> None:
    from .external import build_external_set

    build_external_set(args.input, args.out, reference_dir=args.reference_dir, device=args.device)


def cmd_prepare(args) -> None:
    from .genes import get_gene
    from .prepare import build_gene_table, ncbi_spdi_resolver
    from .reference import load_reference

    gene = get_gene(args.gene)
    ref = load_reference(gene, args.reference_dir)
    spdi = ncbi_spdi_resolver(gene.chromosome_accession) if args.spdi else None
    table = build_gene_table(args.export, ref, spdi)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(args.out)
    log.info("allele status: %s", table["allele_status"].value_counts().to_dict())


def cmd_reference(args) -> None:
    from .genes import GENES
    from .reference import export_windows_from_legacy, fetch_window

    if args.from_legacy:
        export_windows_from_legacy(args.from_legacy, args.out)
    else:
        for gene in GENES.values():
            log.info("downloaded %s", fetch_window(gene, args.out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triplevar", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("reference", help="write the wild-type genomic windows (NCBI download or from legacy data)")
    s.add_argument("--from-legacy", metavar="LEGACY_ROOT", help="recover windows from the published dataset instead")
    s.add_argument("--out", default="data/reference")
    s.set_defaults(func=cmd_reference)

    s = sub.add_parser("prepare", help="raw dbSNP export -> per-allele table with sequences and frequency features")
    s.add_argument("--gene", required=True, choices=["SLC19A2", "SLC19A3"])
    s.add_argument("--export", required=True, help="dbSNP tabular export (data/raw/dbsnp/...)")
    s.add_argument("--reference-dir", default="data/reference")
    s.add_argument("--spdi", action="store_true", help="resolve reference-allele mismatches via NCBI dbSNP (network)")
    s.add_argument("--out", required=True, help="output .pkl")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("reproduce-abstract", help="re-score the ASCPT 2026 abstract model on the published data")
    s.add_argument("--legacy-root", default=DEFAULT_LEGACY_ROOT, help="original 'Both Genes' directory")
    s.add_argument("--labels", default="validation/external/abstract_literature_labels.csv")
    s.add_argument("--out", default="reports/abstract_reproduction")
    s.set_defaults(func=cmd_reproduce_abstract)

    s = sub.add_parser("import-legacy", help="export the published datasets + embeddings as FeatureSets")
    s.add_argument("--legacy-root", default=DEFAULT_LEGACY_ROOT)
    s.add_argument("--out", default="data/processed")
    s.set_defaults(func=cmd_import_legacy)

    s = sub.add_parser("train", help="train/evaluate a YAML configuration")
    s.add_argument("--config", required=True)
    s.add_argument("--data", default="data/processed/labeled", help="FeatureSet prefix")
    s.add_argument("--external", action="append", metavar="NAME=PATH", help="external FeatureSet with label_binary")
    s.add_argument("--seeds", type=int, nargs="+")
    s.add_argument("--label-audit", help="training_label_audit.csv: adds a trusted-label evaluation")
    s.add_argument("--drop-untrusted-labels", action="store_true", help="exclude copied/disagreeing labels from training")
    s.add_argument("--no-save", action="store_true", help="do not write the fold-model ensemble")
    s.add_argument("--final", action="store_true",
                   help="fit the release model on all labeled data (one grouped fold for early stopping); "
                        "several --seeds give an ensemble")
    s.add_argument("--half", action="store_true", help="store ensemble weight matrices in float16")
    s.add_argument("--out", default="outputs/experiments")
    s.add_argument("--device")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="score a FeatureSet with a model bundle")
    s.add_argument("--bundle", required=True)
    s.add_argument("--features", required=True, help="FeatureSet prefix")
    s.add_argument("--out", required=True)
    s.add_argument("--device")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("evaluate", help="score a model (bundle or ensemble) on labelled external sets")
    s.add_argument("--model", required=True)
    s.add_argument("--external", action="append", required=True, metavar="NAME=PATH")
    s.add_argument("--out", help="write the JSON report here")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("embed", help="DNABERT-2 + ESM-2 embeddings for a variant table (GPU)")
    s.add_argument("--input", required=True, help="CSV with transcript_dna and protein_sequence columns")
    s.add_argument("--out", required=True, help="FeatureSet prefix to write")
    s.add_argument("--limit", type=int, help="only embed the first N rows (smoke test)")
    s.add_argument("--device")
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("build-external", help="variant list (gene, pos/ref/alt or protein change) -> FeatureSet")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reference-dir", default="data/reference")
    s.add_argument("--device")
    s.set_defaults(func=cmd_build_external)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args.func(args)


if __name__ == "__main__":
    main()
