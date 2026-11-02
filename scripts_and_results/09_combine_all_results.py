"""Collect completed BBBX training outputs for downstream analyses."""

import argparse
from pathlib import Path
import shutil
import tempfile

import pandas as pd


def read_results_csv(f):
    try:
        return pd.read_csv(f)
    except UnicodeDecodeError:
        print(f"Reading as Windows-1252: {f}")
        return pd.read_csv(f, encoding="cp1252")


def collect_results(model_dirs, out):
    """Keep the original checkpoint naming and CSV-combination operations."""
    (out / "fitted_models").mkdir(parents=True, exist_ok=True)

    for d in model_dirs:
        for f in (d / "fitted_models").glob("*"):
            shutil.copy2(f, out / "fitted_models" / f.name)

    # ImageMol is last without GNN, or second-to-last when GNN is included.
    for d in model_dirs[-2:]:
        for f in d.rglob("model.pt"):
            name = "__".join((d.name, *f.relative_to(d).parts))
            shutil.copy2(f, out / "fitted_models" / name)

    suffixes = ["best_parameters.csv", "fold_metrics.csv", "mean_sd.csv",
                "pooled_metrics.csv", "predictions.csv"]

    for suffix in suffixes:
        files = [f for d in model_dirs for f in d.glob(f"*{suffix}")]
        if files:
            pd.concat([read_results_csv(f) for f in files], ignore_index=True).drop_duplicates().to_csv(
                out / f"all_models_{suffix}", index=False
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine completed BBBX model results and copy their fitted models.",
        epilog=("Paths are relative to the current working directory unless absolute. "
                "An existing output directory is retained as a sibling backup after "
                "the new collection has been built successfully."),
    )
    parser.add_argument(
        "--outerfolds-dir", type=Path, required=True,
        help="Variant directory containing the four standard model-results directories.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Destination for all_models_*.csv and fitted_models/.",
    )
    parser.add_argument(
        "--gnn-dir", type=Path,
        help="Optional GNN results directory. Omit this argument to exclude GNN results.",
    )
    args = parser.parse_args(argv)

    outerfolds_dir = args.outerfolds_dir.expanduser().resolve()
    model_dirs = [
        outerfolds_dir / "custom_mtl_bab_results",
        outerfolds_dir / "logbb_pred_results",
        outerfolds_dir / "baseline_threshold_ablation_results",
        outerfolds_dir / "imagemol_frozen",
    ]
    if args.gnn_dir is not None:
        model_dirs.append(args.gnn_dir.expanduser().resolve())

    missing = [str(d) for d in model_dirs if not d.is_dir()]
    if missing:
        parser.error("Missing model-results directories:\n  " + "\n  ".join(missing))
    if len(set(d.resolve() for d in model_dirs)) != len(model_dirs):
        parser.error("Each model-results directory must be selected only once.")

    output_path = args.output_dir.expanduser()
    if output_path.is_symlink():
        parser.error("--output-dir must be a directory, not a symbolic link.")
    out = output_path.resolve()
    if out.exists() and not out.is_dir():
        parser.error("--output-dir exists and is not a directory.")
    if out == outerfolds_dir or out in outerfolds_dir.parents:
        parser.error("--output-dir must not replace the variant directory or its parents.")
    if any(out == d.resolve() or out in d.resolve().parents or d.resolve() in out.parents
           for d in model_dirs):
        parser.error("--output-dir must be separate from every selected training directory.")

    print("Collecting results from:")
    for d in model_dirs:
        print(f"  {d}")

    # Build a complete collection first, so a read/copy failure leaves the old
    # collection intact and omitted models cannot survive as stale checkpoints.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    backup = None
    try:
        collect_results(model_dirs, staging)
        if out.exists():
            backup = Path(tempfile.mkdtemp(prefix=f"{out.name}.backup-", dir=out.parent))
            backup.rmdir()
            out.rename(backup)
        try:
            staging.rename(out)
        except OSError:
            if backup is not None:
                backup.rename(out)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"Saved combined results to: {out}")
    if backup is not None:
        print(f"Previous collection preserved at: {backup}")


if __name__ == "__main__":
    main()
