#!/usr/bin/env python3
"""
Create Butina-cluster-separated nested CV folds for the BBBX drug-like dataset.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


DEFAULT_SMILES_COL = "smiles"
DEFAULT_CLASS_COL = "bbb_class"
DEFAULT_TARGET_COL = "logBB"
DEFAULT_OUTER_FOLD_COL = "outer_fold"
DEFAULT_CLUSTER_COL = "butina_cluster_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-col", default=DEFAULT_SMILES_COL)
    parser.add_argument("--class-col", default=DEFAULT_CLASS_COL)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--outer-fold-col", default=DEFAULT_OUTER_FOLD_COL)
    parser.add_argument("--cluster-col", default=DEFAULT_CLUSTER_COL)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--similarity-cutoff", type=float, default=0.60)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument(
        "--include-chirality",
        action="store_true",
        help="Include chirality when generating Morgan fingerprints.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be >= 2")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be >= 2")
    if not 0.0 < args.similarity_cutoff < 1.0:
        raise ValueError("--similarity-cutoff must be between 0 and 1")
    if args.radius < 1:
        raise ValueError("--radius must be >= 1")
    if args.fp_size < 128:
        raise ValueError("--fp-size looks implausibly small")


def load_and_validate_data(args: argparse.Namespace) -> pd.DataFrame:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, low_memory=False)

    required = {args.smiles_col, args.class_col, args.target_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    classes = pd.to_numeric(df[args.class_col], errors="raise").to_numpy(int)
    if not set(np.unique(classes)).issubset({0, 1}):
        raise ValueError(
            f"{args.class_col} must use 0=penetrant and 1=non-penetrant."
        )

    # Remove old nested-CV assignments. The input CSV remains untouched;
    # these are removed only from the new output dataframe.
    old_inner_cols = [
        c for c in df.columns if str(c).startswith("inner_fold_outer_")
    ]
    if old_inner_cols:
        print(f"Removing {len(old_inner_cols)} old inner-fold columns.")
        df = df.drop(columns=old_inner_cols)

    return df


def make_molecules_and_fingerprints(
    smiles: pd.Series,
    *,
    radius: int,
    fp_size: int,
    include_chirality: bool,
):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=include_chirality,
    )

    mols = []
    fps = []
    invalid = []

    for i, value in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            invalid.append((i, value))
            continue
        mols.append(mol)
        fps.append(generator.GetFingerprint(mol))

    if invalid:
        examples = "\n".join(
            f"  row {i}: {s}" for i, s in invalid[:10]
        )
        raise ValueError(
            f"RDKit could not parse {len(invalid)} SMILES. "
            f"Fix them before making the split. Examples:\n{examples}"
        )

    return mols, fps


def butina_clusters(
    fps,
    *,
    similarity_cutoff: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """
    Cluster fingerprints with Butina using Tanimoto distance.

    RDKit expects the lower triangle of the distance matrix in the order:
      d(1,0), d(2,0), d(2,1), d(3,0), ...
    """
    n = len(fps)
    if n == 0:
        raise ValueError("No molecules available for clustering.")

    n_distances = n * (n - 1) // 2
    print(
        f"Computing {n_distances:,} pairwise Tanimoto distances "
        f"for {n:,} molecules..."
    )

    # float32 avoids the very large memory overhead of a Python list of floats.
    distances = np.empty(n_distances, dtype=np.float32)
    offset = 0

    for i in range(1, n):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        next_offset = offset + i
        distances[offset:next_offset] = 1.0 - np.asarray(
            similarities,
            dtype=np.float32,
        )
        offset = next_offset

    distance_cutoff = 1.0 - similarity_cutoff
    clusters = Butina.ClusterData(
        distances,
        nPts=n,
        distThresh=distance_cutoff,
        isDistData=True,
        reordering=True,
    )

    cluster_ids = np.full(n, -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        cluster_ids[np.asarray(members, dtype=int)] = cluster_id

    if np.any(cluster_ids < 0):
        raise RuntimeError("At least one molecule was not assigned to a Butina cluster.")

    return cluster_ids, clusters


def make_requested_strata(
    classes: np.ndarray,
    measured: np.ndarray,
) -> np.ndarray:
    availability = np.where(measured, "measured", "classification_only")
    return np.asarray(
        [f"class_{c}__{a}" for c, a in zip(classes, availability)],
        dtype=object,
    )


def strata_have_enough_groups(
    strata: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> bool:
    table = pd.DataFrame({"stratum": strata, "group": groups})
    group_counts = table.groupby("stratum")["group"].nunique()
    sample_counts = table["stratum"].value_counts()
    return bool(
        len(group_counts)
        and (group_counts >= n_splits).all()
        and (sample_counts >= n_splits).all()
    )


def choose_strata(
    classes: np.ndarray,
    measured: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[Optional[np.ndarray], str]:
    combined = make_requested_strata(classes, measured)
    if strata_have_enough_groups(combined, groups, n_splits):
        return combined, "bbb_class_x_logBB_availability"

    class_only = np.asarray([f"class_{c}" for c in classes], dtype=object)
    if strata_have_enough_groups(class_only, groups, n_splits):
        return class_only, "bbb_class_only"

    return None, "group_only"


def assign_grouped_folds(
    *,
    classes: np.ndarray,
    measured: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return one fold assignment per input row while never splitting groups."""
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Only {len(np.unique(groups))} unique clusters are available for "
            f"{n_splits} folds."
        )

    strata, strategy = choose_strata(
        classes=classes,
        measured=measured,
        groups=groups,
        n_splits=n_splits,
    )

    assignments = np.full(len(classes), -1, dtype=int)
    placeholder = np.zeros((len(classes), 1), dtype=float)

    if strata is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(placeholder, strata, groups=groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(placeholder, classes, groups=groups)

    for fold, (_, test_idx) in enumerate(iterator):
        assignments[test_idx] = fold

    if np.any(assignments < 0):
        raise RuntimeError("Fold assignment failed for one or more rows.")

    # Hard leakage check.
    group_fold_counts = (
        pd.DataFrame({"group": groups, "fold": assignments})
        .groupby("group")["fold"]
        .nunique()
    )
    if int(group_fold_counts.max()) != 1:
        raise RuntimeError("A chemical cluster was split across folds.")

    return assignments, strategy


def build_fold_summary(
    *,
    df: pd.DataFrame,
    fold_values: np.ndarray,
    groups: np.ndarray,
    class_col: str,
    target_col: str,
    level: str,
    outer_fold: Optional[int] = None,
) -> pd.DataFrame:
    y_class = pd.to_numeric(df[class_col], errors="raise").to_numpy(int)
    y_reg = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    measured = np.isfinite(y_reg)

    rows = []
    for fold in sorted(np.unique(fold_values)):
        mask = fold_values == fold
        rows.append(
            {
                "level": level,
                "outer_fold_context": outer_fold,
                "fold": int(fold),
                "n_samples": int(mask.sum()),
                "n_clusters": int(len(np.unique(groups[mask]))),
                "n_penetrant": int(np.sum(y_class[mask] == 0)),
                "n_nonpenetrant": int(np.sum(y_class[mask] == 1)),
                "n_measured_logBB": int(np.sum(measured[mask])),
                "n_classification_only": int(np.sum(~measured[mask])),
                "fraction_nonpenetrant": float(np.mean(y_class[mask] == 1)),
                "fraction_measured_logBB": float(np.mean(measured[mask])),
            }
        )
    return pd.DataFrame(rows)


def build_cluster_summary(
    df: pd.DataFrame,
    *,
    clusters: tuple[tuple[int, ...], ...],
    class_col: str,
    target_col: str,
    smiles_col: str,
) -> pd.DataFrame:
    y_class = pd.to_numeric(df[class_col], errors="raise").to_numpy(int)
    y_reg = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    measured = np.isfinite(y_reg)
    smiles = df[smiles_col].astype(str).to_numpy()

    rows = []
    for cluster_id, members_tuple in enumerate(clusters):
        members = np.asarray(members_tuple, dtype=int)
        centroid = int(members[0])  # Butina returns centroid first.
        rows.append(
            {
                "cluster_id": f"butina_{cluster_id:06d}",
                "cluster_size": int(len(members)),
                "centroid_row_index": centroid,
                "centroid_smiles": smiles[centroid],
                "n_penetrant": int(np.sum(y_class[members] == 0)),
                "n_nonpenetrant": int(np.sum(y_class[members] == 1)),
                "n_measured_logBB": int(np.sum(measured[members])),
                "n_classification_only": int(np.sum(~measured[members])),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["cluster_size", "cluster_id"],
        ascending=[False, True],
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path == args.input.resolve():
        raise ValueError(
            "Refusing to overwrite the source CSV. Choose a different --output path."
        )

    df = load_and_validate_data(args)
    n = len(df)
    if n == 0:
        raise ValueError("Input CSV is empty.")

    print(f"Loaded {n:,} rows from {args.input}")
    print(
        f"Morgan radius={args.radius}, fpSize={args.fp_size}, "
        f"Tanimoto cutoff={args.similarity_cutoff:.3f}"
    )

    _, fps = make_molecules_and_fingerprints(
        df[args.smiles_col],
        radius=args.radius,
        fp_size=args.fp_size,
        include_chirality=args.include_chirality,
    )

    cluster_ids_int, clusters = butina_clusters(
        fps,
        similarity_cutoff=args.similarity_cutoff,
    )

    cluster_sizes = np.asarray([len(c) for c in clusters], dtype=int)
    print(f"Created {len(clusters):,} Butina clusters.")
    print(f"Largest cluster: {cluster_sizes.max():,} molecules")
    print(f"Median cluster size: {np.median(cluster_sizes):.1f}")
    print(f"Singleton clusters: {int(np.sum(cluster_sizes == 1)):,}")

    # Keep this column NON-NUMERIC so your descriptor-inference code cannot
    # accidentally treat cluster IDs as molecular features.
    df[args.cluster_col] = np.asarray(
        [f"butina_{c:06d}" for c in cluster_ids_int],
        dtype=object,
    )

    y_class = pd.to_numeric(df[args.class_col], errors="raise").to_numpy(int)
    y_reg = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)
    measured = np.isfinite(y_reg)

    # ------------------------------------------------------------------
    # OUTER folds: entire Butina clusters are held out together.
    # ------------------------------------------------------------------
    outer_assignments, outer_strategy = assign_grouped_folds(
        classes=y_class,
        measured=measured,
        groups=cluster_ids_int,
        n_splits=args.outer_folds,
        seed=args.seed,
    )
    df[args.outer_fold_col] = outer_assignments
    df["outer_stratum"] = make_requested_strata(y_class, measured)

    print(f"Outer split strategy: {outer_strategy}")

    fold_summaries = [
        build_fold_summary(
            df=df,
            fold_values=outer_assignments,
            groups=cluster_ids_int,
            class_col=args.class_col,
            target_col=args.target_col,
            level="outer",
        )
    ]

    # ------------------------------------------------------------------
    # INNER folds: regenerated separately inside each outer-training set.
    # This is critical: HPO must not split a cluster across train/validation.
    # ------------------------------------------------------------------
    inner_strategies: dict[str, str] = {}

    for outer_fold in range(args.outer_folds):
        train_mask = outer_assignments != outer_fold
        train_idx = np.flatnonzero(train_mask)

        inner_assignments_rel, inner_strategy = assign_grouped_folds(
            classes=y_class[train_idx],
            measured=measured[train_idx],
            groups=cluster_ids_int[train_idx],
            n_splits=args.inner_folds,
            seed=args.seed + 1000 * outer_fold,
        )

        column = f"inner_fold_outer_{outer_fold}"
        # Float column permits NaN for the held-out outer-test rows.
        df[column] = np.nan
        df.loc[train_idx, column] = inner_assignments_rel.astype(float)

        # Verify that each training cluster belongs to exactly one inner fold.
        check = pd.DataFrame(
            {
                "cluster": cluster_ids_int[train_idx],
                "inner_fold": inner_assignments_rel,
            }
        ).groupby("cluster")["inner_fold"].nunique()
        if int(check.max()) != 1:
            raise RuntimeError(
                f"Cluster leakage detected in inner CV for outer fold {outer_fold}."
            )

        inner_strategies[str(outer_fold)] = inner_strategy

        fold_summaries.append(
            build_fold_summary(
                df=df.iloc[train_idx].reset_index(drop=True),
                fold_values=inner_assignments_rel,
                groups=cluster_ids_int[train_idx],
                class_col=args.class_col,
                target_col=args.target_col,
                level="inner",
                outer_fold=outer_fold,
            )
        )

        print(
            f"Outer fold {outer_fold}: generated {args.inner_folds} "
            f"cluster-separated inner folds ({inner_strategy})."
        )

    # Final hard check: every global Butina cluster has exactly one outer fold.
    outer_leakage = (
        pd.DataFrame(
            {"cluster": cluster_ids_int, "outer_fold": outer_assignments}
        )
        .groupby("cluster")["outer_fold"]
        .nunique()
    )
    if int(outer_leakage.max()) != 1:
        raise RuntimeError("Outer-fold chemical-cluster leakage detected.")

    # Save primary training CSV.
    df.to_csv(output_path, index=False)

    # Save diagnostics next to it.
    stem = output_path.with_suffix("")
    cluster_summary_path = Path(f"{stem}_cluster_summary.csv")
    fold_summary_path = Path(f"{stem}_fold_summary.csv")
    manifest_path = Path(f"{stem}_split_manifest.json")

    cluster_summary = build_cluster_summary(
        df,
        clusters=clusters,
        class_col=args.class_col,
        target_col=args.target_col,
        smiles_col=args.smiles_col,
    )
    # cluster_summary.to_csv(cluster_summary_path, index=False)

    fold_summary = pd.concat(fold_summaries, ignore_index=True)
    # fold_summary.to_csv(fold_summary_path, index=False)

    manifest = {
        "input": str(args.input.resolve()),
        "output": str(output_path),
        "n_samples": int(n),
        "n_butina_clusters": int(len(clusters)),
        "largest_cluster_size": int(cluster_sizes.max()),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "n_singleton_clusters": int(np.sum(cluster_sizes == 1)),
        "fingerprint": {
            "type": "Morgan bit vector",
            "radius": int(args.radius),
            "fp_size": int(args.fp_size),
            "include_chirality": bool(args.include_chirality),
        },
        "similarity": "Tanimoto",
        "similarity_cutoff": float(args.similarity_cutoff),
        "butina_distance_cutoff": float(1.0 - args.similarity_cutoff),
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "outer_split_strategy": outer_strategy,
        "inner_split_strategies": inner_strategies,
        "seed": int(args.seed),
        "class_encoding": "0=penetrant, 1=non-penetrant",
        "rdkit_version": rdBase.rdkitVersion,
        "sklearn_version": sklearn_version,
        "leakage_check": "PASS: each Butina cluster occurs in exactly one outer fold and one inner fold within each outer-training set",
    }
    # manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nDONE")
    print(f"Training CSV:      {output_path}")
    # print(f"Cluster summary:   {cluster_summary_path}")
    # print(f"Fold summary:      {fold_summary_path}")
    # print(f"Split manifest:    {manifest_path}")
    print("\nOuter-fold summary:")
    print(fold_summary.loc[fold_summary["level"] == "outer"].to_string(
            index=False))

if __name__ == "__main__":
    main()
