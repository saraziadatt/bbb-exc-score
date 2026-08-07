#!/usr/bin/env python3
"""Generate target-independent descriptor clusters without training models.

This is the clustering-only portion of train_cluster_medoid_lightgbm_models.py.
It reads the full descriptor table, clusters redundant descriptors using
1 - |correlation|, chooses an algorithmic medoid for each cluster, and writes
exactly the membership table expected by the grouped-SHAP workflow:

    OUTPUT_DIR/clusters/final_cluster_membership.csv

No predictive model is trained, tuned, loaded, or evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer


DEFAULT_METADATA_NAMES = {
    "logbb", "bbb_class", "class", "outer_fold", "smiles",
    "canonical_smiles", "sample_id", "compound_id", "id", "has_logbb",
    "logbb_is_measured", "outer_stratum",
}
EXCLUDED_PREFIXES = (
    "inner_fold_outer_", "pred_", "prediction", "oof_", "shap_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate descriptor-correlation clusters only; fit no models."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster_medoid_lightgbm_results"),
        help="Outputs are written under OUTPUT_DIR/clusters/.",
    )
    parser.add_argument("--target-col", default="logBB")
    parser.add_argument("--class-col", default="bbb_class")
    parser.add_argument("--fold-col", default="outer_fold")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    parser.add_argument(
        "--correlation-method", choices=["spearman", "pearson"], default="spearman"
    )
    parser.add_argument(
        "--linkage-method", choices=["complete", "average"], default="complete"
    )
    parser.add_argument("--min-nonmissing", type=int, default=20)
    parser.add_argument(
        "--representative-overrides",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing representative_feature and either cluster_id "
            "or algorithmic_medoid. scope may be final or all."
        ),
    )
    parser.add_argument("--save-correlation-matrix", action="store_true")
    parser.add_argument("--no-save-within-cluster-pairs", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace OUTPUT_DIR/clusters/ but leave other OUTPUT_DIR files untouched.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def prepare_cluster_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = output_dir / "clusters"

    if cluster_dir.exists() and any(cluster_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Cluster directory is not empty: {cluster_dir}. Use --overwrite."
            )
        shutil.rmtree(cluster_dir)

    cluster_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, cluster_dir


def infer_feature_columns(frame: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    metadata = {
        *DEFAULT_METADATA_NAMES,
        args.target_col.lower(),
        args.class_col.lower(),
        args.fold_col.lower(),
        args.smiles_col.lower(),
        args.id_col.lower(),
    }
    features: list[str] = []
    for column in frame.columns:
        lower = str(column).strip().lower()
        if lower in metadata:
            continue
        if any(lower.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if lower.startswith("unnamed:") or lower in {"index", "level_0"}:
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            features.append(str(column))

    if not features:
        raise ValueError("No numeric descriptor columns were found.")
    if len(features) != len(set(features)):
        raise ValueError("Duplicate descriptor names were detected.")
    return features


def filter_usable_features(
    X: pd.DataFrame, min_nonmissing: int
) -> tuple[list[str], pd.DataFrame]:
    kept: list[str] = []
    rows: list[dict[str, Any]] = []
    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")
        n_nonmissing = int(values.notna().sum())
        n_unique = int(values.nunique(dropna=True))
        if n_nonmissing < min_nonmissing:
            status = "excluded_too_incomplete"
        elif n_unique <= 1:
            status = "excluded_constant"
        else:
            status = "included"
            kept.append(str(feature))
        rows.append(
            {
                "feature": str(feature),
                "n_nonmissing": n_nonmissing,
                "missing_fraction": float(values.isna().mean()),
                "n_unique_nonmissing": n_unique,
                "status": status,
            }
        )
    if not kept:
        raise ValueError("No usable descriptors remained before clustering.")
    return kept, pd.DataFrame(rows)


def load_overrides(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if "representative_feature" not in table.columns:
        raise KeyError("Override CSV must contain representative_feature.")
    if "cluster_id" not in table.columns and "algorithmic_medoid" not in table.columns:
        raise KeyError("Override CSV must contain cluster_id or algorithmic_medoid.")
    if "scope" not in table.columns:
        table["scope"] = "final"
    table["scope"] = table["scope"].astype(str)
    table["representative_feature"] = table["representative_feature"].astype(str)
    return table[table["scope"].isin(["final", "all"])].copy()


def build_clusters(
    X: pd.DataFrame,
    *,
    correlation_method: str,
    correlation_threshold: float,
    linkage_method: str,
    min_nonmissing: int,
    overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, filter_audit = filter_usable_features(X, min_nonmissing)
    frame = X[features].copy()
    missing_fraction = frame.isna().mean()
    n_nonmissing = frame.notna().sum()

    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    imputed = np.asarray(imputer.fit_transform(frame), dtype=float)
    imputed_frame = pd.DataFrame(imputed, columns=features, index=frame.index)
    variance = imputed_frame.var(axis=0, ddof=0)

    correlation = imputed_frame.corr(method=correlation_method).abs()
    correlation = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    correlation = correlation.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(correlation.values, 1.0)

    if len(features) == 1:
        raw_ids = np.asarray([1], dtype=int)
    else:
        distance = (1.0 - correlation).clip(lower=0.0, upper=1.0)
        np.fill_diagonal(distance.values, 0.0)
        condensed = squareform(distance.to_numpy(float), checks=False)
        hierarchy = linkage(condensed, method=linkage_method)
        raw_ids = fcluster(
            hierarchy,
            t=1.0 - correlation_threshold,
            criterion="distance",
        )

    raw_groups: dict[int, list[str]] = {}
    for feature, raw_id in zip(features, raw_ids):
        raw_groups.setdefault(int(raw_id), []).append(feature)
    ordered_raw_ids = sorted(
        raw_groups,
        key=lambda raw_id: (min(raw_groups[raw_id]), len(raw_groups[raw_id])),
    )
    remap = {raw_id: position + 1 for position, raw_id in enumerate(ordered_raw_ids)}
    cluster_by_feature = {
        feature: remap[int(raw_id)] for feature, raw_id in zip(features, raw_ids)
    }

    membership_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for cluster_id in sorted(set(cluster_by_feature.values())):
        members = sorted(
            feature for feature, cid in cluster_by_feature.items() if cid == cluster_id
        )
        candidate_stats: list[dict[str, Any]] = []

        for feature in members:
            others = [item for item in members if item != feature]
            if others:
                values = correlation.loc[feature, others].to_numpy(float)
                mean_abs_corr = float(np.mean(values))
                min_abs_corr = float(np.min(values))
                max_abs_corr = float(np.max(values))
            else:
                mean_abs_corr = min_abs_corr = max_abs_corr = 1.0
            candidate_stats.append(
                {
                    "feature": feature,
                    "mean_distance_to_cluster": float(1.0 - mean_abs_corr),
                    "mean_abs_correlation_to_cluster": mean_abs_corr,
                    "min_abs_correlation_to_cluster": min_abs_corr,
                    "max_abs_correlation_to_cluster": max_abs_corr,
                    "missing_fraction": float(missing_fraction[feature]),
                    "n_nonmissing": int(n_nonmissing[feature]),
                    "variance_after_imputation": float(variance[feature]),
                }
            )

        candidate_stats.sort(
            key=lambda row: (
                row["mean_distance_to_cluster"],
                row["missing_fraction"],
                -row["variance_after_imputation"],
                row["feature"],
            )
        )
        algorithmic_medoid = str(candidate_stats[0]["feature"])
        selected = algorithmic_medoid
        source = "algorithmic_medoid"

        if not overrides.empty:
            matching = pd.DataFrame()
            if "cluster_id" in overrides.columns:
                ids = pd.to_numeric(overrides["cluster_id"], errors="coerce")
                matching = overrides[ids == cluster_id]
            if matching.empty and "algorithmic_medoid" in overrides.columns:
                matching = overrides[
                    overrides["algorithmic_medoid"].astype(str) == algorithmic_medoid
                ]
            if len(matching) > 1:
                raise ValueError(f"Multiple overrides matched final cluster {cluster_id}.")
            if len(matching) == 1:
                replacement = str(matching.iloc[0]["representative_feature"])
                if replacement not in members:
                    raise ValueError(
                        f"Override {replacement!r} is not in cluster {cluster_id}: {members}"
                    )
                selected = replacement
                source = "user_override"

        members_json = json.dumps(members)
        candidate_order = [row["feature"] for row in candidate_stats]
        pair_values: list[float] = []
        for left_position, left in enumerate(members):
            for right in members[left_position + 1 :]:
                value = float(correlation.loc[left, right])
                pair_values.append(value)
                pair_rows.append(
                    {
                        "scope": "final",
                        "cluster_id": cluster_id,
                        "feature_a": left,
                        "feature_b": right,
                        "absolute_correlation": value,
                        "correlation_method": correlation_method,
                    }
                )

        summary_rows.append(
            {
                "scope": "final",
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "algorithmic_medoid": algorithmic_medoid,
                "selected_representative": selected,
                "representative_source": source,
                "mean_pairwise_abs_correlation": (
                    float(np.mean(pair_values)) if pair_values else 1.0
                ),
                "minimum_pairwise_abs_correlation": (
                    float(np.min(pair_values)) if pair_values else 1.0
                ),
                "maximum_pairwise_abs_correlation": (
                    float(np.max(pair_values)) if pair_values else 1.0
                ),
                "members_json": members_json,
                "medoid_candidate_order_json": json.dumps(candidate_order),
            }
        )

        for rank, row in enumerate(candidate_stats, start=1):
            candidate_rows.append(
                {
                    "scope": "final",
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "medoid_candidate_rank": rank,
                    **row,
                    "algorithmic_medoid": algorithmic_medoid,
                    "selected_representative": selected,
                    "is_algorithmic_medoid": row["feature"] == algorithmic_medoid,
                    "is_selected_representative": row["feature"] == selected,
                    "members_json": members_json,
                }
            )
            membership_rows.append(
                {
                    "scope": "final",
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "feature": row["feature"],
                    "algorithmic_medoid": algorithmic_medoid,
                    "selected_representative": selected,
                    "representative_source": source,
                    "is_algorithmic_medoid": row["feature"] == algorithmic_medoid,
                    "is_selected_representative": row["feature"] == selected,
                    "medoid_candidate_rank": rank,
                    "mean_distance_to_cluster": row["mean_distance_to_cluster"],
                    "mean_abs_correlation_to_cluster": row[
                        "mean_abs_correlation_to_cluster"
                    ],
                    "min_abs_correlation_to_cluster": row[
                        "min_abs_correlation_to_cluster"
                    ],
                    "max_abs_correlation_to_cluster": row[
                        "max_abs_correlation_to_cluster"
                    ],
                    "missing_fraction": row["missing_fraction"],
                    "n_nonmissing": row["n_nonmissing"],
                    "variance_after_imputation": row["variance_after_imputation"],
                    "members_json": members_json,
                }
            )

    membership = pd.DataFrame(membership_rows).sort_values(
        ["cluster_id", "medoid_candidate_rank", "feature"]
    )
    summary = pd.DataFrame(summary_rows).sort_values("cluster_id")
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["cluster_id", "medoid_candidate_rank"]
    )
    pairs = pd.DataFrame(pair_rows)
    return membership, summary, candidates, pairs, correlation, filter_audit


def main() -> None:
    args = parse_args()
    if not (0.0 < args.correlation_threshold <= 1.0):
        raise ValueError("--correlation-threshold must be in (0, 1].")
    if args.min_nonmissing < 2:
        raise ValueError("--min-nonmissing must be at least 2.")

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir, cluster_dir = prepare_cluster_dir(args.output_dir, args.overwrite)

    frame = pd.read_csv(input_path, low_memory=False)
    feature_names = infer_feature_columns(frame, args)
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    overrides = load_overrides(args.representative_overrides)

    print(
        f"Clustering {len(feature_names)} candidate descriptors across {len(frame)} "
        f"compounds using |{args.correlation_method}| >= "
        f"{args.correlation_threshold:.3f} and {args.linkage_method} linkage."
    )

    membership, summary, candidates, pairs, correlation, filter_audit = build_clusters(
        X,
        correlation_method=args.correlation_method,
        correlation_threshold=args.correlation_threshold,
        linkage_method=args.linkage_method,
        min_nonmissing=args.min_nonmissing,
        overrides=overrides,
    )

    membership_path = cluster_dir / "final_cluster_membership.csv"
    membership.to_csv(membership_path, index=False)
    summary.to_csv(cluster_dir / "final_cluster_summary.csv", index=False)
    candidates.to_csv(cluster_dir / "final_medoid_candidates.csv", index=False)
    filter_audit.to_csv(cluster_dir / "final_feature_filter_audit.csv", index=False)

    representatives = summary["selected_representative"].astype(str).tolist()
    (cluster_dir / "final_selected_representatives.txt").write_text(
        "\n".join(representatives) + "\n"
    )

    override_template = summary[
        ["scope", "cluster_id", "algorithmic_medoid", "selected_representative"]
    ].copy()
    override_template["representative_feature"] = override_template[
        "selected_representative"
    ]
    override_template.to_csv(
        cluster_dir / "final_representative_override_template.csv", index=False
    )

    if not args.no_save_within_cluster_pairs:
        pairs.to_csv(
            cluster_dir / "final_within_cluster_pairwise_correlations.csv.gz",
            index=False,
            compression="gzip",
        )
    if args.save_correlation_matrix:
        correlation.to_csv(
            cluster_dir / "final_absolute_correlation_matrix.csv.gz",
            compression="gzip",
        )

    manifest = {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_dir": str(output_dir),
        "cluster_dir": str(cluster_dir),
        "n_compounds": int(len(frame)),
        "n_candidate_features": int(len(feature_names)),
        "n_included_features": int(len(membership)),
        "n_clusters": int(summary["cluster_id"].nunique()),
        "n_singleton_clusters": int((summary["cluster_size"] == 1).sum()),
        "uses_target_or_class_labels": False,
        "distance": "1 - absolute feature correlation",
        "correlation_method": args.correlation_method,
        "absolute_correlation_threshold": args.correlation_threshold,
        "linkage_method": args.linkage_method,
        "representative_method": "cluster_medoid",
        "min_nonmissing": args.min_nonmissing,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (cluster_dir / "final_clustering_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )

    print("\nCompleted target-independent clustering.")
    print(f"Included features: {len(membership)}")
    print(f"Clusters formed: {summary['cluster_id'].nunique()}")
    print(f"Grouped-SHAP membership file: {membership_path}")
    print("No predictive models were trained or loaded.")


if __name__ == "__main__":
    main()
