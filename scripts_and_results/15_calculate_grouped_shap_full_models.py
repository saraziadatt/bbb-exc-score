#!/usr/bin/env python3
"""
Correlation-aware grouped TreeSHAP for the saved full-feature BBB models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.pipeline import Pipeline


MODEL_MSE = "lightgbm_mse_regression"
MODEL_MTL = "custom_mtl_original"

EXACT_METADATA_NAMES = {
    "logbb",
    "bbb_class",
    "class",
    "outer_fold",
    "smiles",
    "canonical_smiles",
    "sample_id",
    "compound_id",
    "id",
    "has_logbb",
    "logbb_is_measured",
    "outer_stratum",
}
EXCLUDED_PREFIXES = (
    "inner_fold_outer_",
    "pred_",
    "prediction",
    "oof_",
    "shap_",
)


@dataclass
class DataBundle:
    frame: pd.DataFrame
    X: pd.DataFrame
    feature_names: list[str]
    y_reg: np.ndarray
    y_class: np.ndarray
    outer_fold: np.ndarray
    sample_id: np.ndarray
    smiles: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate cluster-aggregated TreeSHAP for saved full-feature "
            "LightGBM outer-fold models selected by name and directory."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--full-results-dir", type=Path,
        help="Shared results directory; use --model-dirs for separate directories.",
    )
    parser.add_argument(
        "--model-dirs", nargs="+", type=Path,
        help="One results/fitted_models directory per --models entry, in the same order.",
    )
    parser.add_argument("--cluster-results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./shap/grouped_shap_full_models"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Saved model filename prefixes, without _outer_<fold> or extension.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--save-local-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save compressed per-fold grouped SHAP matrices. These permit "
            "later local analyses without recalculating TreeSHAP."
        ),
    )
    parser.add_argument(
        "--save-local-top-n",
        type=int,
        default=20,
        help=(
            "Save a pooled long table of local OOF grouped SHAP values for "
            "the union of leading groups. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--multiple-testing",
        choices=["bh", "holm", "none"],
        default="bh",
        help=(
            "Adjustment applied only to MTL-minus-MSE Nadeau-Bengio p-values."
        ),
    )
    parser.add_argument("--target-col", default="logBB")
    parser.add_argument("--class-col", default="bbb_class")
    parser.add_argument("--fold-col", default="outer_fold")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("--models must contain unique names.")
    if any("/" in name or "\\" in name or name in {".", ".."} for name in args.models):
        parser.error("--models entries must be filename prefixes, not paths.")
    if args.model_dirs is not None:
        if args.full_results_dir is not None:
            parser.error("Use either --model-dirs or --full-results-dir.")
        if len(args.model_dirs) != len(args.models):
            parser.error("Supply exactly one --model-dirs entry per --models entry.")
    elif args.full_results_dir is not None:
        args.model_dirs = [args.full_results_dir] * len(args.models)
    else:
        parser.error("Supply --model-dirs or --full-results-dir.")
    return args


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir()
    (path / "rankings").mkdir()
    (path / "local_group_shap").mkdir()
    return path


def infer_descriptor_columns(
    frame: pd.DataFrame,
    *,
    target_col: str,
    class_col: str,
    fold_col: str,
    smiles_col: str,
    id_col: str,
) -> list[str]:
    metadata = {
        *EXACT_METADATA_NAMES,
        target_col.lower(),
        class_col.lower(),
        fold_col.lower(),
        smiles_col.lower(),
        id_col.lower(),
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
    return features


def load_data(args: argparse.Namespace) -> DataBundle:
    path = args.input.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False, float_precision="round_trip",) 

    required = {args.target_col, args.class_col, args.fold_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    if args.id_col not in frame.columns:
        frame[args.id_col] = np.arange(len(frame), dtype=int)
    if args.smiles_col not in frame.columns:
        frame[args.smiles_col] = ""

    feature_names = infer_descriptor_columns(
        frame,
        target_col=args.target_col,
        class_col=args.class_col,
        fold_col=args.fold_col,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
    )
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")

    return DataBundle(
        frame=frame,
        X=X,
        feature_names=feature_names,
        y_reg=pd.to_numeric(
            frame[args.target_col], errors="coerce"
        ).to_numpy(float),
        y_class=pd.to_numeric(
            frame[args.class_col], errors="raise"
        ).to_numpy(int),
        outer_fold=pd.to_numeric(
            frame[args.fold_col], errors="raise"
        ).to_numpy(int),
        sample_id=frame[args.id_col].to_numpy(),
        smiles=frame[args.smiles_col].astype(str).to_numpy(),
    )


def load_cluster_membership(cluster_results_dir: Path) -> pd.DataFrame:
    path = (
        cluster_results_dir.resolve()
        / "clusters"
        / "final_cluster_membership.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    membership = pd.read_csv(path)
    required = {
        "cluster_id",
        "cluster_size",
        "feature",
        "algorithmic_medoid",
        "selected_representative",
        "medoid_candidate_rank",
    }
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")

    membership["feature"] = membership["feature"].astype(str)
    membership["algorithmic_medoid"] = (
        membership["algorithmic_medoid"].astype(str)
    )
    membership["selected_representative"] = (
        membership["selected_representative"].astype(str)
    )
    membership["cluster_key"] = membership["cluster_id"].map(
        lambda value: f"cluster_{int(value)}"
    )
    membership["cluster_label"] = membership.apply(
        lambda row: (
            f"Cluster {int(row['cluster_id'])}: "
            f"{row['algorithmic_medoid']} "
            f"(n={int(row['cluster_size'])})"
        ),
        axis=1,
    )
    return membership.sort_values(
        ["cluster_id", "medoid_candidate_rank", "feature"]
    ).reset_index(drop=True)


def build_global_group_metadata(
    membership: pd.DataFrame,
    dataset_features: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    feature_to_group = dict(
        zip(membership["feature"], membership["cluster_key"])
    )

    metadata_rows: list[dict[str, Any]] = []
    for cluster_key, group in membership.groupby("cluster_key", sort=False):
        first = group.iloc[0]
        members = group.sort_values(
            ["medoid_candidate_rank", "feature"]
        )["feature"].astype(str).tolist()
        metadata_rows.append(
            {
                "cluster_key": cluster_key,
                "cluster_id": int(first["cluster_id"]),
                "cluster_label": str(first["cluster_label"]),
                "algorithmic_medoid": str(first["algorithmic_medoid"]),
                "selected_representative": str(
                    first["selected_representative"]
                ),
                "cluster_size_defined": int(first["cluster_size"]),
                "members_json": json.dumps(members),
            }
        )

    missing_features = sorted(set(dataset_features).difference(feature_to_group))
    for feature in missing_features:
        cluster_key = f"unclustered::{feature}"
        feature_to_group[feature] = cluster_key
        metadata_rows.append(
            {
                "cluster_key": cluster_key,
                "cluster_id": np.nan,
                "cluster_label": f"Unclustered: {feature} (n=1)",
                "algorithmic_medoid": feature,
                "selected_representative": feature,
                "cluster_size_defined": 1,
                "members_json": json.dumps([feature]),
            }
        )

    metadata = pd.DataFrame(metadata_rows).drop_duplicates("cluster_key")
    return metadata, feature_to_group


def model_paths(
    full_results_dir: Path,
    model_name: str,
    outer_fold: int,
) -> tuple[Path, Path | None]:
    directory = full_results_dir.resolve()
    fitted_dir = directory / "fitted_models"
    if not fitted_dir.is_dir():
        fitted_dir = directory
    stem = f"{model_name}_outer_{outer_fold}"
    candidates = [fitted_dir / f"{stem}{suffix}" for suffix in (".joblib", ".txt")]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No saved model found; tried: {candidates}")
    if len(existing) > 1:
        raise ValueError(f"Ambiguous saved model: {existing}. Keep one format per fold.")
    sidecar = fitted_dir / f"{stem}_preprocessor.joblib"
    return existing[0], sidecar if sidecar.is_file() else None


def get_lightgbm_booster(estimator: Any) -> lgb.Booster:
    if isinstance(estimator, lgb.Booster):
        return estimator
    booster = getattr(estimator, "booster_", None)
    if isinstance(booster, lgb.Booster):
        return booster
    booster = getattr(estimator, "_Booster", None)
    if isinstance(booster, lgb.Booster):
        return booster
    raise TypeError(
        f"Could not obtain a LightGBM Booster from {type(estimator)!r}."
    )


class SavedShapModel:
    """Load, transform, predict, and calculate native LightGBM TreeSHAP."""

    def __init__(
        self,
        *,
        model_name: str,
        full_results_dir: Path,
        outer_fold: int,
        fallback_feature_names: list[str],
    ) -> None:
        self.model_name = model_name
        self.outer_fold = int(outer_fold)
        model_path, preprocessor_path = model_paths(
            full_results_dir,
            model_name,
            outer_fold,
        )
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        fitted = (
            joblib.load(model_path) if model_path.suffix == ".joblib"
            else lgb.Booster(model_file=str(model_path))
        )
        self.pipeline = fitted if isinstance(fitted, Pipeline) else None
        self.preprocessor = None
        if self.pipeline is not None:
            self.booster = get_lightgbm_booster(fitted.steps[-1][1])
            original_names = getattr(fitted, "feature_names_in_", None)
            self.input_feature_names = (
                list(map(str, original_names)) if original_names is not None
                else list(fallback_feature_names)
            )
            self.output_feature_names = list(self.input_feature_names)
            for _, transformer in fitted.steps[:-1]:
                if transformer is None or (isinstance(transformer, str) and transformer == "passthrough"):
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    self.output_feature_names = list(map(str,
                        transformer.get_feature_names_out(self.output_feature_names)))
                elif hasattr(transformer, "get_support"):
                    self.output_feature_names = np.asarray(self.output_feature_names)[
                        transformer.get_support()].tolist()
                else:
                    raise TypeError(
                        f"Cannot establish descriptor names after {type(transformer)!r}."
                    )
        else:
            self.booster = get_lightgbm_booster(fitted)
            if preprocessor_path is not None:
                self.preprocessor = joblib.load(preprocessor_path)
                if not isinstance(self.preprocessor, dict) or "feature_names" not in self.preprocessor:
                    raise TypeError(f"Expected a preprocessor dictionary with feature_names: {preprocessor_path}")
                self.input_feature_names = list(map(str, self.preprocessor["feature_names"]))
            else:
                booster_names = list(map(str, self.booster.feature_name()))
                generic = all(re.fullmatch(r"Column_\d+", name) for name in booster_names)
                if generic:
                    raise ValueError(
                        f"{model_path} has generic feature names; supply its "
                        "_preprocessor.joblib with the training feature_names."
                    )
                self.input_feature_names = booster_names
            self.output_feature_names = list(self.input_feature_names)

        if len(self.output_feature_names) != self.booster.num_feature():
            raise ValueError(
                f"Feature-name count ({len(self.output_feature_names)}) does "
                f"not match booster feature count ({self.booster.num_feature()}) "
                f"for {model_name}, fold {outer_fold}."
            )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.input_feature_names).difference(X.columns))
        if missing:
            raise KeyError(
                f"Input data are missing {len(missing)} features required by "
                f"{self.model_name}, fold {self.outer_fold}: {missing[:20]}"
            )

        matrix = X[self.input_feature_names]
        if self.pipeline is not None:
            if len(self.pipeline.steps) > 1:
                matrix = self.pipeline[:-1].transform(matrix)
        elif self.preprocessor is not None:
            imputer = self.preprocessor.get("imputer")
            if imputer is not None:
                matrix = imputer.transform(matrix)
        return np.asarray(matrix, dtype=float)

    def predict_and_shap(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = self.transform(X)
        prediction = np.asarray(
            self.booster.predict(matrix, raw_score=True),
            dtype=float,
        )
        contributions = np.asarray(
            self.booster.predict(matrix, pred_contrib=True),
            dtype=float,
        )
        expected_columns = matrix.shape[1] + 1
        if (
            contributions.ndim != 2
            or contributions.shape[1] != expected_columns
        ):
            raise ValueError(
                f"Unexpected TreeSHAP shape {contributions.shape}; expected "
                f"({len(matrix)}, {expected_columns})."
            )
        return prediction, contributions[:, :-1], contributions[:, -1]


def group_shap_values(
    feature_shap: np.ndarray,
    feature_names: list[str],
    feature_to_group: dict[str, str],
    group_metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    group_to_indices: dict[str, list[int]] = {}
    for index, feature in enumerate(feature_names):
        cluster_key = feature_to_group.get(
            feature,
            f"unclustered::{feature}",
        )
        group_to_indices.setdefault(cluster_key, []).append(index)

    metadata_index = group_metadata.set_index("cluster_key", drop=False)
    ordered_keys = sorted(
        group_to_indices,
        key=lambda key: (
            1 if key.startswith("unclustered::") else 0,
            (
                float(metadata_index.loc[key, "cluster_id"])
                if key in metadata_index.index
                and pd.notna(metadata_index.loc[key, "cluster_id"])
                else np.inf
            ),
            key,
        ),
    )

    grouped = np.empty(
        (feature_shap.shape[0], len(ordered_keys)),
        dtype=float,
    )
    mapping_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for group_position, cluster_key in enumerate(ordered_keys):
        indices = group_to_indices[cluster_key]
        grouped[:, group_position] = feature_shap[:, indices].sum(axis=1)

        if cluster_key in metadata_index.index:
            metadata_row = metadata_index.loc[cluster_key]
            cluster_label = str(metadata_row["cluster_label"])
            cluster_id = metadata_row["cluster_id"]
            algorithmic_medoid = str(metadata_row["algorithmic_medoid"])
            selected_representative = str(
                metadata_row["selected_representative"]
            )
            cluster_size_defined = int(
                metadata_row["cluster_size_defined"]
            )
            members_json = str(metadata_row["members_json"])
        else:
            feature = feature_names[indices[0]]
            cluster_label = f"Unclustered: {feature} (n=1)"
            cluster_id = np.nan
            algorithmic_medoid = feature
            selected_representative = feature
            cluster_size_defined = 1
            members_json = json.dumps([feature])

        used_features = [feature_names[index] for index in indices]
        for feature_index in indices:
            mapping_rows.append(
                {
                    "cluster_key": cluster_key,
                    "cluster_id": cluster_id,
                    "cluster_label": cluster_label,
                    "feature": feature_names[feature_index],
                    "feature_position_in_booster": int(feature_index),
                    "cluster_size_defined": cluster_size_defined,
                    "cluster_size_used_by_model": len(indices),
                    "algorithmic_medoid": algorithmic_medoid,
                    "selected_representative": selected_representative,
                    "members_defined_json": members_json,
                    "members_used_json": json.dumps(used_features),
                }
            )

        abs_after_sum = np.abs(grouped[:, group_position])
        sum_abs_members = np.sum(
            np.abs(feature_shap[:, indices]),
            axis=1,
        )
        denominator = float(np.mean(sum_abs_members))
        grouped_importance = float(np.mean(abs_after_sum))
        diagnostic_rows.append(
            {
                "cluster_key": cluster_key,
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "algorithmic_medoid": algorithmic_medoid,
                "selected_representative": selected_representative,
                "cluster_size_defined": cluster_size_defined,
                "cluster_size_used_by_model": len(indices),
                "members_defined_json": members_json,
                "members_used_json": json.dumps(used_features),
                "mean_abs_group_shap": grouped_importance,
                "sd_abs_group_shap_across_compounds": float(
                    np.std(abs_after_sum, ddof=0)
                ),
                "mean_signed_group_shap": float(
                    np.mean(grouped[:, group_position])
                ),
                "mean_sum_abs_member_shap": denominator,
                "within_group_retained_fraction": (
                    grouped_importance / denominator
                    if denominator > 0
                    else np.nan
                ),
                "within_group_cancellation_fraction": (
                    1.0 - grouped_importance / denominator
                    if denominator > 0
                    else np.nan
                ),
            }
        )

    group_table = pd.DataFrame(diagnostic_rows)
    total = float(group_table["mean_abs_group_shap"].sum())
    group_table["normalized_mean_abs_group_shap"] = (
        group_table["mean_abs_group_shap"] / total
        if total > 0
        else np.nan
    )
    return grouped, pd.DataFrame(mapping_rows), group_table


def benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices) == 0:
        return adjusted

    ordered_indices = finite_indices[
        np.argsort(p_values[finite_indices], kind="mergesort")
    ]
    ordered_p = p_values[ordered_indices]
    m = len(ordered_p)
    ordered_adjusted = np.empty(m, dtype=float)
    running_min = 1.0
    for reverse_index in range(m - 1, -1, -1):
        rank = reverse_index + 1
        candidate = ordered_p[reverse_index] * m / rank
        running_min = min(running_min, candidate)
        ordered_adjusted[reverse_index] = min(running_min, 1.0)
    adjusted[ordered_indices] = ordered_adjusted
    return adjusted


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices) == 0:
        return adjusted

    ordered_indices = finite_indices[
        np.argsort(p_values[finite_indices], kind="mergesort")
    ]
    ordered_p = p_values[ordered_indices]
    running_max = 0.0
    for rank, (index, p_value) in enumerate(
        zip(ordered_indices, ordered_p)
    ):
        candidate = (len(ordered_p) - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def adjust_p_values(
    p_values: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "bh":
        return benjamini_hochberg_adjust(p_values)
    if method == "holm":
        return holm_adjust(p_values)
    if method == "none":
        return np.asarray(p_values, dtype=float)
    raise ValueError(method)


def nadeau_bengio_corrected_paired_ttest(
    first: np.ndarray,
    second: np.ndarray,
    *,
    mean_test_train_ratio: float,
) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    differences = first[valid] - second[valid]
    n = len(differences)

    empty = {
        "mean_difference": np.nan,
        "difference_sd": np.nan,
        "mean_test_train_ratio": float(mean_test_train_ratio),
        "nadeau_bengio_correction_factor": np.nan,
        "nadeau_bengio_standard_error": np.nan,
        "nadeau_bengio_t_statistic": np.nan,
        "nadeau_bengio_degrees_of_freedom": np.nan,
        "p_value_two_sided_nadeau_bengio": np.nan,
    }
    if n < 2 or not np.isfinite(mean_test_train_ratio):
        return empty

    mean_difference = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=1))
    variance = float(np.var(differences, ddof=1))
    correction_factor = (1.0 / n) + mean_test_train_ratio

    if variance == 0.0:
        if mean_difference == 0.0:
            standard_error = 0.0
            t_statistic = 0.0
            p_value = 1.0
        else:
            standard_error = 0.0
            t_statistic = math.copysign(np.inf, mean_difference)
            p_value = 0.0
    else:
        standard_error = math.sqrt(correction_factor * variance)
        t_statistic = mean_difference / standard_error
        p_value = float(
            2.0 * student_t.sf(abs(t_statistic), df=n - 1)
        )

    return {
        "mean_difference": mean_difference,
        "difference_sd": difference_sd,
        "mean_test_train_ratio": float(mean_test_train_ratio),
        "nadeau_bengio_correction_factor": float(correction_factor),
        "nadeau_bengio_standard_error": float(standard_error),
        "nadeau_bengio_t_statistic": float(t_statistic),
        "nadeau_bengio_degrees_of_freedom": float(n - 1),
        "p_value_two_sided_nadeau_bengio": float(p_value),
    }


def format_probability(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def plot_model_importance(
    summary: pd.DataFrame,
    *,
    model_name: str,
    scale: str,
    output_path: Path,
    top_n: int,
) -> None:
    if scale == "raw":
        mean_column = "fold_mean_mean_abs_group_shap"
        sd_column = "fold_sd_mean_abs_group_shap"
        xlabel = "Mean absolute grouped SHAP value"
    elif scale == "normalized":
        mean_column = "fold_mean_normalized_group_shap"
        sd_column = "fold_sd_normalized_group_shap"
        xlabel = "Normalized mean absolute grouped SHAP importance"
    else:
        raise ValueError(scale)

    plot_data = (
        summary[summary["model"] == model_name]
        .sort_values(mean_column, ascending=False)
        .head(top_n)
        .sort_values(mean_column, ascending=True)
        .copy()
    )
    if plot_data.empty:
        return

    figure_height = max(0, 0.2 * len(plot_data) + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, figure_height))
    ax.errorbar(
        plot_data[mean_column],
        plot_data["cluster_label"],
        xerr=plot_data[sd_column],
        fmt="o",
        color="#3B9ED8",
        ecolor="#B8B8B8",
        elinewidth=1.0,
        capsize=0,
        markeredgecolor="#2C526A",
        markeredgewidth=0.7,
        markersize=5.5,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correlated descriptor cluster")
    ax.set_title(model_name)
    ax.grid(axis="x", alpha=0.18, linestyle="--", linewidth=0.7)
    ax.grid(axis="y", alpha=0.08, linestyle=":", linewidth=0.6)
    # ax.text(
    #     1.0,
    #     -0.13,
    #     "Error bars: SD across outer-fold mean grouped SHAP values",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=8.3,
    # )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_difference(
    differences: pd.DataFrame,
    *,
    scale: str,
    adjustment_label: str,
    output_path: Path,
    top_n: int,
) -> None:
    plot_data = (
        differences[differences["importance_scale"] == scale]
        .assign(
            absolute_difference=lambda frame: np.abs(
                frame["mtl_minus_mse"]
            )
        )
        .sort_values("absolute_difference", ascending=False)
        .head(top_n)
        .sort_values("mtl_minus_mse", ascending=True)
        .copy()
    )
    if plot_data.empty:
        return

    if scale == "raw":
        xlabel = (
            "Grouped SHAP importance difference: custom MTL − "
            "LightGBM MSE"
        )
    else:
        xlabel = (
            "Normalized grouped SHAP importance difference: custom MTL − "
            "LightGBM MSE"
        )

    figure_height = max(0, 0.46 * len(plot_data) + 1.4)
    fig, ax = plt.subplots(figsize=(12.5, figure_height))
    ax.errorbar(
        plot_data["mtl_minus_mse"],
        plot_data["cluster_label"],
        xerr=plot_data["fold_sd_difference"],
        fmt="o",
        color="#3B9ED8",
        ecolor="#B8B8B8",
        elinewidth=1.0,
        capsize=0,
        markeredgecolor="#2C526A",
        markeredgewidth=0.7,
        markersize=5.5,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correlated descriptor cluster")
    ax.grid(axis="x", alpha=0.18, linestyle="--", linewidth=0.7)
    ax.grid(axis="y", alpha=0.08, linestyle=":", linewidth=0.6)

    finite = plot_data["mtl_minus_mse"].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    span = max(
        float(np.ptp(finite)) if len(finite) > 1 else 0.0,
        float(np.max(np.abs(finite))) if len(finite) else 1.0,
        1e-8,
    )
    right_edge = float(
        np.nanmax(
            plot_data["mtl_minus_mse"]
            + plot_data["fold_sd_difference"].fillna(0)
        )
    )
    text_x = right_edge + 0.12 * span

    for y_position, row in enumerate(plot_data.itertuples(index=False)):
        text = (
            "NB p = "
            f"{format_probability(row.p_value_two_sided_nadeau_bengio)}; "
            f"{adjustment_label} = "
            f"{format_probability(row.p_value_adjusted)}"
        )
        ax.text(
            text_x,
            y_position,
            text,
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight=(
                "bold"
                if np.isfinite(row.p_value_adjusted)
                and row.p_value_adjusted < 0.05
                else "normal"
            ),
        )

    left, right = ax.get_xlim()
    ax.set_xlim(left, max(right, text_x + 0.34 * span))
    ax.text(
        1.0,
        -0.13,
        (
            "Two-sided Nadeau–Bengio corrected paired t-test across matched "
            f"outer folds; bold: {adjustment_label} < 0.05"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_fold_importance(
    fold_table: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "model",
        "cluster_key",
        "cluster_id",
        "cluster_label",
        "algorithmic_medoid",
        "selected_representative",
        "cluster_size_defined",
        "members_defined_json",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in fold_table.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        row = dict(zip(group_columns, keys))
        weights = group["n_test_compounds"].to_numpy(float)
        raw = group["mean_abs_group_shap"].to_numpy(float)
        normalized = group[
            "normalized_mean_abs_group_shap"
        ].to_numpy(float)
        signed = group["mean_signed_group_shap"].to_numpy(float)
        cancellation = group[
            "within_group_cancellation_fraction"
        ].to_numpy(float)

        row.update(
            {
                "n_outer_folds": int(group["outer_fold"].nunique()),
                "n_oof_compounds": int(np.sum(weights)),
                "pooled_oof_mean_abs_group_shap": float(
                    np.average(raw, weights=weights)
                ),
                "fold_mean_mean_abs_group_shap": float(np.mean(raw)),
                "fold_sd_mean_abs_group_shap": float(
                    np.std(raw, ddof=1)
                ),
                "fold_mean_normalized_group_shap": float(
                    np.mean(normalized)
                ),
                "fold_sd_normalized_group_shap": float(
                    np.std(normalized, ddof=1)
                ),
                "pooled_oof_mean_signed_group_shap": float(
                    np.average(signed, weights=weights)
                ),
                "fold_mean_cancellation_fraction": float(
                    np.nanmean(cancellation)
                ),
                "minimum_fold_mean_abs_group_shap": float(np.min(raw)),
                "maximum_fold_mean_abs_group_shap": float(np.max(raw)),
                "fraction_folds_positive_mean_signed": float(
                    np.mean(signed > 0)
                ),
                "minimum_cluster_size_used_by_model": int(
                    group["cluster_size_used_by_model"].min()
                ),
                "maximum_cluster_size_used_by_model": int(
                    group["cluster_size_used_by_model"].max()
                ),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["raw_importance_rank_within_model"] = (
        summary.groupby("model")["fold_mean_mean_abs_group_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["normalized_importance_rank_within_model"] = (
        summary.groupby("model")["fold_mean_normalized_group_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary.sort_values(
        ["model", "raw_importance_rank_within_model", "cluster_label"]
    )


def calculate_model_differences(
    fold_table: pd.DataFrame,
    data: DataBundle,
    adjustment_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    fold_ratios: dict[int, float] = {}
    for fold in sorted(np.unique(data.outer_fold)):
        n_test = int(np.sum(data.outer_fold == fold))
        n_train = int(np.sum(data.outer_fold != fold))
        fold_ratios[int(fold)] = n_test / n_train

    for cluster_key, cluster_group in fold_table.groupby(
        "cluster_key",
        sort=False,
    ):
        reference = cluster_group.iloc[0]
        for scale, column in (
            ("raw", "mean_abs_group_shap"),
            ("normalized", "normalized_mean_abs_group_shap"),
        ):
            mse = cluster_group[
                cluster_group["model"] == MODEL_MSE
            ][["outer_fold", column]].rename(
                columns={column: "mse_importance"}
            )
            mtl = cluster_group[
                cluster_group["model"] == MODEL_MTL
            ][["outer_fold", column]].rename(
                columns={column: "mtl_importance"}
            )
            paired = mse.merge(
                mtl,
                on="outer_fold",
                how="inner",
                validate="one_to_one",
            )
            if len(paired) < 2:
                continue

            mean_ratio = float(
                np.mean(
                    [
                        fold_ratios[int(fold)]
                        for fold in paired["outer_fold"]
                    ]
                )
            )
            test = nadeau_bengio_corrected_paired_ttest(
                paired["mtl_importance"].to_numpy(float),
                paired["mse_importance"].to_numpy(float),
                mean_test_train_ratio=mean_ratio,
            )
            differences = (
                paired["mtl_importance"] - paired["mse_importance"]
            ).to_numpy(float)
            rows.append(
                {
                    "cluster_key": cluster_key,
                    "cluster_id": reference["cluster_id"],
                    "cluster_label": reference["cluster_label"],
                    "algorithmic_medoid": reference[
                        "algorithmic_medoid"
                    ],
                    "selected_representative": reference[
                        "selected_representative"
                    ],
                    "cluster_size_defined": reference[
                        "cluster_size_defined"
                    ],
                    "members_defined_json": reference[
                        "members_defined_json"
                    ],
                    "importance_scale": scale,
                    "n_paired_outer_folds": len(paired),
                    "mse_mean_importance": float(
                        paired["mse_importance"].mean()
                    ),
                    "mtl_mean_importance": float(
                        paired["mtl_importance"].mean()
                    ),
                    "mtl_minus_mse": float(np.mean(differences)),
                    "fold_sd_difference": float(
                        np.std(differences, ddof=1)
                    ),
                    "minimum_fold_difference": float(
                        np.min(differences)
                    ),
                    "maximum_fold_difference": float(
                        np.max(differences)
                    ),
                    "fraction_folds_mtl_greater": float(
                        np.mean(differences > 0)
                    ),
                    **test,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["p_value_adjusted"] = np.nan
    for _, indices in result.groupby("importance_scale").groups.items():
        indices = np.asarray(list(indices), dtype=int)
        result.loc[indices, "p_value_adjusted"] = adjust_p_values(
            result.loc[
                indices,
                "p_value_two_sided_nadeau_bengio",
            ].to_numpy(float),
            adjustment_method,
        )
    result["significant_adjusted_0_05"] = (
        result["p_value_adjusted"] < 0.05
    )
    result["absolute_mtl_minus_mse"] = np.abs(
        result["mtl_minus_mse"]
    )
    result["difference_rank_by_absolute_value"] = (
        result.groupby("importance_scale")["absolute_mtl_minus_mse"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result.sort_values(
        [
            "importance_scale",
            "difference_rank_by_absolute_value",
            "cluster_label",
        ]
    )


def save_top_local_values(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    top_n: int,
) -> None:
    if top_n <= 0:
        return

    selected_keys: set[str] = set()
    for model_name in summary["model"].unique():
        model_summary = summary[summary["model"] == model_name]
        selected_keys.update(
            model_summary.nsmallest(
                top_n,
                "raw_importance_rank_within_model",
            )["cluster_key"].astype(str)
        )
        selected_keys.update(
            model_summary.nsmallest(
                top_n,
                "normalized_importance_rank_within_model",
            )["cluster_key"].astype(str)
        )
    if not differences.empty:
        for scale in differences["importance_scale"].unique():
            selected_keys.update(
                differences[
                    differences["importance_scale"] == scale
                ].nsmallest(
                    top_n,
                    "difference_rank_by_absolute_value",
                )["cluster_key"].astype(str)
            )

    rows: list[pd.DataFrame] = []
    for npz_path in sorted(
        (output_dir / "local_group_shap").glob("*.npz")
    ):
        payload = np.load(npz_path, allow_pickle=False)
        cluster_keys = payload["cluster_keys"].astype(str)
        positions = [
            index
            for index, key in enumerate(cluster_keys)
            if key in selected_keys
        ]
        if not positions:
            continue

        model_name = str(payload["model_name"].item())
        outer_fold = int(payload["outer_fold"].item())
        sample_ids = payload["sample_id"].astype(str)
        grouped = payload["group_shap"][:, positions]
        selected_cluster_keys = cluster_keys[positions]

        local = pd.DataFrame(
            {
                "model": np.repeat(model_name, grouped.size),
                "outer_fold": np.repeat(outer_fold, grouped.size),
                "sample_id": np.repeat(sample_ids, len(positions)),
                "cluster_key": np.tile(
                    selected_cluster_keys,
                    len(sample_ids),
                ),
                "grouped_shap_value": grouped.reshape(-1),
            }
        )
        rows.append(local)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            output_dir / "top_grouped_shap_oof_values.csv.gz",
            index=False,
            compression="gzip",
        )


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive.")
    if args.save_local_top_n < 0:
        raise ValueError("--save-local-top-n cannot be negative.")

    model_directories = dict(zip(args.models, [path.resolve() for path in args.model_dirs]))
    cluster_results_dir = args.cluster_results_dir.resolve()

    data = load_data(args)
    for name, directory in model_directories.items():
        for fold in np.unique(data.outer_fold):
            model_paths(directory, name, int(fold))
    membership = load_cluster_membership(cluster_results_dir)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    group_metadata, feature_to_group = build_global_group_metadata(
        membership,
        data.feature_names,
    )
    membership.to_csv(
        output_dir / "cluster_membership_used.csv",
        index=False,
    )
    group_metadata.to_csv(
        output_dir / "group_metadata.csv",
        index=False,
    )

    missing_from_dataset = sorted(
        set(membership["feature"]).difference(data.X.columns)
    )
    if missing_from_dataset:
        warnings.warn(
            f"{len(missing_from_dataset)} clustered features are absent from "
            f"the input dataset. Examples: {missing_from_dataset[:10]}"
        )

    folds = sorted(np.unique(data.outer_fold).tolist())
    fold_rows: list[pd.DataFrame] = []
    mapping_rows: list[pd.DataFrame] = []
    additivity_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for outer_fold in folds:
            print(f"{model_name}: TreeSHAP on held-out outer fold {outer_fold}")
            test_indices = np.flatnonzero(data.outer_fold == outer_fold)
            X_test = data.X.iloc[test_indices]

            saved_model = SavedShapModel(
                model_name=model_name,
                full_results_dir=model_directories[model_name],
                outer_fold=int(outer_fold),
                fallback_feature_names=data.feature_names,
            )
            prediction, feature_shap, base_value = (
                saved_model.predict_and_shap(X_test)
            )
            grouped, mapping, group_table = group_shap_values(
                feature_shap,
                saved_model.output_feature_names,
                feature_to_group,
                group_metadata,
            )

            reconstructed_feature = base_value + feature_shap.sum(axis=1)
            reconstructed_group = base_value + grouped.sum(axis=1)
            feature_error = np.abs(prediction - reconstructed_feature)
            group_error = np.abs(prediction - reconstructed_group)
            additivity_rows.append(
                {
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "n_test_compounds": len(test_indices),
                    "n_features_used": feature_shap.shape[1],
                    "n_groups_used": grouped.shape[1],
                    "max_abs_feature_shap_additivity_error": float(
                        np.max(feature_error)
                    ),
                    "mean_abs_feature_shap_additivity_error": float(
                        np.mean(feature_error)
                    ),
                    "max_abs_grouped_shap_additivity_error": float(
                        np.max(group_error)
                    ),
                    "mean_abs_grouped_shap_additivity_error": float(
                        np.mean(group_error)
                    ),
                }
            )

            group_table.insert(0, "model", model_name)
            group_table.insert(1, "outer_fold", int(outer_fold))
            group_table["n_test_compounds"] = len(test_indices)
            fold_rows.append(group_table)

            mapping.insert(0, "model", model_name)
            mapping.insert(1, "outer_fold", int(outer_fold))
            mapping_rows.append(mapping)

            if args.save_local_npz:
                np.savez_compressed(
                    output_dir
                    / "local_group_shap"
                    / f"{model_name}_outer_{outer_fold}.npz",
                    model_name=np.asarray(model_name),
                    outer_fold=np.asarray(int(outer_fold)),
                    row_index=test_indices.astype(np.int64),
                    sample_id=np.asarray(data.sample_id[test_indices].astype(str), dtype=str),
                    cluster_keys=np.asarray(group_table["cluster_key"].astype(str).tolist(), dtype=str),
                    group_shap=grouped.astype(np.float32),
                    prediction=prediction.astype(np.float32),
                    base_value=base_value.astype(np.float32),
                )

    fold_table = pd.concat(fold_rows, ignore_index=True)
    mapping_table = pd.concat(mapping_rows, ignore_index=True)
    additivity = pd.DataFrame(additivity_rows)

    fold_table.to_csv(
        output_dir / "grouped_shap_fold_importance.csv",
        index=False,
    )
    mapping_table.to_csv(
        output_dir / "model_fold_feature_group_mapping.csv",
        index=False,
    )
    additivity.to_csv(
        output_dir / "shap_additivity_checks.csv",
        index=False,
    )

    summary = summarize_fold_importance(fold_table)
    summary.to_csv(
        output_dir / "grouped_shap_summary.csv",
        index=False,
    )

    differences = pd.DataFrame()
    if MODEL_MSE in args.models and MODEL_MTL in args.models:
        differences = calculate_model_differences(
            fold_table,
            data,
            args.multiple_testing,
        )
        differences.to_csv(
            output_dir / "grouped_shap_mtl_minus_mse.csv",
            index=False,
        )

    for model_name in args.models:
        model_summary = summary[
            summary["model"] == model_name
        ].copy()
        model_summary.sort_values(
            "fold_mean_mean_abs_group_shap",
            ascending=False,
        ).to_csv(
            output_dir
            / "rankings"
            / f"{model_name}_raw_grouped_shap_ranking.csv",
            index=False,
        )
        model_summary.sort_values(
            "fold_mean_normalized_group_shap",
            ascending=False,
        ).to_csv(
            output_dir
            / "rankings"
            / f"{model_name}_normalized_grouped_shap_ranking.csv",
            index=False,
        )

        plot_model_importance(
            summary,
            model_name=model_name,
            scale="raw",
            output_path=(
                output_dir
                / "plots"
                / f"{model_name}_raw_grouped_shap_importance.png"
            ),
            top_n=args.top_n,
        )
        plot_model_importance(
            summary,
            model_name=model_name,
            scale="normalized",
            output_path=(
                output_dir
                / "plots"
                / f"{model_name}_normalized_grouped_shap_importance.png"
            ),
            top_n=args.top_n,
        )

    if not differences.empty:
        adjustment_label = {
            "bh": "BH q",
            "holm": "Holm p",
            "none": "p",
        }[args.multiple_testing]
        for scale in ("raw", "normalized"):
            scale_table = differences[
                differences["importance_scale"] == scale
            ].sort_values(
                "absolute_mtl_minus_mse",
                ascending=False,
            )
            scale_table.to_csv(
                output_dir
                / "rankings"
                / f"mtl_minus_mse_{scale}_grouped_shap_ranking.csv",
                index=False,
            )
            plot_model_difference(
                differences,
                scale=scale,
                adjustment_label=adjustment_label,
                output_path=(
                    output_dir
                    / "plots"
                    / f"mtl_minus_mse_{scale}_grouped_shap_difference.png"
                ),
                top_n=args.top_n,
            )

    if args.save_local_npz and args.save_local_top_n > 0:
        save_top_local_values(
            output_dir=output_dir,
            summary=summary,
            differences=differences,
            top_n=args.save_local_top_n,
        )

    manifest = {
        "input": str(args.input.resolve()),
        "model_directories": {name: str(path) for name, path in model_directories.items()},
        "prediction_scale": "raw_score (same additive scale as native TreeSHAP)",
        "cluster_results_dir": str(cluster_results_dir),
        "output_dir": str(output_dir),
        "models": args.models,
        "top_n": args.top_n,
        "multiple_testing": args.multiple_testing,
        "save_local_npz": args.save_local_npz,
        "save_local_top_n": args.save_local_top_n,
        "grouped_shap_definition": (
            "For each held-out compound, signed native LightGBM TreeSHAP "
            "values were summed across all model-used members of each target-"
            "independent descriptor cluster. Global group importance was the "
            "mean absolute value of this summed cluster contribution."
        ),
        "normalization_definition": (
            "Within each model and outer fold, each group's mean absolute "
            "grouped SHAP value was divided by the sum across all groups."
        ),
        "important_caveat": (
            "This is cluster-aggregated TreeSHAP. It is not guaranteed to be "
            "identical to formal group Shapley or Owen values recomputed with "
            "clusters as the original players."
        ),
        "individual_model_inference": (
            "No p-values are calculated or shown for individual-model group "
            "importance rankings."
        ),
        "between_model_inference": (
            "Raw and normalized MTL-minus-MSE group-importance differences "
            "were tested with two-sided Nadeau-Bengio corrected paired t-tests "
            "across matched outer folds. The test/train ratio used the complete "
            "outer-fold partition because grouped SHAP was evaluated over all "
            "held-out compounds."
        ),
        "python": sys.version,
        "lightgbm": lgb.__version__,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\nCompleted successfully.")
    print("Summary:", output_dir / "grouped_shap_summary.csv")
    if not differences.empty:
        print(
            "MTL - MSE:",
            output_dir / "grouped_shap_mtl_minus_mse.csv",
        )
    print("Plots:", output_dir / "plots")


if __name__ == "__main__":
    main()
