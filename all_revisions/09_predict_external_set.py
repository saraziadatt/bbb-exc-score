#!/usr/bin/env python3
"""
python 09_predict_external_set.py \
    --external-csv ../datasets/not_druglike_b3db_labelled.csv \
    --model-dir ./all_models/fitted_models \
    --models custom_mtl_original,lightgbm_mse_regression,dummy_mean_regression,dummy_prior_classification,logbb_pred_shaker_retrained \
    --output-dir external_test_set

b3db_non_drug_like_labelled.csv    

Outputs
-------
OUTPUT_DIR/
    <model>.csv
        Ensemble prediction averaged over the saved outer-fold models.

    <model>_fold_predictions.csv
        One prediction per external compound and saved fold model.

    all_selected_models.csv
        Concatenated ensemble predictions for all requested models.

    external_metrics.csv
        Classification metrics when the external CSV contains `bbb_class`.

    prediction_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MARGIN = -1.0

MODEL_TASKS = {
    "dummy_mean_regression": "regression",
    "lightgbm_mse_regression": "regression",
    "custom_mtl_original": "regression",
    "logbb_pred_shaker_retrained": "regression",
    "dummy_prior_classification": "classification",
}

ALIASES = {
    "all_current": list(MODEL_TASKS),
    "all": list(MODEL_TASKS),
    "dummy_regression": ["dummy_mean_regression"],
    "lightgbm_mse": ["lightgbm_mse_regression"],
    "custom_mtl": ["custom_mtl_original"],
    "original_custom_mtl": ["custom_mtl_original"],
    "shaker_retrained": ["logbb_pred_shaker_retrained"],
    "logbb_pred": ["logbb_pred_shaker_retrained"],
    "logbb_pred_matched": ["logbb_pred_shaker_retrained"],
    "matched_shaker": ["logbb_pred_shaker_retrained"],
    "dummy_classifier": ["dummy_prior_classification"],
}


# ---------------------------------------------------------------------------
# Compatibility classes for saved joblib objects
# ---------------------------------------------------------------------------

class TrainingColumnFilter(BaseEstimator, TransformerMixin):
    """Compatibility implementation used by the current Shaker pipeline."""

    def __init__(self, max_missing_fraction: float = 0.20):
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X: pd.DataFrame, y: Any = None):
        del y
        frame = pd.DataFrame(X).copy()
        missing_fraction = frame.isna().mean()
        candidate = missing_fraction[
            missing_fraction <= self.max_missing_fraction
        ].index.tolist()

        kept = []
        for column in candidate:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.nunique(dropna=True) > 1:
                kept.append(column)

        if not kept:
            raise ValueError("TrainingColumnFilter removed every descriptor.")

        self.input_columns_ = frame.columns.tolist()
        self.kept_columns_ = kept
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing = [c for c in self.kept_columns_ if c not in frame.columns]
        if missing:
            raise KeyError(
                "Input is missing descriptors selected during training: "
                f"{missing[:20]}"
            )
        return frame.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.kept_columns_, dtype=object)


class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Compatibility implementation used by the current Shaker pipeline."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        del y
        frame = pd.DataFrame(X).copy()
        self.columns_ = frame.columns.tolist()
        self.medians_ = frame.median(axis=0, numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        return frame.loc[:, self.columns_].fillna(self.medians_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.columns_, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Compatibility implementation used by the current Shaker pipeline."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None):
        del y
        frame = pd.DataFrame(X).copy()
        correlation = frame.corr().abs()
        upper = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
        )
        drop = [
            column
            for column in upper.columns
            if np.any(upper[column] > self.threshold)
        ]
        self.input_columns_ = frame.columns.tolist()
        self.dropped_columns_ = drop
        self.kept_columns_ = [
            column for column in self.input_columns_ if column not in drop
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X).loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.kept_columns_, dtype=object)


# Compatibility with recently generated optional custom preprocessors.
class CompleteMeasuredTrainingFilter:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        return frame.loc[:, self.kept_columns_].copy()


class TargetAwarePearsonFilter:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        return frame.loc[:, self.kept_columns_].copy()


@dataclass
class Model4ShakerPreprocessor:
    complete_filter: Any
    pearson_filter: Any
    imputer: Any

    @property
    def selected_columns(self) -> list[str]:
        return list(self.pearson_filter.kept_columns_)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        complete = self.complete_filter.transform(X)
        selected = self.pearson_filter.transform(complete)
        return np.asarray(self.imputer.transform(selected), dtype=float)


class CompleteTrainingDescriptorFilter:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        return frame.loc[:, self.kept_columns_].copy()


@dataclass
class ExactShakerFeatureSelector:
    complete_filter: Any
    pcc_filter: Any

    @property
    def selected_columns(self) -> list[str]:
        return list(self.pcc_filter.kept_columns_)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        complete = self.complete_filter.transform(X)
        selected = self.pcc_filter.transform(complete)
        return selected.to_numpy(dtype=float)


@dataclass
class ShakerFeaturePreprocessor:
    selected_columns: list[str]
    removed_missing_columns: list[str]
    removed_constant_columns: list[str]
    removed_correlated_columns: list[str]
    feature_target_abs_correlation: dict[str, float]
    pcc_threshold: float
    scaler: Any

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X).loc[:, self.selected_columns]
        return np.asarray(self.scaler.transform(frame), dtype=float)


ShakerRegressionPreprocessor = ShakerFeaturePreprocessor


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

@dataclass
class FoldPrediction:
    fold: int
    predicted_logbb: np.ndarray | None
    score_np: np.ndarray
    pred_np: np.ndarray
    probability_np: np.ndarray | None
    decision_threshold: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--external-csv", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing the currently saved outer-fold models.",
    )
    parser.add_argument(
        "--models",
        default="all_current",
        help=(
            "Comma-separated model names or aliases. Available names: "
            + ", ".join(MODEL_TASKS)
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Request one model. May be supplied repeatedly. Values are added "
            "to --models, except when --models retains its default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inxight_current_model_predictions"),
    )
    parser.add_argument("--id-col", default="external_compound_id")
    parser.add_argument("--label-col", default="bbb_class")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=MARGIN,
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first requested model that cannot be predicted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing model output CSVs.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print accepted current model names and aliases, then exit.",
    )
    return parser.parse_args()


def expand_names(specification: str, repeated: list[str]) -> list[str]:
    tokens: list[str] = []

    # A repeated --model request should make individual-model use convenient
    # without requiring --models none.
    if repeated and specification == "all_current":
        tokens.extend(repeated)
    else:
        tokens.extend(
            token.strip()
            for token in specification.split(",")
            if token.strip()
        )
        tokens.extend(repeated)

    output: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        expanded = ALIASES.get(token, [token])
        for name in expanded:
            if name not in MODEL_TASKS:
                raise ValueError(
                    f"Unknown model {name!r}. Choices: {sorted(MODEL_TASKS)}"
                )
            if name not in seen:
                seen.add(name)
                output.append(name)

    if not output:
        raise ValueError("No models were requested.")
    return output


def prepare_external(
    path: Path,
    id_col: str,
    label_col: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    if id_col not in frame.columns:
        frame[id_col] = [
            f"external_row_{index:06d}"
            for index in range(len(frame))
        ]

    frame[id_col] = frame[id_col].astype(str).str.strip()
    if frame[id_col].eq("").any() or frame[id_col].duplicated().any():
        raise ValueError(
            f"{id_col} must contain unique, non-empty identifiers."
        )

    if label_col in frame.columns:
        labels = pd.to_numeric(frame[label_col], errors="raise")
        if not set(labels.astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{label_col} must use 0 and 1.")
        frame[label_col] = labels.astype(int)

    return frame


def normalize_names(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() for value in values]


def feature_names_from_estimator(estimator: Any) -> list[str] | None:
    candidates = [
        estimator,
        getattr(estimator, "best_estimator_", None),
    ]

    if hasattr(estimator, "named_steps"):
        candidates.extend(estimator.named_steps.values())

    for candidate in candidates:
        if candidate is None:
            continue

        values = getattr(candidate, "feature_names_in_", None)
        if values is not None:
            return normalize_names(values.tolist())

        values = getattr(candidate, "input_columns_", None)
        if values is not None:
            return normalize_names(values)

    return None


def numeric_frame(
    external: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    names = normalize_names(feature_names)
    if len(names) != len(set(names)):
        raise ValueError(f"{context}: duplicate saved feature names.")

    missing = [name for name in names if name not in external.columns]
    if missing:
        raise KeyError(
            f"{context}: {len(missing)} saved features are absent from the "
            f"external CSV. First missing features: {missing[:25]}"
        )

    matrix = external.loc[:, names].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return matrix.replace([np.inf, -np.inf], np.nan)


def unwrap_loaded(loaded: Any) -> tuple[Any, dict[str, Any]]:
    if hasattr(loaded, "predict"):
        return loaded, {}

    if isinstance(loaded, dict):
        for key in [
            "model",
            "estimator",
            "pipeline",
            "regressor",
            "classifier",
            "fitted_model",
            "best_estimator",
        ]:
            candidate = loaded.get(key)
            if hasattr(candidate, "predict"):
                return candidate, dict(loaded)

    raise TypeError(
        "Saved joblib object is neither a predictor nor a supported model "
        "dictionary."
    )


def fold_number(path: Path) -> int:
    match = re.search(r"_outer_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer outer fold from {path.name}")
    return int(match.group(1))


def class_one_score(estimator: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
        if probabilities.ndim != 2 or 1 not in classes:
            raise ValueError(
                "Classifier predict_proba output has no class-1 column."
            )
        index = int(np.flatnonzero(classes == 1)[0])
        return probabilities[:, index], 0.5

    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        if score.ndim == 2:
            classes = np.asarray(getattr(estimator, "classes_", []))
            if 1 not in classes:
                raise ValueError(
                    "Classifier decision_function output has no class 1."
                )
            score = score[:, int(np.flatnonzero(classes == 1)[0])]
        return score.ravel(), 0.0

    prediction = np.asarray(estimator.predict(X), dtype=float).ravel()
    return prediction, 0.5


# ---------------------------------------------------------------------------
# Current artifact loaders
# ---------------------------------------------------------------------------

def predict_joblib_model(
    *,
    model_name: str,
    model_path: Path,
    external: pd.DataFrame,
    regression_threshold: float,
) -> FoldPrediction:
    loaded = joblib.load(model_path)
    estimator, metadata = unwrap_loaded(loaded)

    feature_names = feature_names_from_estimator(estimator)
    if feature_names is None:
        for key in ["feature_names", "feature_columns", "selected_features"]:
            values = metadata.get(key)
            if values is not None:
                feature_names = normalize_names(values)
                break

    if feature_names is None:
        raise RuntimeError(
            f"{model_path}: could not determine the training feature order."
        )

    X = numeric_frame(
        external,
        feature_names,
        context=f"{model_name}, fold {fold_number(model_path)}",
    )

    task = MODEL_TASKS[model_name]
    if task == "regression":
        predicted_logbb = np.asarray(estimator.predict(X), dtype=float).ravel()
        score_np = -predicted_logbb
        pred_np = (predicted_logbb < regression_threshold).astype(int)
        probability_np = None
        threshold = regression_threshold
    else:
        score_np, score_threshold = class_one_score(estimator, X)
        pred_np = (score_np >= score_threshold).astype(int)
        predicted_logbb = None
        probability_np = score_np if score_threshold == 0.5 else None
        threshold = score_threshold

    return FoldPrediction(
        fold=fold_number(model_path),
        predicted_logbb=predicted_logbb,
        score_np=np.asarray(score_np, dtype=float),
        pred_np=np.asarray(pred_np, dtype=int),
        probability_np=(
            None
            if probability_np is None
            else np.asarray(probability_np, dtype=float)
        ),
        decision_threshold=float(threshold),
        source=str(model_path),
    )


def locate_custom_preprocessor(
    model_dir: Path,
    fold: int,
) -> Path:
    preferred = (
        model_dir
        / f"custom_mtl_original_outer_{fold}_preprocessor.joblib"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        model_dir.glob(
            f"custom_mtl_original_outer_{fold}_*preprocessor.joblib"
        )
    )
    if not candidates:
        candidates = sorted(
            model_dir.glob(
                f"custom_mtl_original_outer_{fold}_*selector.joblib"
            )
        )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Fold {fold}: expected one custom-MTL preprocessor, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def transform_custom_external(
    saved: Any,
    external: pd.DataFrame,
    *,
    context: str,
) -> tuple[np.ndarray, list[str] | None]:
    if not isinstance(saved, dict):
        if not hasattr(saved, "transform"):
            raise TypeError(f"{context}: unsupported preprocessor object.")
        transformed = saved.transform(external)
        return np.asarray(transformed, dtype=float), None

    feature_names = saved.get("feature_names")
    if feature_names is not None:
        feature_names = normalize_names(feature_names)

    # Original model-#4 artifact: saved median imputer + feature order.
    if "imputer" in saved:
        if feature_names is None:
            raise KeyError(f"{context}: imputer artifact lacks feature_names.")
        X = numeric_frame(external, feature_names, context=context)
        return (
            np.asarray(saved["imputer"].transform(X), dtype=float),
            feature_names,
        )

    # Optional newer preprocessors saved under these keys.
    for key in ["preprocessor", "feature_selector", "selector"]:
        processor = saved.get(key)
        if processor is None:
            continue

        try:
            transformed = processor.transform(external)
        except Exception:
            if feature_names is None:
                raise
            X = numeric_frame(external, feature_names, context=context)
            transformed = processor.transform(X)

        return np.asarray(transformed, dtype=float), feature_names

    raise KeyError(
        f"{context}: supported keys are imputer, preprocessor, "
        "feature_selector, or selector."
    )


def predict_custom_mtl(
    *,
    model_path: Path,
    preprocessor_path: Path,
    external: pd.DataFrame,
    regression_threshold: float,
) -> FoldPrediction:
    fold = fold_number(model_path)
    booster = lgb.Booster(model_file=str(model_path))
    preprocessor = joblib.load(preprocessor_path)

    X, _ = transform_custom_external(
        preprocessor,
        external,
        context=f"custom_mtl_original, fold {fold}",
    )
    predicted_logbb = np.asarray(booster.predict(X), dtype=float).ravel()

    return FoldPrediction(
        fold=fold,
        predicted_logbb=predicted_logbb,
        score_np=-predicted_logbb,
        pred_np=(predicted_logbb < regression_threshold).astype(int),
        probability_np=None,
        decision_threshold=float(regression_threshold),
        source=f"{model_path}; {preprocessor_path}",
    )



def discover_joblib_artifacts(
    *,
    model_name: str,
    model_dir: Path,
) -> tuple[list[Path], str]:
    """
    Discover one compatible artifact family without combining distinct runs.

    For the canonical `logbb_pred_shaker_retrained` request, support both:
    - current five-model artifacts:
        logbb_pred_shaker_retrained_outer_<fold>.joblib
    - standalone matched-retraining bundles:
        logbb_pred_matched_outer_<fold>.joblib

    If both families exist, prefer the canonical current-five-model family.
    """
    pattern_families: dict[str, list[str]] = {
        "logbb_pred_shaker_retrained": [
            "logbb_pred_shaker_retrained_outer_*.joblib",
            "logbb_pred_matched_outer_*.joblib",
        ],
    }

    patterns = pattern_families.get(
        model_name,
        [f"{model_name}_outer_*.joblib"],
    )

    for pattern in patterns:
        paths = sorted(
            model_dir.glob(pattern),
            key=fold_number,
        )
        if paths:
            folds = [fold_number(path) for path in paths]
            if len(folds) != len(set(folds)):
                raise RuntimeError(
                    f"{model_name}: duplicate outer-fold artifacts found "
                    f"for pattern {pattern!r}: {folds}"
                )
            return paths, pattern

    return [], patterns[0]


def load_fold_predictions(
    *,
    model_name: str,
    model_dir: Path,
    external: pd.DataFrame,
    regression_threshold: float,
) -> list[FoldPrediction]:
    if model_name == "custom_mtl_original":
        text_models = sorted(
            model_dir.glob("custom_mtl_original_outer_*.txt"),
            key=fold_number,
        )

        # Allow a joblib custom estimator as a fallback.
        if not text_models:
            joblib_models = sorted(
                model_dir.glob("custom_mtl_original_outer_*.joblib"),
                key=fold_number,
            )
            joblib_models = [
                path
                for path in joblib_models
                if "preprocessor" not in path.name
                and "selector" not in path.name
            ]
            return [
                predict_joblib_model(
                    model_name=model_name,
                    model_path=path,
                    external=external,
                    regression_threshold=regression_threshold,
                )
                for path in joblib_models
            ]

        output = []
        for model_path in text_models:
            fold = fold_number(model_path)
            preprocessor_path = locate_custom_preprocessor(model_dir, fold)
            output.append(
                predict_custom_mtl(
                    model_path=model_path,
                    preprocessor_path=preprocessor_path,
                    external=external,
                    regression_threshold=regression_threshold,
                )
            )
        return output

    paths, matched_pattern = discover_joblib_artifacts(
        model_name=model_name,
        model_dir=model_dir,
    )
    if paths:
        print(
            f"Found {len(paths)} artifact(s) using "
            f"{matched_pattern}"
        )

    return [
        predict_joblib_model(
            model_name=model_name,
            model_path=path,
            external=external,
            regression_threshold=regression_threshold,
        )
        for path in paths
    ]


# ---------------------------------------------------------------------------
# Ensembling, output, and optional external metrics
# ---------------------------------------------------------------------------

def ensemble_predictions(
    *,
    model_name: str,
    folds: list[FoldPrediction],
    regression_threshold: float,
    classifier_threshold: float,
) -> dict[str, np.ndarray | float | int | str | None]:
    if not folds:
        raise ValueError(f"No fold models were found for {model_name}.")

    task = MODEL_TASKS[model_name]
    score_matrix = np.vstack([fold.score_np for fold in folds])
    mean_score = score_matrix.mean(axis=0)

    if task == "regression":
        logbb_matrix = np.vstack(
            [fold.predicted_logbb for fold in folds]
        )
        mean_logbb = logbb_matrix.mean(axis=0)
        pred_np = (mean_logbb < regression_threshold).astype(int)
        probability_np = None
        threshold = regression_threshold
    else:
        mean_logbb = None

        # For probability-based classifiers this is 0.5. For decision scores,
        # preserve 0.0. All folds from one model should agree.
        fold_thresholds = {fold.decision_threshold for fold in folds}
        threshold = (
            next(iter(fold_thresholds))
            if len(fold_thresholds) == 1
            else classifier_threshold
        )
        pred_np = (mean_score >= threshold).astype(int)
        probability_np = (
            mean_score
            if all(fold.probability_np is not None for fold in folds)
            else None
        )

    return {
        "predicted_logbb": mean_logbb,
        "score_np": mean_score,
        "pred_np": pred_np,
        "probability_np": probability_np,
        "threshold": float(threshold),
        "n_fold_models": len(folds),
    }


def classification_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)

    np_mask = y_true == 1
    p_mask = y_true == 0

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "np_recall": float(
            recall_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "np_precision": float(
            precision_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "np_f1": float(
            f1_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "penetrant_specificity": float(
            np.mean(pred[p_mask] == 0) if np.any(p_mask) else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, score))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "pr_auc_nonpenetrant": (
            float(average_precision_score(y_true, score))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def write_model_outputs(
    *,
    model_name: str,
    external: pd.DataFrame,
    id_col: str,
    label_col: str,
    folds: list[FoldPrediction],
    ensemble: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    ensemble_path = output_dir / f"{model_name}.csv"
    fold_path = output_dir / f"{model_name}_fold_predictions.csv"

    for path in [ensemble_path, fold_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite or another output dir."
            )

    output = pd.DataFrame(
        {
            id_col: external[id_col].to_numpy(),
            "model": model_name,
            "score_np": ensemble["score_np"],
            "pred_np": ensemble["pred_np"],
            "predicted_logbb": (
                ensemble["predicted_logbb"]
                if ensemble["predicted_logbb"] is not None
                else np.nan
            ),
            "probability_np": (
                ensemble["probability_np"]
                if ensemble["probability_np"] is not None
                else np.nan
            ),
            "threshold": ensemble["threshold"],
            "n_fold_models": ensemble["n_fold_models"],
        }
    )
    if label_col in external.columns:
        output[label_col] = external[label_col].to_numpy()

    fold_parts = []
    for fold in folds:
        part = pd.DataFrame(
            {
                id_col: external[id_col].to_numpy(),
                "model": model_name,
                "outer_fold_model": fold.fold,
                "score_np": fold.score_np,
                "pred_np": fold.pred_np,
                "predicted_logbb": (
                    fold.predicted_logbb
                    if fold.predicted_logbb is not None
                    else np.nan
                ),
                "probability_np": (
                    fold.probability_np
                    if fold.probability_np is not None
                    else np.nan
                ),
                "threshold": fold.decision_threshold,
                "source": fold.source,
            }
        )
        if label_col in external.columns:
            part[label_col] = external[label_col].to_numpy()
        fold_parts.append(part)

    fold_output = pd.concat(fold_parts, ignore_index=True)
    output.to_csv(ensemble_path, index=False)
    fold_output.to_csv(fold_path, index=False)

    metrics = None
    if label_col in external.columns:
        metrics = {
            "model": model_name,
            **classification_metrics(
                external[label_col].to_numpy(dtype=int),
                output["pred_np"].to_numpy(dtype=int),
                output["score_np"].to_numpy(dtype=float),
            ),
        }

    return output, metrics


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Current model names:")
        for model, task in MODEL_TASKS.items():
            print(f"  {model:<36} {task}")
        print("\nAliases:")
        for alias, values in ALIASES.items():
            print(f"  {alias:<24} {','.join(values)}")
        return

    model_names = expand_names(args.models, args.model)
    external = prepare_external(
        args.external_csv,
        args.id_col,
        args.label_col,
    )

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    manifest_models: dict[str, Any] = {}
    failures: dict[str, str] = {}

    print(f"External compounds: {len(external)}")
    print(f"Model directory: {model_dir}")
    print("Requested models:", ", ".join(model_names))

    for model_name in model_names:
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        try:
            folds = load_fold_predictions(
                model_name=model_name,
                model_dir=model_dir,
                external=external,
                regression_threshold=args.regression_threshold,
            )
            if not folds:
                accepted = (
                    "logbb_pred_shaker_retrained_outer_*.joblib or "
                    "logbb_pred_matched_outer_*.joblib"
                    if model_name == "logbb_pred_shaker_retrained"
                    else f"{model_name}_outer_*.joblib"
                )
                present = sorted(
                    path.name for path in model_dir.iterdir()
                    if path.is_file()
                )
                raise FileNotFoundError(
                    f"No saved fold artifacts found for {model_name} in "
                    f"{model_dir}. Accepted pattern(s): {accepted}. "
                    f"Files present: {present[:30]}"
                )

            ensemble = ensemble_predictions(
                model_name=model_name,
                folds=folds,
                regression_threshold=args.regression_threshold,
                classifier_threshold=args.classifier_threshold,
            )
            output, metrics = write_model_outputs(
                model_name=model_name,
                external=external,
                id_col=args.id_col,
                label_col=args.label_col,
                folds=folds,
                ensemble=ensemble,
                output_dir=output_dir,
                overwrite=args.overwrite,
            )
            combined.append(output)
            if metrics is not None:
                metric_rows.append(metrics)

            manifest_models[model_name] = {
                "task": MODEL_TASKS[model_name],
                "n_fold_models": len(folds),
                "folds": [fold.fold for fold in folds],
                "sources": [fold.source for fold in folds],
                "ensemble_threshold": ensemble["threshold"],
            }
            print(
                f"Wrote {model_name}.csv using "
                f"{len(folds)} saved fold models."
            )

        except Exception as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {failures[model_name]}", file=sys.stderr)
            if args.strict:
                raise

    if combined:
        combined_output = pd.concat(combined, ignore_index=True)
        combined_output.to_csv(
            output_dir / "all_selected_models.csv",
            index=False,
        )

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "external_metrics.csv",
            index=False,
        )

    manifest = {
        "external_csv": str(args.external_csv.resolve()),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "n_external_compounds": int(len(external)),
        "requested_models": model_names,
        "successful_models": list(manifest_models),
        "failed_models": failures,
        "models": manifest_models,
        "regression_threshold": args.regression_threshold,
        "classifier_threshold": args.classifier_threshold,
        "label_column_present": args.label_col in external.columns,
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    if failures:
        print("\nFailed models:")
        for model, error in failures.items():
            print(f"  {model}: {error}")

    if not combined:
        raise RuntimeError(
            "No requested model produced predictions. See failure messages."
        )

    print("\nCompleted successfully.")
    print("Combined predictions:", output_dir / "all_selected_models.csv")
    if metric_rows:
        print("External metrics:", output_dir / "external_metrics.csv")


if __name__ == "__main__":
    main()
