#!/usr/bin/env python3
"""
Predict external datasets using all trained models, including the threshold tuned lightgbm. 
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

def argv_requests_pretrained_models() -> bool:
    """Detect explicit pretrained runs before native scientific imports."""
    # Descriptor subprocesses inherit the original flags, including ImageMol
    # paths. Those flags must not cause PyTorch to be imported in a worker.
    if "--_descriptor-worker" in sys.argv[1:]:
        return False
    command_line = " ".join(sys.argv[1:]).lower()
    return any(
        marker in command_line
        for marker in ("imagemol", "unimol", "gnn", ".pt", "all_discovered")
    )


# ImageMol is stable when PyTorch is imported before the scientific stack, but
# standard LightGBM-only runs must not load PyTorch unnecessarily. Select the
# native runtime from the explicit command line before importing scikit-learn.
if argv_requests_pretrained_models():
    try:
        import torch
    except ImportError:  # Reported with model-specific context when requested.
        torch = None
else:
    torch = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# Native LightGBM is imported lazily only when a .txt booster is requested.
# Pretrained-only prediction runs therefore do not load LightGBM or its native
# OpenMP runtime at all.
lgb = None


MARGIN = -1.0

MODEL_TASKS = {
    "dummy_mean_regression": "regression",
    "lightgbm_mse_regression": "regression",
    "custom_mtl_bab": "regression",
    "logbb_pred_shaker_retrained": "regression",
    "dummy_prior_classification": "classification",
    "lightgbm_mse_regression_threshold": "regression",
}

# This baseline uses saved inner-fold thresholds; other built-ins stay native.
# Do not infer threshold tuning from filenames of unrelated dummy artifacts.
TUNED_THRESHOLD_MODELS: set[str] = {"lightgbm_mse_regression_threshold"}

# Logical model names are deliberately separated from physical artifact names.
# This preserves compatibility with the current outputs of script 07 without
# treating every artifact containing ``_threshold`` as threshold-tuned.
PHYSICAL_PREFIX_TO_LOGICAL = {
    "dummy_mean_regression_threshold": ["dummy_mean_regression"],
    "lightgbm_mse_regression_threshold": [
        "lightgbm_mse_regression",
        "lightgbm_mse_regression_threshold",
    ],
    "dummy_prior_classification_threshold": [
        "dummy_prior_classification"
    ],
}

ALIASES = {
    "all_current": list(MODEL_TASKS),
    "all": list(MODEL_TASKS),
    "dummy_regression": ["dummy_mean_regression"],
    "lightgbm_mse": ["lightgbm_mse_regression"],
    "custom_mtl": ["custom_mtl_bab"],
    "original_custom_mtl": ["custom_mtl_bab"],
    "shaker_retrained": ["logbb_pred_shaker_retrained"],
    "logbb_pred": ["logbb_pred_shaker_retrained"],
    "logbb_pred_matched": ["logbb_pred_shaker_retrained"],
    "matched_shaker": ["logbb_pred_shaker_retrained"],
    "dummy_classifier": ["dummy_prior_classification"],
}

DISCOVERY_ALIASES = {"all_discovered", "discovered", "auto"}

RESERVED_OUTPUT_COLUMNS = {
    "model",
    "model_task",
    "score_type",
    "score_np",
    "pred_np",
    "predicted_logbb",
    "probability_np",
    "threshold",
    "n_fold_models",
    "outer_fold_model",
    "source",
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
    score_type: str
    source: str


@dataclass
class PretrainedInferenceConfig:
    device_name: str
    batch_size: int
    unimol_model_size: str
    unimol_remove_hs: bool
    imagemol_checkpoint: Path | None
    imagemol_resume_key: str
    imagemol_workers: int
    embedding_cache: dict[str, np.ndarray]


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
            "Comma-separated descriptor-model names or aliases. Conventional "
            "joblib/pickle artifacts and frozen Uni-Mol/ImageMol .pt heads "
            "are auto-detected. "
            "Use all_discovered to evaluate every compatible artifact family. "
            "Built-in names: " + ", ".join(MODEL_TASKS)
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
        "--model-task",
        action="append",
        default=[],
        metavar="MODEL=TASK",
        help=(
            "Declare the task for a new model (regression or classification). "
            "May be repeated. Usually unnecessary for standard scikit-learn "
            "estimators because their task is detected automatically."
        ),
    )
    parser.add_argument(
        "--artifact-pattern",
        action="append",
        default=[],
        metavar="MODEL=GLOB",
        help=(
            "Override artifact discovery for a model, for example "
            "--artifact-pattern 'random_forest=random_forest_fold_*.pkl'. "
            "The glob is evaluated inside --model-dir and may be repeated."
        ),
    )
    parser.add_argument(
        "--pretrained-device",
        default="auto",
        help="Device for frozen Uni-Mol/ImageMol inference, such as cpu or cuda.",
    )
    parser.add_argument(
        "--pretrained-batch-size",
        type=int,
        default=64,
        help="Batch size for pretrained embedding and linear-head inference.",
    )
    parser.add_argument(
        "--unimol-model-size",
        default="84m",
        help="Uni-Mol v2 size; ignored for Uni-Mol v1 checkpoints.",
    )
    parser.add_argument(
        "--unimol-remove-hs",
        action="store_true",
        help="Match training runs that generated Uni-Mol embeddings with --remove-hs.",
    )
    parser.add_argument(
        "--imagemol-checkpoint",
        type=Path,
        default=None,
        help="Official ImageMol.pth.tar used to regenerate external embeddings.",
    )
    parser.add_argument("--imagemol-resume-key", default="state_dict")
    parser.add_argument("--imagemol-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inxight_current_model_predictions"),
    )
    parser.add_argument("--id-col", default="external_compound_id")
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="SMILES column copied to prediction outputs when present.",
    )
    parser.add_argument("--label-col", default="bbb_class")
    parser.add_argument(
        "--regression-label-col",
        default="logBB",
        help="Measured external regression target, when present.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=MARGIN,
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to class-1 probabilities.",
    )
    parser.add_argument(
        "--classifier-decision-threshold",
        type=float,
        default=0.0,
        help="Threshold applied to classifier decision_function scores.",
    )
    parser.add_argument(
        "--required-folds",
        default="0,1,2,3,4",
        help=(
            "Comma-separated required fold numbers. Use 'auto' to accept the "
            "first successful model's folds and require subsequent models to "
            "match them."
        ),
    )
    parser.add_argument(
        "--tuned-threshold-model",
        action="append",
        default=[],
        metavar="MODEL",
        help=(
            "Mark an additional regression model as using per-fold tuned "
            "threshold JSON metadata. May be repeated."
        ),
    )
    parser.add_argument(
        "--threshold-pattern",
        action="append",
        default=[],
        metavar="MODEL=GLOB",
        help=(
            "Override threshold JSON discovery. The glob may contain {fold}; "
            "for example MODEL='name_fold_{fold}.json'."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first requested model that cannot be predicted.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Permit a successful exit when some requested models fail. "
            "Without this option, any model failure makes the run fail."
        ),
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
    parser.add_argument(
        "--_descriptor-worker", nargs=2, metavar=("MODEL", "RESULT"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_key_value_options(
    values: list[str],
    *,
    option: str,
    allowed_values: set[str] | None = None,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects MODEL=VALUE, received {value!r}.")
        name, setting = (part.strip() for part in value.split("=", 1))
        if not name or not setting:
            raise ValueError(f"{option} expects non-empty MODEL=VALUE.")
        if allowed_values is not None and setting not in allowed_values:
            raise ValueError(
                f"{option} for {name!r} must be one of "
                f"{sorted(allowed_values)}, received {setting!r}."
            )
        if name in output and output[name] != setting:
            raise ValueError(f"Conflicting {option} values for {name!r}.")
        output[name] = setting
    return output


def parse_required_folds(value: str) -> list[int] | None:
    if value.strip().lower() == "auto":
        return None
    try:
        folds = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError("--required-folds must be comma-separated integers or 'auto'.") from exc
    if not folds or len(folds) != len(set(folds)) or any(fold < 0 for fold in folds):
        raise ValueError("--required-folds must contain unique non-negative integers.")
    return sorted(folds)


def validate_model_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
        raise ValueError(
            f"Unsafe model name {name!r}. Use letters, numbers, '.', '_', or '-' "
            "without path components."
        )


def discover_model_names(model_dir: Path) -> list[str]:
    """Find model prefixes from conventional outer-fold artifact names."""
    names: set[str] = set()
    ignored = ("preprocessor", "selector")

    for path in model_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {
            ".joblib", ".pkl", ".pickle", ".txt", ".pt"
        }:
            continue
        lowered = path.stem.lower()
        if any(token in lowered for token in ignored):
            continue
        match = re.match(r"(.+?)_(?:outer|fold)_(\d+)(?:_|$)", path.stem)
        if match:
            physical_name = match.group(1)
            if path.suffix.lower() == ".txt" and physical_name != "custom_mtl_bab":
                continue
            logical_names = PHYSICAL_PREFIX_TO_LOGICAL.get(
                physical_name, [physical_name]
            )
            names.update(logical_names)

    # Original frozen-encoder runs store checkpoints as
    # MODEL_NAME/outer_fold_N/model.pt rather than as flat files.
    for path in model_dir.rglob("model.pt"):
        fold_match = re.fullmatch(r"outer_fold_(\d+)", path.parent.name)
        if fold_match and path.parent.parent != model_dir:
            names.add(path.parent.parent.name)

    return sorted(names)


def expand_names(
    specification: str,
    repeated: list[str],
    discovered: list[str],
) -> list[str]:
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
        if token in DISCOVERY_ALIASES:
            expanded = discovered
        else:
            expanded = ALIASES.get(token, [token])
        for name in expanded:
            validate_model_name(name)
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
    regression_label_col: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    if frame.empty:
        raise ValueError("The external CSV contains no rows.")

    if id_col not in frame.columns:
        frame[id_col] = [
            f"external_row_{index:06d}"
            for index in range(len(frame))
        ]

    elif frame[id_col].isna().any():
        raise ValueError(f"{id_col} contains missing identifiers.")

    frame[id_col] = frame[id_col].astype(str).str.strip()
    if frame[id_col].eq("").any() or frame[id_col].duplicated().any():
        raise ValueError(
            f"{id_col} must contain unique, non-empty identifiers."
        )

    if label_col in frame.columns:
        labels = pd.to_numeric(frame[label_col], errors="raise")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{label_col} must use 0 and 1.")
        frame[label_col] = labels.astype(int)

    if regression_label_col in frame.columns:
        regression_values = pd.to_numeric(
            frame[regression_label_col], errors="raise"
        )
        if np.isinf(regression_values.to_numpy(dtype=float)).any():
            raise ValueError(f"{regression_label_col} contains infinite values.")
        frame[regression_label_col] = regression_values.astype(float)

    return frame


def normalize_names(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() for value in values]


def feature_names_from_estimator(estimator: Any) -> list[str] | None:
    candidates = [
        estimator,
        getattr(estimator, "best_estimator_", None),
    ]

    if hasattr(estimator, "named_steps"):
        # Only the first pipeline step sees raw external columns. Inspecting
        # later steps can mistake transformed feature names for input columns.
        first_step = next(iter(estimator.named_steps.values()), None)
        candidates.append(first_step)

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
    match = re.search(r"_(?:outer|fold)_(\d+)(?:_|\.|$)", path.name)
    if not match:
        raise ValueError(
            f"Cannot infer fold from {path.name}. Use a filename containing "
            "'_outer_<number>' or '_fold_<number>'."
        )
    return int(match.group(1))


def pretrained_fold_number(path: Path) -> int:
    """Read a fold from a flat artifact name or .../outer_fold_N/model.pt."""
    try:
        return fold_number(path)
    except ValueError:
        match = re.fullmatch(r"outer_fold_(\d+)", path.parent.name)
        if match and path.name == "model.pt":
            return int(match.group(1))
        raise


def discover_pretrained_artifacts(
    *,
    model_name: str,
    model_dir: Path,
    artifact_pattern: str | None = None,
) -> tuple[list[Path], str]:
    patterns = (
        [artifact_pattern]
        if artifact_pattern is not None and artifact_pattern.lower().endswith(".pt")
        else [
            f"{model_name}_outer_*.pt",
            f"{model_name}_fold_*.pt",
            f"{model_name}/outer_fold_*/model.pt",
        ]
    )
    for pattern in patterns:
        paths = sorted(model_dir.glob(pattern), key=pretrained_fold_number)
        if paths:
            folds = [pretrained_fold_number(path) for path in paths]
            if len(folds) != len(set(folds)):
                raise RuntimeError(
                    f"{model_name}: duplicate pretrained checkpoints for "
                    f"pattern {pattern!r}: {folds}"
                )
            return paths, pattern
    return [], patterns[0]


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    if torch is None:
        raise ImportError(
            "PyTorch is required to load frozen pretrained .pt checkpoints."
        )
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {path}.")
    return loaded


def resolve_torch_device(device_name: str):
    if torch is None:
        raise ImportError("PyTorch is required for pretrained model inference.")
    selected = (
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")
    return device


def unimol_external_embeddings(
    smiles: list[str],
    *,
    encoder_name: str,
    config: PretrainedInferenceConfig,
) -> np.ndarray:
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError(
            "Uni-Mol checkpoints require unimol-tools in this environment."
        ) from exc

    model_name = encoder_name.removesuffix("_frozen")
    representation_model = UniMolRepr(
        data_type="molecule",
        batch_size=config.batch_size,
        remove_hs=config.unimol_remove_hs,
        model_name=model_name,
        model_size=config.unimol_model_size,
        use_ddp=False,
    )
    result = representation_model.get_repr(smiles, return_atomic_reprs=False)
    if isinstance(result, dict):
        if "cls_repr" not in result:
            raise KeyError(
                f"Uni-Mol output has no 'cls_repr'; keys are {sorted(result)}."
            )
        values = result["cls_repr"]
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        if any("cls_repr" not in item for item in result):
            raise KeyError("At least one Uni-Mol result item lacks 'cls_repr'.")
        values = [item["cls_repr"] for item in result]
    else:
        # Some unimol-tools versions directly return the ordered CLS vectors.
        values = result
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(smiles):
        raise RuntimeError(
            f"Uni-Mol returned shape {embeddings.shape}; expected "
            f"({len(smiles)}, embedding_dim)."
        )
    return embeddings


def imagemol_external_embeddings(
    smiles: list[str],
    *,
    config: PretrainedInferenceConfig,
) -> np.ndarray:
    if torch is None:
        raise ImportError("PyTorch is required for ImageMol inference.")
    if config.imagemol_checkpoint is None:
        raise ValueError(
            "ImageMol checkpoints were requested, but --imagemol-checkpoint "
            "was not supplied."
        )
    checkpoint_path = config.imagemol_checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise ImportError(
            "ImageMol inference requires rdkit, torch, and torchvision."
        ) from exc

    class MoleculeImageDataset(Dataset):
        def __init__(self, values: list[str]):
            self.values = values
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            molecule = Chem.MolFromSmiles(self.values[index])
            if molecule is None:
                raise ValueError(
                    f"Invalid SMILES for ImageMol at row {index}: "
                    f"{self.values[index]!r}"
                )
            image = Draw.MolsToGridImage(
                [molecule], molsPerRow=1, subImgSize=(224, 224)
            ).convert("RGB")
            return self.transform(image)

    encoder = models.resnet18(weights=None)
    encoder.fc = torch.nn.Linear(encoder.fc.in_features, 1)
    official = torch_load_checkpoint(checkpoint_path)
    if config.imagemol_resume_key not in official:
        raise KeyError(
            f"{checkpoint_path} has no {config.imagemol_resume_key!r}; "
            f"available keys: {sorted(official)}"
        )
    official_state = official[config.imagemol_resume_key]
    if not isinstance(official_state, dict):
        raise TypeError("The ImageMol resume key does not contain a state dictionary.")
    current_state = encoder.state_dict()
    source_keys = list(official_state)[:120]
    destination_keys = list(current_state)[:120]
    if len(source_keys) != 120 or len(destination_keys) != 120:
        raise RuntimeError("Unexpected ResNet-18/ImageMol state-dict layout.")
    for source_key, destination_key in zip(source_keys, destination_keys):
        source = official_state[source_key]
        if source.shape != current_state[destination_key].shape:
            raise RuntimeError(
                f"ImageMol checkpoint mismatch: {source_key} "
                f"{tuple(source.shape)} -> {destination_key} "
                f"{tuple(current_state[destination_key].shape)}"
            )
        current_state[destination_key] = source
    encoder.load_state_dict(current_state)
    encoder.fc = torch.nn.Identity()
    device = resolve_torch_device(config.device_name)
    encoder = encoder.to(device).eval()
    loader = DataLoader(
        MoleculeImageDataset(smiles),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.imagemol_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            chunks.append(encoder(images.to(device)).detach().cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    if embeddings.shape != (len(smiles), 512):
        raise RuntimeError(
            f"Expected ImageMol embeddings of shape ({len(smiles)}, 512), "
            f"got {embeddings.shape}."
        )
    return embeddings


def external_embeddings_for_encoder(
    external: pd.DataFrame,
    *,
    smiles_col: str,
    encoder_name: str,
    config: PretrainedInferenceConfig,
) -> np.ndarray:
    if smiles_col not in external.columns:
        raise KeyError(
            f"Frozen pretrained model {encoder_name!r} requires the "
            f"SMILES column {smiles_col!r}."
        )
    if encoder_name in config.embedding_cache:
        return config.embedding_cache[encoder_name]
    smiles = external[smiles_col].astype(str).tolist()
    if encoder_name.startswith("unimol"):
        embeddings = unimol_external_embeddings(
            smiles, encoder_name=encoder_name, config=config
        )
    elif encoder_name == "imagemol_resnet18_frozen":
        embeddings = imagemol_external_embeddings(smiles, config=config)
    else:
        raise ValueError(
            f"Unsupported frozen encoder {encoder_name!r} in pretrained checkpoint."
        )
    config.embedding_cache[encoder_name] = embeddings
    return embeddings


def predict_pretrained_head(
    *,
    model_path: Path,
    external: pd.DataFrame,
    smiles_col: str,
    config: PretrainedInferenceConfig,
    regression_threshold: float = MARGIN,
) -> FoldPrediction:
    if torch is None:
        raise ImportError("PyTorch is required for pretrained model inference.")
    
    checkpoint = torch_load_checkpoint(model_path)

    if str(checkpoint.get("model_name", "")).startswith("gnn_"):
        import importlib
        gnn_module = importlib.import_module("08_train_gnn_mtl_ablation")
        BBBGINE = gnn_module.BBBGINE
        smiles_to_graph = gnn_module.smiles_to_graph
        from torch_geometric.loader import DataLoader

        if smiles_col not in external.columns:
            raise KeyError(f"Missing SMILES column {smiles_col!r}; specify --smiles-col.")
        if external[smiles_col].isna().any():
            raise ValueError(f"{smiles_col} contains missing SMILES.")

        device = resolve_torch_device(config.device_name)
        model = BBBGINE(
            hidden_dim=int(checkpoint["hidden_dim"]),
            n_layers=int(checkpoint["layers"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device).eval()

        graphs = [
            smiles_to_graph(smiles, float("nan"), 0, i)
            for i, smiles in enumerate(external[smiles_col].astype(str))
        ]
        loader = DataLoader(
            graphs, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            predictions = [
                model(batch.to(device)).detach().cpu().numpy()
                for batch in loader
            ]
        predicted_logbb = np.concatenate(predictions).astype(float)

        if predicted_logbb.shape != (len(external),) or not np.isfinite(predicted_logbb).all():
            raise ValueError(f"{model_path}: invalid or nonfinite GNN predictions.")

        threshold = float(regression_threshold)
        return FoldPrediction(
            fold=pretrained_fold_number(model_path),
            predicted_logbb=predicted_logbb,
            score_np=-predicted_logbb,
            pred_np=(predicted_logbb < threshold).astype(int),
            probability_np=None,
            decision_threshold=threshold,
            score_type="negative_predicted_logbb",
            source=f"{model_path}; architecture=GINE; native_logbb_threshold={threshold}",
        )

    required = {
        "model_state_dict",
        "embedding_mean",
        "embedding_scale",
        "encoder_name",
        "input_dim",
        "params",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"{model_path} is missing checkpoint fields: {missing}")
    encoder_name = str(checkpoint["encoder_name"])
    embeddings = external_embeddings_for_encoder(
        external,
        smiles_col=smiles_col,
        encoder_name=encoder_name,
        config=config,
    )
    input_dim = int(checkpoint["input_dim"])
    mean = np.asarray(checkpoint["embedding_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["embedding_scale"], dtype=np.float32)
    if embeddings.shape != (len(external), input_dim):
        raise ValueError(
            f"{model_path}: embeddings have shape {embeddings.shape}, expected "
            f"({len(external)}, {input_dim})."
        )
    if mean.shape != (input_dim,) or scale.shape != (input_dim,):
        raise ValueError(
            f"{model_path}: invalid normalization shapes {mean.shape}, {scale.shape}."
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"{model_path}: invalid embedding normalization values.")

    class LinearTransferHead(torch.nn.Module):
        def __init__(self, dimension: int, dropout: float):
            super().__init__()
            self.dropout = torch.nn.Dropout(dropout)
            self.output = torch.nn.Linear(dimension, 1)

        def forward(self, values):
            return self.output(self.dropout(values)).squeeze(-1)

    params = checkpoint["params"]
    if not isinstance(params, dict):
        raise TypeError(f"{model_path}: params must be a dictionary.")
    model = LinearTransferHead(input_dim, float(params.get("dropout", 0.0)))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = resolve_torch_device(config.device_name)
    model = model.to(device).eval()
    standardized = ((embeddings - mean) / scale).astype(np.float32)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(standardized), config.batch_size):
            tensor = torch.from_numpy(
                standardized[start : start + config.batch_size]
            ).to(device)
            chunks.append(model(tensor).detach().cpu().numpy())
    predicted_logbb = np.concatenate(chunks).astype(float)

    # Use the same native logBB cutoff for fold and ensemble decisions.
    logbb_threshold = float(regression_threshold)
    return FoldPrediction(
        fold=pretrained_fold_number(model_path),
        predicted_logbb=predicted_logbb,
        score_np=-predicted_logbb,
        pred_np=(predicted_logbb < logbb_threshold).astype(int),
        probability_np=None,
        decision_threshold=logbb_threshold,
        score_type="negative_predicted_logbb",
        source=(
            f"{model_path}; encoder={encoder_name}; "
            f"native_logbb_threshold={logbb_threshold}"
        ),
    )


def class_one_score(
    estimator: Any,
    X: pd.DataFrame,
    *,
    probability_threshold: float,
    decision_threshold: float,
) -> tuple[np.ndarray, float, np.ndarray | None, str]:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
        if probabilities.ndim != 2 or 1 not in classes:
            raise ValueError(
                "Classifier predict_proba output has no class-1 column."
            )
        index = int(np.flatnonzero(classes == 1)[0])
        score = probabilities[:, index]
        return score, probability_threshold, score, "probability"

    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        if score.ndim == 2:
            classes = np.asarray(getattr(estimator, "classes_", []))
            if 1 not in classes:
                raise ValueError(
                    "Classifier decision_function output has no class 1."
                )
            score = score[:, int(np.flatnonzero(classes == 1)[0])]
        return score.ravel(), decision_threshold, None, "decision_function"

    prediction = np.asarray(estimator.predict(X), dtype=float).ravel()
    if not set(np.unique(prediction)).issubset({0.0, 1.0}):
        raise ValueError(
            "A classifier without predict_proba or decision_function must "
            "return hard labels encoded as 0 and 1."
        )
    return prediction, probability_threshold, None, "hard_label"


# ---------------------------------------------------------------------------
# Current artifact loaders
# ---------------------------------------------------------------------------

def load_tuned_threshold_for_fold(
    model_dir: Path,
    model_name: str,
    fold: int,
    threshold_pattern: str | None = None,
) -> tuple[float, str]:
    """Load the tuned classification threshold for one outer-fold model."""
    pattern = (
        threshold_pattern.format(fold=fold)
        if threshold_pattern is not None
        else f"{model_name}_outer_{fold}.json"
    )
    candidates = sorted(model_dir.glob(pattern))

    matches = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        if not isinstance(payload, dict) or "tuned_threshold" not in payload:
            continue

        try:
            saved_fold = int(payload.get("outer_fold"))
            threshold = float(payload["tuned_threshold"])
        except (TypeError, ValueError):
            continue

        if saved_fold == fold and np.isfinite(threshold):
            matches.append((threshold, str(path)))

    if len(matches) != 1:
        raise ValueError(
            f"{model_name}, fold {fold}: expected exactly one valid tuned-"
            f"threshold JSON using {pattern!r}; found "
            f"{[path for _, path in matches]}."
        )

    threshold, source = matches[0]
    return threshold, source


def predict_joblib_model(
    *,
    model_name: str,
    task: str,
    model_path: Path,
    external: pd.DataFrame,
    regression_threshold: float,
    classifier_threshold: float,
    classifier_decision_threshold: float,
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

    if task == "regression":
        predicted_logbb = np.asarray(estimator.predict(X), dtype=float).ravel()
        score_np = -predicted_logbb
        pred_np = (predicted_logbb < regression_threshold).astype(int)
        probability_np = None
        threshold = regression_threshold
        score_type = "negative_predicted_logbb"
    else:
        score_np, score_threshold, probability_np, score_type = class_one_score(
            estimator,
            X,
            probability_threshold=classifier_threshold,
            decision_threshold=classifier_decision_threshold,
        )
        pred_np = (score_np >= score_threshold).astype(int)
        predicted_logbb = None
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
        score_type=score_type,
        source=str(model_path),
    )


def locate_custom_preprocessor(
    model_dir: Path,
    fold: int,
) -> Path:
    preferred = (
        model_dir
        / f"custom_mtl_bab_outer_{fold}_preprocessor.joblib"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        model_dir.glob(
            f"custom_mtl_bab_outer_{fold}_*preprocessor.joblib"
        )
    )
    if not candidates:
        candidates = sorted(
            model_dir.glob(
                f"custom_mtl_bab_outer_{fold}_*selector.joblib"
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

    # Current no-imputation custom-MTL artifact. Preserve NaNs so the native
    # LightGBM missing-value handling used during training is reproduced.
    if feature_names is not None and not any(
        saved.get(key) is not None
        for key in ["imputer", "preprocessor", "feature_selector", "selector"]
    ):
        X = numeric_frame(external, feature_names, context=context)
        return X.to_numpy(dtype=float), feature_names

    # Original model-#4 artifact: saved median imputer + feature order.
    if saved.get("imputer") is not None:
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
    global lgb
    if lgb is None:
        try:
            import lightgbm as lightgbm_module
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required to load native custom-MTL .txt models. "
                "It is not required for other model families."
            ) from exc
        lgb = lightgbm_module
    fold = fold_number(model_path)
    booster = lgb.Booster(model_file=str(model_path))
    preprocessor = joblib.load(preprocessor_path)

    X, _ = transform_custom_external(
        preprocessor,
        external,
        context=f"custom_mtl_bab, fold {fold}",
    )
    predicted_logbb = np.asarray(booster.predict(X), dtype=float).ravel()

    return FoldPrediction(
        fold=fold,
        predicted_logbb=predicted_logbb,
        score_np=-predicted_logbb,
        pred_np=(predicted_logbb < regression_threshold).astype(int),
        probability_np=None,
        decision_threshold=float(regression_threshold),
        score_type="negative_predicted_logbb",
        source=f"{model_path}; {preprocessor_path}",
    )



def discover_joblib_artifacts(
    *,
    model_name: str,
    model_dir: Path,
    artifact_pattern: str | None = None,
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
        "dummy_mean_regression": [
            "dummy_mean_regression_outer_*.joblib",
            "dummy_mean_regression_threshold_outer_*.joblib",
        ],
        "lightgbm_mse_regression": [
            "lightgbm_mse_regression_outer_*.joblib",
            "lightgbm_mse_regression_threshold_outer_*.joblib",
        ],
        "lightgbm_mse_regression_threshold": [
            "lightgbm_mse_regression_threshold_outer_*.joblib",
        ],
        "dummy_prior_classification": [
            "dummy_prior_classification_outer_*.joblib",
            "dummy_prior_classification_threshold_outer_*.joblib",
        ],
        "logbb_pred_shaker_retrained": [
            "logbb_pred_shaker_retrained_outer_*.joblib",
            "logbb_pred_matched_outer_*.joblib",
        ],
    }

    patterns = (
        [artifact_pattern]
        if artifact_pattern is not None
        else pattern_families.get(
            model_name,
            [
                f"{model_name}_outer_*.joblib",
                f"{model_name}_outer_*.pkl",
                f"{model_name}_outer_*.pickle",
                f"{model_name}_fold_*.joblib",
                f"{model_name}_fold_*.pkl",
                f"{model_name}_fold_*.pickle",
            ],
        )
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


def infer_model_task(
    *,
    model_name: str,
    model_dir: Path,
    artifact_pattern: str | None,
    explicit_tasks: dict[str, str],
) -> str:
    """Resolve a model task without limiting model names to a fixed registry."""
    if model_name in explicit_tasks:
        return explicit_tasks[model_name]
    if model_name in MODEL_TASKS:
        return MODEL_TASKS[model_name]

    pretrained_paths, _ = discover_pretrained_artifacts(
        model_name=model_name,
        model_dir=model_dir,
        artifact_pattern=artifact_pattern,
    )
    if pretrained_paths:
        checkpoint = torch_load_checkpoint(pretrained_paths[0])
        if str(checkpoint.get("model_name", "")).startswith("gnn_"):
            return "regression"
        model_kind = str(checkpoint.get("model_kind", "")).strip().lower()
        if model_kind not in {"regression", "mtl"}:
            raise ValueError(
                f"{pretrained_paths[0]} has unsupported model_kind "
                f"{model_kind!r}; expected 'regression' or 'mtl'."
            )
        # Both frozen-head objectives produce a continuous predicted logBB.
        return "regression"

    paths, _ = discover_joblib_artifacts(
        model_name=model_name,
        model_dir=model_dir,
        artifact_pattern=artifact_pattern,
    )
    if paths:
        try:
            estimator, metadata = unwrap_loaded(joblib.load(paths[0]))
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect {paths[0].name} to determine whether "
                f"{model_name!r} is regression or classification. Supply "
                f"--model-task {model_name}=regression (or classification)."
            ) from exc

        for key in ["task", "model_task", "estimator_type"]:
            saved_task = str(metadata.get(key, "")).lower().strip()
            if saved_task in {"regression", "regressor"}:
                return "regression"
            if saved_task in {"classification", "classifier"}:
                return "classification"

        if is_classifier(estimator) or getattr(estimator, "_estimator_type", None) == "classifier":
            return "classification"
        if is_regressor(estimator) or getattr(estimator, "_estimator_type", None) == "regressor":
            return "regression"

    raise ValueError(
        f"The task for new model {model_name!r} could not be detected. Add "
        f"--model-task {model_name}=regression or "
        f"--model-task {model_name}=classification."
    )


def load_fold_predictions(
    *,
    model_name: str,
    task: str,
    model_dir: Path,
    external: pd.DataFrame,
    smiles_col: str,
    regression_threshold: float,
    classifier_threshold: float,
    classifier_decision_threshold: float,
    tuned_threshold_models: set[str],
    pretrained_config: PretrainedInferenceConfig,
    artifact_pattern: str | None = None,
    threshold_pattern: str | None = None,
) -> list[FoldPrediction]:
    pretrained_paths, matched_pretrained_pattern = discover_pretrained_artifacts(
        model_name=model_name,
        model_dir=model_dir,
        artifact_pattern=artifact_pattern,
    )
    if pretrained_paths:
        if task != "regression":
            raise ValueError(
                f"{model_name}: frozen pretrained heads produce predicted logBB "
                "and must use task=regression."
            )
        print(
            f"Found {len(pretrained_paths)} pretrained checkpoint(s) using "
            f"{matched_pretrained_pattern}"
        )
        return [
            predict_pretrained_head(
                model_path=path,
                external=external,
                smiles_col=smiles_col,
                config=pretrained_config,
                regression_threshold=regression_threshold,
            )
            for path in pretrained_paths
        ]

    if model_name == "custom_mtl_bab":
        text_models = sorted(
            model_dir.glob("custom_mtl_bab_outer_*.txt"),
            key=fold_number,
        )

        # Allow a joblib custom estimator as a fallback.
        if not text_models:
            joblib_models = sorted(
                model_dir.glob("custom_mtl_bab_outer_*.joblib"),
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
                    task=task,
                    model_path=path,
                    external=external,
                    regression_threshold=regression_threshold,
                    classifier_threshold=classifier_threshold,
                    classifier_decision_threshold=classifier_decision_threshold,
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
        artifact_pattern=artifact_pattern,
    )
    if paths:
        print(
            f"Found {len(paths)} artifact(s) using "
            f"{matched_pattern}"
        )

    output = [
        predict_joblib_model(
            model_name=model_name,
            task=task,
            model_path=path,
            external=external,
            regression_threshold=regression_threshold,
            classifier_threshold=classifier_threshold,
            classifier_decision_threshold=classifier_decision_threshold,
        )
        for path in paths
    ]

    # Threshold-ablation model: each saved outer-fold regressor has its own
    # threshold selected from that fold's inner OOF predictions.
    if model_name in tuned_threshold_models:
        if task != "regression":
            raise ValueError(
                f"{model_name}: tuned thresholds are only supported for "
                "regression models producing predicted logBB."
            )
        for fold_prediction in output:
            threshold, threshold_source = load_tuned_threshold_for_fold(
                model_dir,
                model_name,
                fold_prediction.fold,
                threshold_pattern=threshold_pattern,
            )
            fold_prediction.pred_np = (
                fold_prediction.predicted_logbb < threshold
            ).astype(int)
            fold_prediction.decision_threshold = threshold
            fold_prediction.source += f"; threshold={threshold_source}"

    return output


# ---------------------------------------------------------------------------
# Ensembling, output, and optional external metrics
# ---------------------------------------------------------------------------

def validate_fold_predictions(
    *,
    model_name: str,
    task: str,
    folds: list[FoldPrediction],
    n_external: int,
    required_folds: list[int] | None,
) -> list[int]:
    if not folds:
        raise ValueError(f"No fold models were found for {model_name}.")

    observed_folds = sorted(fold.fold for fold in folds)
    if len(observed_folds) != len(set(observed_folds)):
        raise ValueError(f"{model_name}: duplicate folds {observed_folds}.")
    if required_folds is not None and observed_folds != required_folds:
        raise ValueError(
            f"{model_name}: expected folds {required_folds}, found "
            f"{observed_folds}. Refusing to ensemble an incomplete set."
        )

    score_types = {fold.score_type for fold in folds}
    if len(score_types) != 1:
        raise ValueError(
            f"{model_name}: fold models produced incompatible score types "
            f"{sorted(score_types)}."
        )

    for fold in folds:
        arrays = {"score_np": fold.score_np, "pred_np": fold.pred_np}
        if task == "regression":
            arrays["predicted_logbb"] = fold.predicted_logbb
        for array_name, values in arrays.items():
            if values is None:
                raise ValueError(
                    f"{model_name}, fold {fold.fold}: {array_name} is missing."
                )
            array = np.asarray(values)
            if array.ndim != 1 or len(array) != n_external:
                raise ValueError(
                    f"{model_name}, fold {fold.fold}: {array_name} must have "
                    f"shape ({n_external},), found {array.shape}."
                )
            if not np.isfinite(array.astype(float)).all():
                raise ValueError(
                    f"{model_name}, fold {fold.fold}: {array_name} contains "
                    "NaN or infinite predictions."
                )
        if not set(np.unique(fold.pred_np)).issubset({0, 1}):
            raise ValueError(
                f"{model_name}, fold {fold.fold}: predictions are not encoded "
                "as 0 and 1."
            )
        if fold.score_type == "probability" and (
            np.any(fold.score_np < 0.0) or np.any(fold.score_np > 1.0)
        ):
            raise ValueError(
                f"{model_name}, fold {fold.fold}: probability scores fall "
                "outside [0, 1]."
            )

    return observed_folds

def ensemble_predictions(
    *,
    model_name: str,
    task: str,
    folds: list[FoldPrediction],
    regression_threshold: float,
    tuned_threshold_models: set[str],
) -> dict[str, np.ndarray | float | int | str | None]:
    if not folds:
        raise ValueError(f"No fold models were found for {model_name}.")

    score_matrix = np.vstack([fold.score_np for fold in folds])
    mean_score = score_matrix.mean(axis=0)

    if task == "regression":
        logbb_matrix = np.vstack(
            [fold.predicted_logbb for fold in folds]
        )
        mean_logbb = logbb_matrix.mean(axis=0)

        if model_name in tuned_threshold_models:
            # For a fully external set, every outer-fold model predicts every
            # compound. Average each fold's signed decision margin. This is
            # equivalent to comparing mean predicted logBB with the mean of
            # the fold-specific tuned thresholds.
            threshold = float(
                np.mean([fold.decision_threshold for fold in folds])
            )
            pred_np = (mean_logbb < threshold).astype(int)
        else:
            threshold = regression_threshold
            pred_np = (mean_logbb < threshold).astype(int)

        probability_np = None
    else:
        mean_logbb = None
        fold_thresholds = {fold.decision_threshold for fold in folds}
        if len(fold_thresholds) != 1:
            raise ValueError(
                f"{model_name}: classifier folds use inconsistent thresholds "
                f"{sorted(fold_thresholds)}."
            )
        threshold = next(iter(fold_thresholds))
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
        "score_type": folds[0].score_type,
    }


def classification_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    *,
    score_type: str,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)

    np_mask = y_true == 1
    p_mask = y_true == 0
    wrong = pred != y_true

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
            if len(np.unique(y_true)) == 2 and score_type != "hard_label"
            else np.nan
        ),
        "pr_auc_nonpenetrant": (
            float(average_precision_score(y_true, score))
            if len(np.unique(y_true)) == 2 and score_type != "hard_label"
            else np.nan
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "classification_wrong_side_count": int(np.sum(wrong)),
        "classification_wrong_side_rate": float(np.mean(wrong)),
        "np_wrong_side_count": int(np.sum(wrong & np_mask)),
        "penetrant_wrong_side_count": int(np.sum(wrong & p_mask)),
    }


def regression_metrics(
    y_true: np.ndarray,
    predicted_logbb: np.ndarray,
    *,
    threshold: float,
    y_class: np.ndarray | None = None,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    predicted_logbb = np.asarray(predicted_logbb, dtype=float)
    if y_class is not None:
        y_class = np.asarray(y_class, dtype=int)
        if y_class.shape != y_true.shape:
            raise ValueError("Regression targets and class labels differ in shape.")
    evaluable = np.isfinite(y_true) & np.isfinite(predicted_logbb)
    y_true = y_true[evaluable]
    predicted_logbb = predicted_logbb[evaluable]
    if y_class is not None:
        y_class = y_class[evaluable]

    if len(y_true) == 0:
        return {"n_regression": 0}

    true_np = y_class == 1 if y_class is not None else y_true < threshold
    predicted_np = predicted_logbb < threshold
    wrong = true_np != predicted_np
    wrong_distances = np.abs(predicted_logbb[wrong] - threshold)
    np_wrong = wrong & true_np
    penetrant_wrong = wrong & ~true_np
    all_distances = np.abs(predicted_logbb - threshold)
    pearson = (
        float(np.corrcoef(y_true, predicted_logbb)[0, 1])
        if len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(predicted_logbb) > 0
        else np.nan
    )

    return {
        "n_regression": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, predicted_logbb)),
        "mse": float(mean_squared_error(y_true, predicted_logbb)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predicted_logbb))),
        "median_absolute_error": float(
            median_absolute_error(y_true, predicted_logbb)
        ),
        "r2": (
            float(r2_score(y_true, predicted_logbb))
            if len(y_true) >= 2
            else np.nan
        ),
        "pearson_r": pearson,
        "np_mae": (
            float(mean_absolute_error(y_true[true_np], predicted_logbb[true_np]))
            if np.any(true_np)
            else np.nan
        ),
        "penetrant_mae": (
            float(
                mean_absolute_error(
                    y_true[~true_np], predicted_logbb[~true_np]
                )
            )
            if np.any(~true_np)
            else np.nan
        ),
        "wrong_side_count": int(np.sum(wrong)),
        "wrong_side_rate": float(np.mean(wrong)),
        "wrong_side_mean_distance": (
            float(np.mean(wrong_distances)) if np.any(wrong) else np.nan
        ),
        "np_wrong_side_mean_distance": (
            float(np.mean(all_distances[np_wrong]))
            if np.any(np_wrong)
            else np.nan
        ),
        "penetrant_wrong_side_mean_distance": (
            float(np.mean(all_distances[penetrant_wrong]))
            if np.any(penetrant_wrong)
            else np.nan
        ),
        "median_wrong_side_distance": (
            float(np.median(wrong_distances)) if np.any(wrong) else np.nan
        ),
        "max_wrong_side_distance": (
            float(np.max(wrong_distances)) if np.any(wrong) else np.nan
        ),
    }


def build_model_outputs(
    *,
    model_name: str,
    external: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    label_col: str,
    regression_label_col: str,
    task: str,
    regression_threshold: float,
    folds: list[FoldPrediction],
    ensemble: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    output = pd.DataFrame(
        {
            id_col: external[id_col].to_numpy(),
            "model": model_name,
            "model_task": task,
            "score_type": ensemble["score_type"],
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
    if smiles_col in external.columns:
        output.insert(1, smiles_col, external[smiles_col].to_numpy())
    if label_col in external.columns:
        output[label_col] = external[label_col].to_numpy()
    if regression_label_col in external.columns:
        output[regression_label_col] = external[
            regression_label_col
        ].to_numpy()

    fold_parts = []
    for fold in folds:
        part = pd.DataFrame(
            {
                id_col: external[id_col].to_numpy(),
                "model": model_name,
                "model_task": task,
                "score_type": fold.score_type,
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
        if smiles_col in external.columns:
            part.insert(1, smiles_col, external[smiles_col].to_numpy())
        if label_col in external.columns:
            part[label_col] = external[label_col].to_numpy()
        if regression_label_col in external.columns:
            part[regression_label_col] = external[
                regression_label_col
            ].to_numpy()
        fold_parts.append(part)

    fold_output = pd.concat(fold_parts, ignore_index=True)

    metrics: dict[str, Any] = {
        "model": model_name,
        "model_task": task,
        "score_type": ensemble["score_type"],
    }
    if label_col in external.columns:
        metrics.update(
            classification_metrics(
                external[label_col].to_numpy(dtype=int),
                output["pred_np"].to_numpy(dtype=int),
                output["score_np"].to_numpy(dtype=float),
                score_type=str(ensemble["score_type"]),
            )
        )
    if task == "regression" and regression_label_col in external.columns:
        metrics.update(
            regression_metrics(
                external[regression_label_col].to_numpy(dtype=float),
                output["predicted_logbb"].to_numpy(dtype=float),
                threshold=float(ensemble["threshold"]),
                y_class=(
                    external[label_col].to_numpy(dtype=int)
                    if label_col in external.columns
                    else None
                ),
            )
        )
        if label_col in external.columns:
            measured = external[regression_label_col].to_numpy(dtype=float)
            finite = np.isfinite(measured)
            class_from_logbb = (measured[finite] < regression_threshold).astype(int)
            supplied_class = external[label_col].to_numpy(dtype=int)[finite]
            metrics["label_logbb_disagreement_count"] = int(
                np.sum(class_from_logbb != supplied_class)
            )

    if len(metrics) == 3:
        metrics = None

    return output, fold_output, metrics


def check_output_collisions(
    *,
    output_dir: Path,
    model_names: list[str],
    overwrite: bool,
) -> None:
    planned = [
        output_dir / "all_selected_models.csv",
        output_dir / "external_metrics.csv",
        output_dir / "prediction_manifest.json",
    ]
    for model_name in model_names:
        planned.extend(
            [
                output_dir / f"{model_name}.csv",
                output_dir / f"{model_name}_fold_predictions.csv",
            ]
        )

    if overwrite:
        existing = [path for path in planned if path.exists()]
        nonfiles = [path for path in existing if not path.is_file()]
        if nonfiles:
            raise IsADirectoryError(
                "Expected output files but found directories: "
                + ", ".join(str(path) for path in nonfiles)
            )
        for path in existing:
            path.unlink()
        return

    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(existing)
        )


def isolated_descriptor_predictions(model_name: str) -> tuple[str, list[FoldPrediction]]:
    """Use a fresh interpreter so LightGBM does not share PyTorch's runtime."""
    with tempfile.TemporaryDirectory(prefix="bbbx_external_") as temporary:
        result_path = Path(temporary) / "fold_predictions.joblib"
        print("Running descriptor model in a separate Python process.", flush=True)
        completed = subprocess.run(
            [
                sys.executable, "-u", str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--_descriptor-worker", model_name, str(result_path),
            ],
            check=False,
        )
        if completed.returncode != 0:
            reason = (
                f"signal {signal.Signals(-completed.returncode).name}"
                if completed.returncode < 0
                else f"exit code {completed.returncode}"
            )
            raise RuntimeError(
                f"{model_name}: isolated descriptor process failed with {reason}. "
                "See its error output above."
            )
        task, fold_records = joblib.load(result_path)
    return task, [FoldPrediction(**record) for record in fold_records]


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)

    explicit_tasks = parse_key_value_options(
        args.model_task,
        option="--model-task",
        allowed_values={"regression", "classification"},
    )
    artifact_patterns = parse_key_value_options(
        args.artifact_pattern,
        option="--artifact-pattern",
    )
    threshold_patterns = parse_key_value_options(
        args.threshold_pattern,
        option="--threshold-pattern",
    )
    required_folds = parse_required_folds(args.required_folds)
    auto_required_folds: list[int] | None = None
    tuned_threshold_models = set(TUNED_THRESHOLD_MODELS)
    for model_name in args.tuned_threshold_model:
        validate_model_name(model_name)
        tuned_threshold_models.add(model_name)

    if not np.isfinite(args.regression_threshold):
        raise ValueError("--regression-threshold must be finite.")
    if not 0.0 <= args.classifier_threshold <= 1.0:
        raise ValueError("--classifier-threshold must be between 0 and 1.")
    if not np.isfinite(args.classifier_decision_threshold):
        raise ValueError("--classifier-decision-threshold must be finite.")
    if args.pretrained_batch_size < 1:
        raise ValueError("--pretrained-batch-size must be positive.")
    if args.imagemol_workers < 0:
        raise ValueError("--imagemol-workers cannot be negative.")
    configured_columns = [
        args.id_col,
        args.smiles_col,
        args.label_col,
        args.regression_label_col,
    ]
    if len(configured_columns) != len(set(configured_columns)):
        raise ValueError(
            "--id-col, --smiles-col, --label-col, and "
            "--regression-label-col must name distinct columns."
        )
    reserved_collisions = sorted(
        set(configured_columns).intersection(RESERVED_OUTPUT_COLUMNS)
    )
    if reserved_collisions:
        raise ValueError(
            "Configured input column names collide with reserved output "
            f"columns: {reserved_collisions}."
        )
    discovered_models = discover_model_names(model_dir)

    if args.list_models:
        print("Built-in model names:")
        for model, task in MODEL_TASKS.items():
            print(f"  {model:<36} {task}")
        print("\nAliases:")
        for alias, values in ALIASES.items():
            print(f"  {alias:<24} {','.join(values)}")
        print("  all_discovered          every conventional artifact family")
        print("\nDiscovered in model directory:")
        if discovered_models:
            for model in discovered_models:
                print(f"  {model}")
        else:
            print("  (none)")
        return

    model_names = expand_names(
        args.models,
        args.model,
        discovered_models,
    )
    pretrained_config = PretrainedInferenceConfig(
        device_name=args.pretrained_device,
        batch_size=args.pretrained_batch_size,
        unimol_model_size=args.unimol_model_size,
        unimol_remove_hs=args.unimol_remove_hs,
        imagemol_checkpoint=args.imagemol_checkpoint,
        imagemol_resume_key=args.imagemol_resume_key,
        imagemol_workers=args.imagemol_workers,
        embedding_cache={},
    )
    external = prepare_external(
        args.external_csv,
        args.id_col,
        args.label_col,
        args.regression_label_col,
    )

    def prepare_model(model_name: str) -> tuple[str, list[FoldPrediction]]:
        task = infer_model_task(
            model_name=model_name,
            model_dir=model_dir,
            artifact_pattern=artifact_patterns.get(model_name),
            explicit_tasks=explicit_tasks,
        )
        print(f"Task: {task}", flush=True)
        folds = load_fold_predictions(
            model_name=model_name,
            task=task,
            model_dir=model_dir,
            external=external,
            smiles_col=args.smiles_col,
            regression_threshold=args.regression_threshold,
            classifier_threshold=args.classifier_threshold,
            classifier_decision_threshold=args.classifier_decision_threshold,
            tuned_threshold_models=tuned_threshold_models,
            pretrained_config=pretrained_config,
            artifact_pattern=artifact_patterns.get(model_name),
            threshold_pattern=threshold_patterns.get(model_name),
        )
        return task, folds

    if args._descriptor_worker is not None:
        worker_model, result_path = args._descriptor_worker
        if discover_pretrained_artifacts(
            model_name=worker_model,
            model_dir=model_dir,
            artifact_pattern=artifact_patterns.get(worker_model),
        )[0]:
            raise ValueError("A descriptor worker cannot load .pt checkpoints.")
        task, folds = prepare_model(worker_model)
        # Only plain metadata and prediction arrays cross the process boundary.
        # The parent retains all fold validation, ensembling, metrics and writes.
        joblib.dump((task, [asdict(fold) for fold in folds]), result_path)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_output_collisions(
        output_dir=output_dir,
        model_names=model_names,
        overwrite=args.overwrite,
    )

    combined: list[pd.DataFrame] = []
    fold_outputs: dict[str, pd.DataFrame] = {}
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
            pretrained_paths, _ = discover_pretrained_artifacts(
                model_name=model_name,
                model_dir=model_dir,
                artifact_pattern=artifact_patterns.get(model_name),
            )
            if torch is not None and not pretrained_paths:
                task, folds = isolated_descriptor_predictions(model_name)
            else:
                task, folds = prepare_model(model_name)
            if not folds:
                accepted = artifact_patterns.get(model_name) or (
                    "logbb_pred_shaker_retrained_outer_*.joblib or "
                    "logbb_pred_matched_outer_*.joblib"
                    if model_name == "logbb_pred_shaker_retrained"
                    else (
                        f"{model_name}_outer_*.joblib/.pkl or "
                        f"{model_name}_fold_*.joblib/.pkl, flat .pt files, "
                        f"or {model_name}/outer_fold_*/model.pt"
                    )
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

            observed_folds = validate_fold_predictions(
                model_name=model_name,
                task=task,
                folds=folds,
                n_external=len(external),
                required_folds=(
                    required_folds
                    if required_folds is not None
                    else auto_required_folds
                ),
            )
            if required_folds is None and auto_required_folds is None:
                auto_required_folds = observed_folds

            ensemble = ensemble_predictions(
                model_name=model_name,
                task=task,
                folds=folds,
                regression_threshold=args.regression_threshold,
                tuned_threshold_models=tuned_threshold_models,
            )
            output, fold_output, metrics = build_model_outputs(
                model_name=model_name,
                external=external,
                id_col=args.id_col,
                smiles_col=args.smiles_col,
                label_col=args.label_col,
                regression_label_col=args.regression_label_col,
                task=task,
                regression_threshold=args.regression_threshold,
                folds=folds,
                ensemble=ensemble,
            )
            combined.append(output)
            fold_outputs[model_name] = fold_output
            if metrics is not None:
                metric_rows.append(metrics)

            manifest_models[model_name] = {
                "task": task,
                "n_fold_models": len(folds),
                "folds": [fold.fold for fold in folds],
                "sources": [fold.source for fold in folds],
                "ensemble_threshold": ensemble["threshold"],
                "threshold_variant": (
                    "inner_tuned" if model_name in tuned_threshold_models
                    else "native"
                ),
                "score_type": ensemble["score_type"],
            }
            print(
                f"Prepared {model_name} using "
                f"{len(folds)} saved fold models."
            )

        except Exception as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {failures[model_name]}", file=sys.stderr)
            if args.strict:
                raise

    manifest = {
        "status": (
            "failed"
            if failures and not args.allow_partial
            else "partial"
            if failures
            else "succeeded"
        ),
        "external_csv": str(args.external_csv.resolve()),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "n_external_compounds": int(len(external)),
        "requested_models": model_names,
        "successful_models": list(manifest_models),
        "failed_models": failures,
        "models": manifest_models,
        "regression_threshold": args.regression_threshold,
        "regression_threshold_variant": (
            "per_model" if set(model_names) & tuned_threshold_models else "native"
        ),
        "classifier_threshold": args.classifier_threshold,
        "classifier_decision_threshold": args.classifier_decision_threshold,
        "required_folds": (
            required_folds
            if required_folds is not None
            else auto_required_folds
        ),
        "tuned_threshold_models": sorted(tuned_threshold_models),
        "pretrained_inference": {
            "threshold_variant": "native",
            "logbb_threshold": args.regression_threshold,
            "device": args.pretrained_device,
            "batch_size": args.pretrained_batch_size,
            "unimol_model_size": args.unimol_model_size,
            "unimol_remove_hs": args.unimol_remove_hs,
            "imagemol_checkpoint": (
                str(args.imagemol_checkpoint.resolve())
                if args.imagemol_checkpoint is not None
                else None
            ),
            "imagemol_resume_key": args.imagemol_resume_key,
            "imagemol_workers": args.imagemol_workers,
        },
        "label_column_present": args.label_col in external.columns,
        "smiles_column_present": args.smiles_col in external.columns,
        "regression_label_column_present": (
            args.regression_label_col in external.columns
        ),
    }
    if failures:
        print("\nFailed models:")
        for model, error in failures.items():
            print(f"  {model}: {error}")

    if not combined:
        (output_dir / "prediction_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        raise RuntimeError(
            "No requested model produced predictions. See failure messages."
        )

    if failures and not args.allow_partial:
        (output_dir / "prediction_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        raise RuntimeError(
            "One or more requested models failed. Only the diagnostic manifest "
            "was written; prediction CSVs were not partially replaced. Use "
            "--allow-partial only when partial evaluation is intentional."
        )

    for model_name, output in zip(manifest_models, combined):
        output.to_csv(output_dir / f"{model_name}.csv", index=False)
        fold_outputs[model_name].to_csv(
            output_dir / f"{model_name}_fold_predictions.csv",
            index=False,
        )

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

    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nCompleted successfully.")
    print("Combined predictions:", output_dir / "all_selected_models.csv")
    if metric_rows:
        print("External metrics:", output_dir / "external_metrics.csv")


if __name__ == "__main__":
    main()
