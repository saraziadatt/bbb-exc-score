#!/usr/bin/env python3
"""Diagnose whether fitted custom-MTL LightGBM Hessians are well behaved.

This script performs two complementary checks:

1. Sample-level analytic Hessians
   The custom objective is reconstructed on each outer-training fold at the
   predictions used immediately before selected trees are fitted, plus once
   after the final tree.

2. Training-time leaf Hessian sums
   The fitted model's stored ``leaf_weight`` values are extracted from
   ``Booster.dump_model()``. In LightGBM, these are the aggregate Hessian
   weights stored for the fitted leaves and are more authoritative than
   regrouping every training row through each tree after fitting.

The outputs needed for a concise Hessian assessment are:

    hessian_snapshot_diagnostics.csv
    stored_leaf_hessian_diagnostics.csv
    fold_hessian_summary.csv
    diagnostic_manifest.json

Important reconstruction assumption
-----------------------------------
The master CSV, row order, fold assignments, saved preprocessing object, loss
parameters, and initial score must match those used during training. This
script records the files and assumptions, but it cannot prove that an older
model was trained from an unchanged CSV unless the training pipeline saved
matching hashes or row identifiers.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


# =============================================================================
# USER PATHS / SETTINGS: EDIT THESE
# =============================================================================

MASTER_CSV = Path(
    "../datasets/druglike_b3db_labelled_outerfolds.csv"
)

MODEL_DIR = Path(
    "./custom_mtl_results/fitted_models"
)

OUTPUT_DIR = Path(
    "./custom_mtl_hessian_diagnostics"
)

TARGET_COL = "logBB"
CLASS_COL = "bbb_class"
FOLD_COL = "outer_fold"

# "original":
#   grad_reg = alpha * residual
#   hess_reg = alpha
#
# "mse":
#   grad_reg = 2 * alpha / n_regression * residual
#   hess_reg = 2 * alpha / n_regression
LOSS_MODE = "original"  # choices: "original", "mse"

# Keep model naming tied to LOSS_MODE so the derivative mode cannot be changed
# without also changing the selected model family.
MODEL_PREFIX_BY_LOSS_MODE = {
    "original": "custom_mtl_original",
    "mse": "custom_mtl_mse_tuned_weights",
}
MODEL_PREFIX = MODEL_PREFIX_BY_LOSS_MODE[LOSS_MODE]
MODEL_GLOB = f"{MODEL_PREFIX}_outer_*.txt"
PREPROCESSOR_TEMPLATE = f"{MODEL_PREFIX}_outer_{{fold}}_preprocessor.joblib"

# These are tree numbers, not post-fit iteration labels.
# For tree t, derivatives are evaluated using predictions after trees 1..t-1.
REQUESTED_TREE_NUMBERS = (1, 10, 25, 50, 100, 250, 500, 750, 1000)

# Your custom-objective models were trained without an explicit init_score.
# Change this only if the training data used a different initial raw score.
INITIAL_SCORE = 0.0

# Tolerance used only to classify numerical values as zero.
ZERO_TOLERANCE = 1e-12

# Features that should never be interpreted as ordinary molecular descriptors.
# They are reported as warnings because this script may intentionally diagnose
# an older model that contains one of them.
SUSPICIOUS_FEATURES = {
    TARGET_COL,
    CLASS_COL,
    FOLD_COL,
    "has_logbb",
    "logbb_is_measured",
    "outer_stratum",
}

# =============================================================================
# END USER SETTINGS
# =============================================================================


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_outer_fold(model_path: Path) -> int:
    """Extract the outer-fold integer from a model filename."""
    match = re.search(r"_outer_(-?\d+)\.txt$", model_path.name)
    if match is None:
        raise ValueError(
            f"Could not infer outer fold from model filename: {model_path.name}"
        )
    return int(match.group(1))


def require_finite_positive(name: str, value: float) -> float:
    """Validate a strictly positive, finite loss parameter."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}.")
    return value


def read_saved_parameters(
    preprocessor_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Extract and validate the loss parameters used by the fitted model."""
    parameters = dict(preprocessor_bundle.get("parameters", {}))

    if "alpha" not in parameters:
        raise KeyError("Saved parameters do not contain alpha.")
    alpha = require_finite_positive("alpha", parameters["alpha"])

    beta_value = None
    beta_source = None
    for key in ("final_beta", "beta", "inner_beta"):
        if key in parameters:
            beta_value = parameters[key]
            beta_source = key
            break

    if beta_value is None:
        raise KeyError(
            "Could not find final_beta, beta, or inner_beta in the saved "
            "parameter dictionary."
        )
    beta = require_finite_positive("beta", beta_value)

    if "margin_buffer" not in parameters:
        raise KeyError("Saved parameters do not contain margin_buffer.")
    margin_buffer = float(parameters["margin_buffer"])
    if not np.isfinite(margin_buffer) or margin_buffer < 0.0:
        raise ValueError(
            "margin_buffer must be finite and >= 0; "
            f"got {margin_buffer!r}."
        )

    margin_value = parameters.get(
        "margin",
        preprocessor_bundle.get("threshold"),
    )
    if margin_value is None:
        raise KeyError(
            "Could not find the objective margin in parameters['margin'] "
            "or preprocessor_bundle['threshold']."
        )
    margin = float(margin_value)
    if not np.isfinite(margin):
        raise ValueError(f"margin must be finite; got {margin!r}.")

    saved_loss_mode = parameters.get(
        "loss_mode",
        preprocessor_bundle.get("loss_mode"),
    )
    if saved_loss_mode is not None and str(saved_loss_mode) != LOSS_MODE:
        raise ValueError(
            f"Saved loss_mode is {saved_loss_mode!r}, but this script is "
            f"configured for LOSS_MODE={LOSS_MODE!r}."
        )

    return {
        "alpha": alpha,
        "beta": beta,
        "margin": margin,
        "margin_buffer": margin_buffer,
        "beta_source": beta_source,
    }


def read_optional_numeric_column(
    series: pd.Series,
    name: str,
) -> np.ndarray:
    """Read a numeric column while allowing genuine missing values.

    Invalid non-numeric strings raise an error instead of being silently
    converted into missing regression labels. Positive and negative infinity
    are also rejected.
    """
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} contains a non-numeric value that cannot be treated "
            "as a valid measurement or a missing value."
        ) from error

    values = numeric.to_numpy(dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} contains positive or negative infinity.")

    return values


def read_binary_class_column(series: pd.Series, name: str) -> np.ndarray:
    """Read a non-missing binary 0/1 column without silent truncation."""
    numeric = pd.to_numeric(series, errors="raise")

    if numeric.isna().any():
        raise ValueError(f"{name} contains missing values.")

    valid = numeric.isin([0, 1])
    if not bool(valid.all()):
        invalid = sorted(numeric.loc[~valid].unique().tolist())
        raise ValueError(
            f"{name} must contain only 0 and 1; found {invalid[:20]}."
        )

    return numeric.to_numpy(dtype=int)


def read_integer_column(series: pd.Series, name: str) -> np.ndarray:
    """Read a non-missing integer-valued column without silent truncation."""
    numeric = pd.to_numeric(series, errors="raise")

    if numeric.isna().any():
        raise ValueError(f"{name} contains missing values.")

    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")

    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=0.0):
        bad = values[values != rounded]
        raise ValueError(
            f"{name} must be integer-valued; found {bad[:20].tolist()}."
        )

    return rounded.astype(int)


def custom_mtl_grad_hess(
    predictions: np.ndarray,
    y_class: np.ndarray,
    y_reg_with_nans: np.ndarray,
    *,
    alpha: float,
    beta: float,
    margin: float,
    margin_buffer: float,
    loss_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Reproduce the analytic derivatives used by the custom objective.

    Class convention
    ----------------
    y_class == 0: BBB penetrant
    y_class == 1: BBB non-penetrant

    The classification penalty is active only when a prediction violates the
    class-specific buffered margin.
    """
    predictions = np.asarray(predictions, dtype=float)
    y_class = np.asarray(y_class, dtype=int)
    y_reg_with_nans = np.asarray(y_reg_with_nans, dtype=float)

    if not (
        predictions.shape
        == y_class.shape
        == y_reg_with_nans.shape
    ):
        raise ValueError(
            "predictions, y_class, and y_reg_with_nans must have equal shapes."
        )
    if predictions.ndim != 1:
        raise ValueError("This diagnostic expects one-dimensional predictions.")
    if loss_mode not in {"original", "mse"}:
        raise ValueError("loss_mode must be 'original' or 'mse'.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain non-finite values.")

    has_regression = np.isfinite(y_reg_with_nans)
    classification_only = ~has_regression

    n_regression = int(np.sum(has_regression))
    n_penetrant = int(np.sum(y_class == 0))
    n_nonpenetrant = int(np.sum(y_class == 1))

    if n_regression == 0:
        raise ValueError("No measured regression labels in this training fold.")
    if n_penetrant == 0 or n_nonpenetrant == 0:
        raise ValueError("Both BBB classes must be present in this training fold.")

    grad_reg = np.zeros_like(predictions)
    hess_reg = np.zeros_like(predictions)

    residual = (
        predictions[has_regression]
        - y_reg_with_nans[has_regression]
    )

    if loss_mode == "original":
        grad_reg[has_regression] = alpha * residual
        hess_reg[has_regression] = alpha
    else:
        regression_scale = 2.0 * alpha / n_regression
        grad_reg[has_regression] = regression_scale * residual
        hess_reg[has_regression] = regression_scale

    penetrant_target = margin + margin_buffer
    nonpenetrant_target = margin - margin_buffer

    wrong_penetrant = (
        (y_class == 0)
        & (predictions < penetrant_target)
    )
    wrong_nonpenetrant = (
        (y_class == 1)
        & (predictions > nonpenetrant_target)
    )
    wrong_side = wrong_penetrant | wrong_nonpenetrant

    grad_class = np.zeros_like(predictions)
    hess_class = np.zeros_like(predictions)

    penetrant_scale = 2.0 * beta / n_penetrant
    nonpenetrant_scale = 2.0 * beta / n_nonpenetrant

    grad_class[wrong_penetrant] = penetrant_scale * (
        predictions[wrong_penetrant] - penetrant_target
    )
    hess_class[wrong_penetrant] = penetrant_scale

    grad_class[wrong_nonpenetrant] = nonpenetrant_scale * (
        predictions[wrong_nonpenetrant] - nonpenetrant_target
    )
    hess_class[wrong_nonpenetrant] = nonpenetrant_scale

    total_grad = grad_reg + grad_class
    total_hess = hess_reg + hess_class

    masks = {
        "has_regression": has_regression,
        "classification_only": classification_only,
        "wrong_penetrant": wrong_penetrant,
        "wrong_nonpenetrant": wrong_nonpenetrant,
        "wrong_side": wrong_side,
    }
    return total_grad, total_hess, masks


def predict_raw_after_iterations(
    model: lgb.Booster,
    X: np.ndarray,
    n_iterations: int,
) -> np.ndarray:
    """Return raw predictions after exactly n_iterations boosting rounds."""
    if n_iterations < 0:
        raise ValueError("n_iterations must be >= 0.")

    if n_iterations == 0:
        return np.full(
            X.shape[0],
            float(INITIAL_SCORE),
            dtype=float,
        )

    prediction = np.asarray(
        model.predict(
            X,
            raw_score=True,
            start_iteration=0,
            num_iteration=n_iterations,
        ),
        dtype=float,
    )

    if prediction.ndim != 1:
        raise ValueError(
            "This script expects one-dimensional raw predictions; "
            f"received shape {prediction.shape}."
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"Non-finite predictions after {n_iterations} iterations."
        )

    return prediction


def summarize_snapshot(
    *,
    outer_fold: int,
    stage: str,
    tree_number: int | None,
    n_trees_in_prediction: int,
    grad: np.ndarray,
    hess: np.ndarray,
    masks: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> dict[str, Any]:
    """Summarize sample-level gradient and Hessian behavior."""
    classification_only = masks["classification_only"]
    has_regression = masks["has_regression"]
    wrong_side = masks["wrong_side"]

    zero_hess = np.abs(hess) <= ZERO_TOLERANCE
    zero_grad = np.abs(grad) <= ZERO_TOLERANCE
    positive_hess = hess[hess > ZERO_TOLERANCE]

    class_only_correct = classification_only & ~wrong_side
    class_only_wrong = classification_only & wrong_side

    # With alpha > 0 and beta > 0, the only expected zero-Hessian observations
    # are classification-only samples whose margin penalty is inactive.
    expected_zero_hess = class_only_correct
    unexpected_zero_hess = zero_hess & ~expected_zero_hess
    expected_zero_but_positive = expected_zero_hess & ~zero_hess

    class_only_hess = hess[classification_only]
    class_only_zero_hess = (
        np.abs(class_only_hess) <= ZERO_TOLERANCE
    )

    return {
        "outer_fold": outer_fold,
        "stage": stage,
        "tree_number_being_trained": tree_number,
        "n_trees_in_prediction": n_trees_in_prediction,
        "alpha": parameters["alpha"],
        "beta": parameters["beta"],
        "margin": parameters["margin"],
        "margin_buffer": parameters["margin_buffer"],
        "loss_mode": LOSS_MODE,
        "n_samples": int(len(hess)),
        "n_measured": int(np.sum(has_regression)),
        "n_classification_only": int(np.sum(classification_only)),
        "n_wrong_side": int(np.sum(wrong_side)),
        "wrong_side_fraction": float(np.mean(wrong_side)),
        "zero_hessian_count_all": int(np.sum(zero_hess)),
        "zero_hessian_fraction_all": float(np.mean(zero_hess)),
        "zero_hessian_count_measured": int(
            np.sum(zero_hess & has_regression)
        ),
        "zero_hessian_count_classification_only": int(
            np.sum(class_only_zero_hess)
        ),
        "zero_hessian_fraction_classification_only": (
            float(np.mean(class_only_zero_hess))
            if len(class_only_hess)
            else np.nan
        ),
        "classification_only_correct_count": int(
            np.sum(class_only_correct)
        ),
        "classification_only_wrong_count": int(
            np.sum(class_only_wrong)
        ),
        "unexpected_zero_hessian_count": int(
            np.sum(unexpected_zero_hess)
        ),
        "expected_zero_but_positive_hessian_count": int(
            np.sum(expected_zero_but_positive)
        ),
        "zero_gradient_and_hessian_count": int(
            np.sum(zero_grad & zero_hess)
        ),
        "zero_gradient_and_hessian_fraction": float(
            np.mean(zero_grad & zero_hess)
        ),
        "sum_hessian": float(np.sum(hess)),
        "min_hessian": float(np.min(hess)),
        "min_positive_hessian": (
            float(np.min(positive_hess))
            if len(positive_hess)
            else np.nan
        ),
        "median_positive_hessian": (
            float(np.median(positive_hess))
            if len(positive_hess)
            else np.nan
        ),
        "max_hessian": float(np.max(hess)),
        "negative_hessian_count": int(
            np.sum(hess < -ZERO_TOLERANCE)
        ),
        "nonfinite_gradient_count": int(
            np.sum(~np.isfinite(grad))
        ),
        "nonfinite_hessian_count": int(
            np.sum(~np.isfinite(hess))
        ),
    }


def iter_leaf_nodes(
    node: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield all leaf dictionaries below a dumped LightGBM tree node."""
    if "split_index" not in node:
        yield node
        return

    if "left_child" not in node or "right_child" not in node:
        raise KeyError(
            "A split node in the LightGBM dump is missing a child."
        )

    yield from iter_leaf_nodes(node["left_child"])
    yield from iter_leaf_nodes(node["right_child"])


def stored_leaf_hessian_report(
    *,
    model: lgb.Booster,
    outer_fold: int,
) -> pd.DataFrame:
    """Extract the fitted model's stored leaf Hessian sums.

    ``leaf_weight`` is taken directly from ``Booster.dump_model()`` rather
    than reconstructed by sending all rows through each fitted tree.
    """
    model_dump = model.dump_model(
        start_iteration=0,
        num_iteration=-1,
    )
    tree_info = model_dump.get("tree_info", [])

    if not tree_info:
        raise ValueError("The fitted model dump does not contain any trees.")

    expected_tree_count = int(model.num_trees())
    if len(tree_info) != expected_tree_count:
        raise ValueError(
            f"The model contains {expected_tree_count} trees, but dump_model "
            f"returned {len(tree_info)} trees."
        )

    rows: list[dict[str, Any]] = []

    for tree_number, tree in enumerate(tree_info, start=1):
        tree_structure = tree.get("tree_structure")
        if not isinstance(tree_structure, dict):
            raise TypeError(
                f"Tree {tree_number} has an invalid tree_structure."
            )

        leaves = list(iter_leaf_nodes(tree_structure))
        reported_num_leaves = int(tree.get("num_leaves", len(leaves)))

        if reported_num_leaves != len(leaves):
            raise ValueError(
                f"Tree {tree_number} reports {reported_num_leaves} leaves, "
                f"but {len(leaves)} were found in tree_structure."
            )

        for local_leaf_number, leaf in enumerate(leaves):
            has_leaf_weight = "leaf_weight" in leaf

            # LightGBM can omit leaf_weight for a one-leaf tree. That is a
            # missing stored diagnostic, not a non-finite Hessian. In a split
            # tree, however, every leaf is expected to contain leaf_weight.
            if not has_leaf_weight and reported_num_leaves > 1:
                raise KeyError(
                    f"Tree {tree_number}, leaf {local_leaf_number} is missing "
                    "leaf_weight despite belonging to a split tree."
                )

            leaf_weight = (
                float(leaf["leaf_weight"])
                if has_leaf_weight
                else np.nan
            )
            leaf_count = int(leaf.get("leaf_count", -1))
            leaf_index = int(leaf.get("leaf_index", local_leaf_number))
            leaf_value = float(leaf.get("leaf_value", np.nan))

            leaf_weight_is_finite = bool(
                has_leaf_weight and np.isfinite(leaf_weight)
            )

            rows.append(
                {
                    "outer_fold": outer_fold,
                    "tree_number": tree_number,
                    "tree_num_leaves": reported_num_leaves,
                    "stump_tree": bool(reported_num_leaves <= 1),
                    "leaf_index": leaf_index,
                    "leaf_count": leaf_count,
                    "leaf_value": leaf_value,
                    "stored_leaf_sum_hessian": leaf_weight,
                    "stored_leaf_hessian_is_missing": bool(
                        not has_leaf_weight
                    ),
                    "stored_leaf_hessian_is_nonfinite": bool(
                        has_leaf_weight and not np.isfinite(leaf_weight)
                    ),
                    "stored_leaf_hessian_is_negative": bool(
                        leaf_weight_is_finite
                        and leaf_weight < -ZERO_TOLERANCE
                    ),
                    "stored_leaf_hessian_is_zero": bool(
                        leaf_weight_is_finite
                        and abs(leaf_weight) <= ZERO_TOLERANCE
                    ),
                    "stored_leaf_hessian_is_positive": bool(
                        leaf_weight_is_finite
                        and leaf_weight > ZERO_TOLERANCE
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_model_and_preprocessing(
    *,
    model: lgb.Booster,
    feature_names: list[str],
    X_train: np.ndarray,
    model_path: Path,
) -> None:
    """Validate the model/preprocessor reconstruction."""
    if not feature_names:
        raise ValueError("Saved feature_names is empty.")

    duplicate_counts = Counter(feature_names)
    duplicates = sorted(
        name for name, count in duplicate_counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Saved feature_names contains duplicates: {duplicates[:20]}."
        )

    if X_train.ndim != 2:
        raise ValueError(
            f"Preprocessing must produce a 2D matrix; got {X_train.shape}."
        )
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Preprocessing produced {X_train.shape[1]} columns, but "
            f"{len(feature_names)} feature names were saved."
        )
    if not np.all(np.isfinite(X_train)):
        raise ValueError(
            "The preprocessed training matrix contains non-finite values."
        )

    if int(model.num_feature()) != X_train.shape[1]:
        raise ValueError(
            f"{model_path.name} expects {model.num_feature()} features, "
            f"but preprocessing produced {X_train.shape[1]}."
        )

    num_models_per_iteration = int(model.num_model_per_iteration())
    if num_models_per_iteration != 1:
        raise ValueError(
            "This diagnostic expects one model/tree per boosting iteration; "
            f"found num_model_per_iteration={num_models_per_iteration}."
        )

    n_iterations = int(model.current_iteration())
    n_trees = int(model.num_trees())
    if n_trees != n_iterations:
        raise ValueError(
            f"Expected n_trees == n_iterations for a single-output model; "
            f"found {n_trees} trees and {n_iterations} iterations."
        )

    model_feature_names = list(model.feature_name())
    generic_model_names = all(
        bool(
            re.fullmatch(
                r"(?:Column_|feature_|f)\d+",
                str(name),
            )
        )
        for name in model_feature_names
    )

    if model_feature_names != feature_names:
        if generic_model_names:
            warnings.warn(
                f"{model_path.name} stores only generic feature names. "
                "The feature count matches, but the saved model cannot "
                "independently verify descriptor order.",
                RuntimeWarning,
            )
        else:
            mismatch_index = next(
                (
                    index
                    for index, (model_name, saved_name)
                    in enumerate(zip(model_feature_names, feature_names))
                    if model_name != saved_name
                ),
                None,
            )
            raise ValueError(
                f"Feature names or ordering in {model_path.name} do not "
                "match the saved preprocessor. First mismatch index: "
                f"{mismatch_index}."
            )

    boosting_value = model.params.get(
        "boosting_type",
        model.params.get("boosting"),
    )
    if boosting_value is None:
        warnings.warn(
            f"Could not verify boosting_type from {model_path.name}. "
            "Pre-tree reconstruction assumes ordinary additive GBDT.",
            RuntimeWarning,
        )
    else:
        boosting_type = str(boosting_value).strip().lower()
        if boosting_type not in {"gbdt", "gbrt"}:
            raise ValueError(
                "Historical pre-tree predictions are reconstructed only "
                "for ordinary GBDT models; found boosting_type="
                f"{boosting_type!r}."
            )


def main() -> None:
    if LOSS_MODE not in MODEL_PREFIX_BY_LOSS_MODE:
        raise ValueError(
            f"Unsupported LOSS_MODE={LOSS_MODE!r}; expected one of "
            f"{sorted(MODEL_PREFIX_BY_LOSS_MODE)}."
        )

    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"MASTER_CSV does not exist: {MASTER_CSV}\n"
            "Edit MASTER_CSV in the USER PATHS / SETTINGS section."
        )
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"MODEL_DIR does not exist: {MODEL_DIR}\n"
            "Edit MODEL_DIR in the USER PATHS / SETTINGS section."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_CSV, low_memory=False)
    required_columns = {TARGET_COL, CLASS_COL, FOLD_COL}
    missing_required = required_columns.difference(master.columns)
    if missing_required:
        raise KeyError(
            f"Master CSV is missing columns: {sorted(missing_required)}"
        )

    y_reg_all = read_optional_numeric_column(
        master[TARGET_COL],
        TARGET_COL,
    )

    y_class_all = read_binary_class_column(
        master[CLASS_COL],
        CLASS_COL,
    )
    outer_fold_all = read_integer_column(
        master[FOLD_COL],
        FOLD_COL,
    )

    model_paths = list(MODEL_DIR.glob(MODEL_GLOB))
    if not model_paths:
        raise FileNotFoundError(
            f"No files matching {MODEL_GLOB!r} were found in {MODEL_DIR}."
        )
    model_paths = sorted(model_paths, key=extract_outer_fold)

    seen_folds: set[int] = set()
    snapshot_rows: list[dict[str, Any]] = []
    leaf_reports: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, Any]] = []
    model_manifest_rows: list[dict[str, Any]] = []

    for model_path in model_paths:
        outer_fold = extract_outer_fold(model_path)

        if outer_fold in seen_folds:
            raise ValueError(
                f"Multiple model files were found for outer fold {outer_fold}."
            )
        seen_folds.add(outer_fold)

        if not np.any(outer_fold_all == outer_fold):
            raise ValueError(
                f"Outer fold {outer_fold} from {model_path.name} does not "
                "occur in the master CSV."
            )

        preprocessor_path = MODEL_DIR / PREPROCESSOR_TEMPLATE.format(
            fold=outer_fold
        )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Missing preprocessor for outer fold {outer_fold}: "
                f"{preprocessor_path}"
            )

        print(f"Checking outer fold {outer_fold}...")

        bundle = joblib.load(preprocessor_path)
        if not isinstance(bundle, dict):
            raise TypeError(
                f"Expected a dict in {preprocessor_path}, got {type(bundle)}."
            )

        if "imputer" not in bundle:
            raise KeyError(f"{preprocessor_path} does not contain 'imputer'.")
        if "feature_names" not in bundle:
            raise KeyError(
                f"{preprocessor_path} does not contain 'feature_names'."
            )

        imputer = bundle["imputer"]
        feature_names = [str(name) for name in bundle["feature_names"]]
        parameters = read_saved_parameters(bundle)

        missing_features = sorted(
            set(feature_names).difference(master.columns)
        )
        if missing_features:
            raise KeyError(
                "The master CSV is missing features required by the saved "
                f"preprocessor: {missing_features[:20]}"
            )

        suspicious_included = sorted(
            set(feature_names).intersection(SUSPICIOUS_FEATURES)
        )
        if suspicious_included:
            warnings.warn(
                f"Outer fold {outer_fold} includes potentially leaking or "
                f"metadata features: {suspicious_included}",
                RuntimeWarning,
            )

        train_mask = outer_fold_all != outer_fold
        if not np.any(train_mask):
            raise ValueError(
                f"Outer fold {outer_fold} leaves no rows for training."
            )

        X_train_df = master.loc[train_mask, feature_names]
        X_train = np.asarray(
            imputer.transform(X_train_df),
            dtype=float,
        )
        y_reg_train = y_reg_all[train_mask]
        y_class_train = y_class_all[train_mask]

        model = lgb.Booster(model_file=str(model_path))
        validate_model_and_preprocessing(
            model=model,
            feature_names=feature_names,
            X_train=X_train,
            model_path=model_path,
        )

        n_trees = int(model.current_iteration())

        selected_tree_numbers = sorted(
            {
                tree_number
                for tree_number in REQUESTED_TREE_NUMBERS
                if 1 <= tree_number <= n_trees
            }
            | {1, n_trees}
        )

        current_fold_snapshots: list[dict[str, Any]] = []

        # Pre-tree snapshots: for tree t, use predictions from t-1 trees.
        for tree_number in selected_tree_numbers:
            n_previous_trees = tree_number - 1
            predictions = predict_raw_after_iterations(
                model,
                X_train,
                n_previous_trees,
            )

            grad, hess, masks = custom_mtl_grad_hess(
                predictions,
                y_class_train,
                y_reg_train,
                alpha=parameters["alpha"],
                beta=parameters["beta"],
                margin=parameters["margin"],
                margin_buffer=parameters["margin_buffer"],
                loss_mode=LOSS_MODE,
            )

            row = summarize_snapshot(
                outer_fold=outer_fold,
                stage="before_tree",
                tree_number=tree_number,
                n_trees_in_prediction=n_previous_trees,
                grad=grad,
                hess=hess,
                masks=masks,
                parameters=parameters,
            )
            snapshot_rows.append(row)
            current_fold_snapshots.append(row)

        # Final fitted-model snapshot.
        final_predictions = predict_raw_after_iterations(
            model,
            X_train,
            n_trees,
        )
        final_grad, final_hess, final_masks = custom_mtl_grad_hess(
            final_predictions,
            y_class_train,
            y_reg_train,
            alpha=parameters["alpha"],
            beta=parameters["beta"],
            margin=parameters["margin"],
            margin_buffer=parameters["margin_buffer"],
            loss_mode=LOSS_MODE,
        )
        final_snapshot = summarize_snapshot(
            outer_fold=outer_fold,
            stage="after_final_tree",
            tree_number=None,
            n_trees_in_prediction=n_trees,
            grad=final_grad,
            hess=final_hess,
            masks=final_masks,
            parameters=parameters,
        )
        snapshot_rows.append(final_snapshot)
        current_fold_snapshots.append(final_snapshot)

        fold_snapshot_frame = pd.DataFrame(current_fold_snapshots)

        leaf_report = stored_leaf_hessian_report(
            model=model,
            outer_fold=outer_fold,
        )
        leaf_reports.append(leaf_report)

        finite_leaf_weights = leaf_report.loc[
            np.isfinite(leaf_report["stored_leaf_sum_hessian"]),
            "stored_leaf_sum_hessian",
        ].to_numpy(dtype=float)
        positive_leaf_weights = finite_leaf_weights[
            finite_leaf_weights > ZERO_TOLERANCE
        ]

        tree_stump_status = (
            leaf_report.groupby("tree_number", sort=True)["stump_tree"]
            .first()
        )

        fold_summary_rows.append(
            {
                "outer_fold": outer_fold,
                "model_path": str(model_path),
                "preprocessor_path": str(preprocessor_path),
                "n_training_samples": int(np.sum(train_mask)),
                "n_trees": n_trees,
                "alpha": parameters["alpha"],
                "beta": parameters["beta"],
                "beta_source": parameters["beta_source"],
                "margin": parameters["margin"],
                "margin_buffer": parameters["margin_buffer"],
                "loss_mode": LOSS_MODE,
                "initial_score_assumed": float(INITIAL_SCORE),
                "final_zero_hessian_fraction_all": final_snapshot[
                    "zero_hessian_fraction_all"
                ],
                "final_zero_hessian_fraction_classification_only": (
                    final_snapshot[
                        "zero_hessian_fraction_classification_only"
                    ]
                ),
                "final_wrong_side_fraction": final_snapshot[
                    "wrong_side_fraction"
                ],
                "max_negative_hessian_count_in_any_snapshot": int(
                    fold_snapshot_frame["negative_hessian_count"].max()
                ),
                "max_nonfinite_hessian_count_in_any_snapshot": int(
                    fold_snapshot_frame["nonfinite_hessian_count"].max()
                ),
                "max_nonfinite_gradient_count_in_any_snapshot": int(
                    fold_snapshot_frame["nonfinite_gradient_count"].max()
                ),
                "max_unexpected_zero_hessian_count_in_any_snapshot": int(
                    fold_snapshot_frame[
                        "unexpected_zero_hessian_count"
                    ].max()
                ),
                "max_expected_zero_but_positive_count_in_any_snapshot": int(
                    fold_snapshot_frame[
                        "expected_zero_but_positive_hessian_count"
                    ].max()
                ),
                "minimum_positive_sample_hessian_across_snapshots": float(
                    fold_snapshot_frame["min_positive_hessian"].min()
                ),
                "maximum_zero_hessian_fraction_all_across_snapshots": float(
                    fold_snapshot_frame[
                        "zero_hessian_fraction_all"
                    ].max()
                ),
                "n_stored_leaves": int(len(leaf_report)),
                "stump_tree_fraction": float(tree_stump_status.mean()),
                "minimum_stored_leaf_sum_hessian": (
                    float(np.min(finite_leaf_weights))
                    if len(finite_leaf_weights)
                    else np.nan
                ),
                "minimum_positive_stored_leaf_sum_hessian": (
                    float(np.min(positive_leaf_weights))
                    if len(positive_leaf_weights)
                    else np.nan
                ),
                "negative_stored_leaf_hessian_count": int(
                    leaf_report[
                        "stored_leaf_hessian_is_negative"
                    ].sum()
                ),
                "zero_stored_leaf_hessian_count": int(
                    leaf_report[
                        "stored_leaf_hessian_is_zero"
                    ].sum()
                ),
                "nonfinite_stored_leaf_hessian_count": int(
                    leaf_report[
                        "stored_leaf_hessian_is_nonfinite"
                    ].sum()
                ),
                "missing_stored_leaf_hessian_count": int(
                    leaf_report[
                        "stored_leaf_hessian_is_missing"
                    ].sum()
                ),
                "suspicious_features_included": ";".join(
                    suspicious_included
                ),
            }
        )

        model_manifest_rows.append(
            {
                "outer_fold": outer_fold,
                "model_path": str(model_path.resolve()),
                "model_sha256": sha256_file(model_path),
                "preprocessor_path": str(preprocessor_path.resolve()),
                "preprocessor_sha256": sha256_file(preprocessor_path),
            }
        )

    snapshots = pd.DataFrame(snapshot_rows)
    leaves = pd.concat(leaf_reports, ignore_index=True)
    fold_summary = pd.DataFrame(fold_summary_rows).sort_values(
        "outer_fold"
    )

    fold_summary["sample_level_hessian_checks_pass"] = (
        fold_summary[
            "max_negative_hessian_count_in_any_snapshot"
        ].eq(0)
        & fold_summary[
            "max_nonfinite_hessian_count_in_any_snapshot"
        ].eq(0)
        & fold_summary[
            "max_unexpected_zero_hessian_count_in_any_snapshot"
        ].eq(0)
        & fold_summary[
            "max_expected_zero_but_positive_count_in_any_snapshot"
        ].eq(0)
        & fold_summary[
            "minimum_positive_sample_hessian_across_snapshots"
        ].gt(0.0)
    )

    fold_summary["stored_leaf_hessian_checks_pass"] = (
        fold_summary["negative_stored_leaf_hessian_count"].eq(0)
        & fold_summary["zero_stored_leaf_hessian_count"].eq(0)
        & fold_summary["nonfinite_stored_leaf_hessian_count"].eq(0)
        & fold_summary["minimum_positive_stored_leaf_sum_hessian"].gt(0.0)
    )

    fold_summary["hessian_checks_pass"] = (
        fold_summary["sample_level_hessian_checks_pass"]
        & fold_summary["stored_leaf_hessian_checks_pass"]
    )

    snapshots.to_csv(
        OUTPUT_DIR / "hessian_snapshot_diagnostics.csv",
        index=False,
    )
    leaves.to_csv(
        OUTPUT_DIR / "stored_leaf_hessian_diagnostics.csv",
        index=False,
    )
    fold_summary.to_csv(
        OUTPUT_DIR / "fold_hessian_summary.csv",
        index=False,
    )

    manifest = {
        "master_csv": str(MASTER_CSV.resolve()),
        "master_csv_sha256": sha256_file(MASTER_CSV),
        "model_dir": str(MODEL_DIR.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "loss_mode": LOSS_MODE,
        "model_prefix": MODEL_PREFIX,
        "model_glob": MODEL_GLOB,
        "requested_tree_numbers": list(REQUESTED_TREE_NUMBERS),
        "sample_level_scope": (
            "Selected pre-tree stages plus one after-final-tree stage; "
            "not every boosting iteration unless explicitly requested."
        ),
        "stored_leaf_scope": (
            "Every fitted tree and every leaf for which LightGBM stores "
            "leaf_weight."
        ),
        "initial_score_assumed": float(INITIAL_SCORE),
        "initial_score_verified_from_saved_training_data": False,
        "zero_tolerance": float(ZERO_TOLERANCE),
        "outer_folds_checked": sorted(
            fold_summary["outer_fold"].astype(int).tolist()
        ),
        "reconstruction_assumption": (
            "The master CSV, row order, fold assignments, preprocessing, "
            "loss parameters, and initial score match training."
        ),
        "models": model_manifest_rows,
        "versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    (OUTPUT_DIR / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nCompleted.")
    print(OUTPUT_DIR / "hessian_snapshot_diagnostics.csv")
    print(OUTPUT_DIR / "stored_leaf_hessian_diagnostics.csv")
    print(OUTPUT_DIR / "fold_hessian_summary.csv")
    print(OUTPUT_DIR / "diagnostic_manifest.json")

    display_columns = [
        "outer_fold",
        "n_trees",
        "hessian_checks_pass",
        "sample_level_hessian_checks_pass",
        "stored_leaf_hessian_checks_pass",
        "max_negative_hessian_count_in_any_snapshot",
        "max_nonfinite_hessian_count_in_any_snapshot",
        "max_unexpected_zero_hessian_count_in_any_snapshot",
        "minimum_positive_sample_hessian_across_snapshots",
        "final_zero_hessian_fraction_all",
        "minimum_stored_leaf_sum_hessian",
        "negative_stored_leaf_hessian_count",
        "zero_stored_leaf_hessian_count",
        "nonfinite_stored_leaf_hessian_count",
        "missing_stored_leaf_hessian_count",
    ]

    print("\nMinimal Hessian summary:")
    print(
        fold_summary[display_columns].to_string(
            index=False,
        )
    )

    if bool(fold_summary["hessian_checks_pass"].all()):
        print(
            "\nAll evaluated sample-level stages and all available "
            "stored leaf Hessian weights passed the configured checks."
        )
    else:
        failed_folds = fold_summary.loc[
            ~fold_summary["hessian_checks_pass"],
            "outer_fold",
        ].astype(int).tolist()
        print(
            "\nWARNING: One or more Hessian checks failed for outer "
            f"folds {failed_folds}. Inspect the detailed CSV files."
        )

    if int(fold_summary["missing_stored_leaf_hessian_count"].sum()) > 0:
        print(
            "Note: missing stored leaf weights occurred only for stump "
            "trees and were not classified as non-finite Hessians."
        )


if __name__ == "__main__":
    main()
