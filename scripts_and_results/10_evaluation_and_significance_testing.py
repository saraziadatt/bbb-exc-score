#!/usr/bin/env python3
"""
Paired model-comparison analysis for BBB model variants\.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MARGIN = -1.0

REGRESSION_METRICS = [
    "mae",
    "mse",
    "pearson_r",
    "r2",
    "np_mae",
    "wrong_side_mean_distance",
    "np_wrong_side_mean_distance",
    "penetrant_wrong_side_mean_distance",
]

CLASSIFICATION_METRICS = [
    "balanced_accuracy",
    "np_recall",
    "np_precision",
    "np_f1",
    "roc_auc",
    "pr_auc_nonpenetrant",
    "mcc",
    "cohens_kappa",
    "wrong_side_count",
    "np_wrong_side_count",
    "wrong_side_mean_distance",
    "np_wrong_side_mean_distance",
    "penetrant_wrong_side_mean_distance",
]

LOWER_IS_BETTER = {
    "mae",
    "mse",
    "np_mae",
    "wrong_side_count",
    "np_wrong_side_count",
    "wrong_side_mean_distance",
    "np_wrong_side_mean_distance",
    "penetrant_wrong_side_mean_distance",
}

HIGHER_IS_BETTER = {
    "pearson_r",
    "r2",
    "balanced_accuracy",
    "np_recall",
    "np_precision",
    "np_f1",
    "roc_auc",
    "pr_auc_nonpenetrant",
    "mcc",
    "cohens_kappa"
}

SUBSET_LABELS = {
    "measured_regression": "Measured compounds: regression",
    "measured_threshold": "Measured compounds: threshold-derived",
    "all_threshold": "All labelled compounds: threshold-derived",
    "classification_only_threshold": (
        "Classification-only compounds: threshold-derived"
    ),
}

# Preferred ordering for the established models. Any additional valid model
# names (for example, frozen ImageMol or UniMol variants) are appended
# alphabetically instead of being ignored.
PREFERRED_MODEL_ORDER = [
    "dummy_mean_regression",
    "dummy_prior_classification",
    "lightgbm_mse_regression",
    "lightgbm_mse_regression_threshold",
    "custom_mtl_original",
    "logbb_pred_shaker_retrained",
]

MODEL_DISPLAY = {
    "dummy_mean_regression": "Dummy regression mean",
    "dummy_prior_classification": "Dummy classification baseline",
    "lightgbm_mse_regression": "LightGBM MSE (native threshold)",
    "lightgbm_mse_regression_threshold": (
        "LightGBM MSE (altered threshold)"
    ),
    "custom_mtl_original": "Custom MTL LightGBM clean",
    "logbb_pred_shaker_retrained": (
        "Matched-retrained logBB_pred strategy (Shaker et al.)"
    ),
}

MODEL_ALIASES = {
    "custom_mtl_bab": "custom_mtl_original",
    "custom_mtl_original": "custom_mtl_original",
    "logbb_pred_shaker_retrained": "logbb_pred_shaker_retrained",
    "logbb_pred_matched_retraining": "logbb_pred_shaker_retrained",
    "logbb_pred_reproduction_fold_safe": "logbb_pred_shaker_retrained",
    "logbb_pred_reproduction": "logbb_pred_shaker_retrained",
}


def display_model_name(model_name: str) -> str:
    """Return a readable label without restricting which models are valid."""
    return MODEL_DISPLAY.get(
        model_name,
        model_name.replace("_", " ").strip().title(),
    )


def model_sort_key(model_name: str) -> tuple[str, int, str]:
    """Keep native and altered decision variants adjacent, native first."""
    for suffix, variant_order in [
        ("_native_threshold", 0),
        ("_altered_threshold", 1),
    ]:
        if model_name.endswith(suffix):
            return (model_name[: -len(suffix)], variant_order, model_name)
    return (model_name, 0, model_name)

def metric_suffix(metric: str) -> str:
    for prefix in ("np_", "penetrant_"):
        if metric.startswith(prefix):
            return metric[len(prefix) :]
    return metric


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    suffix = metric_suffix(metric)
    if suffix in LOWER_IS_BETTER:
        return "lower"
    if suffix in HIGHER_IS_BETTER:
        return "higher"
    raise KeyError(f"No direction defined for metric {metric!r}.")


def metric_improvement(
    focus_value: np.ndarray | float,
    comparator_value: np.ndarray | float,
    metric: str,
):
    """Positive means focus is better than comparator."""
    if metric_direction(metric) == "lower":
        return np.asarray(comparator_value) - np.asarray(focus_value)
    return np.asarray(focus_value) - np.asarray(comparator_value)


def holm_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(adjusted, index=values.index)

    ordered = valid[np.argsort(p[valid])]
    running_max = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return pd.Series(adjusted, index=values.index)


def stable_seed(base_seed: int, *values: str) -> int:
    digest = hashlib.sha256("::".join(values).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def make_stratified_bootstrap_matrix(
    folds: np.ndarray,
    classes: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return paired bootstrap indices stratified by outer fold × class."""
    folds = np.asarray(folds, dtype=int)
    classes = np.asarray(classes, dtype=int)
    if len(folds) == 0:
        return np.empty((n_bootstrap, 0), dtype=np.uint16)
    if len(folds) != len(classes):
        raise ValueError("folds and classes have different lengths.")

    labels = np.asarray(
        [
            f"fold_{fold}_class_{label}"
            for fold, label in zip(folds, classes)
        ],
        dtype=object,
    )
    strata = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
    ]
    if any(len(indices) == 0 for indices in strata):
        raise RuntimeError("An empty bootstrap stratum was generated.")

    rng = np.random.default_rng(seed)
    index_dtype = np.uint16 if len(folds) < 65535 else np.uint32
    sampled_parts = [
        rng.choice(
            indices,
            size=(n_bootstrap, len(indices)),
            replace=True,
        ).astype(index_dtype)
        for indices in strata
    ]
    matrix = np.concatenate(sampled_parts, axis=1)
    if matrix.shape != (n_bootstrap, len(folds)):
        raise RuntimeError(
            f"Unexpected bootstrap shape {matrix.shape}; "
            f"expected {(n_bootstrap, len(folds))}."
        )
    return matrix


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator == 0, np.nan, result)


def pearson_rows(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    p_centered = prediction - np.mean(
        prediction, axis=1, keepdims=True
    )
    numerator = np.sum(y_centered * p_centered, axis=1)
    denominator = np.sqrt(
        np.sum(y_centered**2, axis=1)
        * np.sum(p_centered**2, axis=1)
    )
    constant = (np.ptp(y, axis=1) == 0) | (np.ptp(prediction, axis=1) == 0)
    return np.where(constant, np.nan, safe_divide(numerator, denominator))


def regression_metrics_rows(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate only the four prespecified overall regression metrics."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - y
    absolute_error = np.abs(error)
    mse = np.mean(error**2, axis=1)
    y_mean = np.mean(y, axis=1, keepdims=True)
    sst = np.sum((y - y_mean) ** 2, axis=1)
    sse = np.sum(error**2, axis=1)
    return {
        "mae": np.mean(absolute_error, axis=1),
        "mse": mse,
        "pearson_r": pearson_rows(y, prediction),
        "r2": 1.0 - safe_divide(sse, sst),
    }


def observed_regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    values = regression_metrics_rows(
        np.asarray(y, dtype=float)[None, :],
        np.asarray(prediction, dtype=float)[None, :],
    )
    return {key: float(value[0]) for key, value in values.items()}


def wrong_side_distance_rows(
    *, y_class: np.ndarray, prediction: np.ndarray,
    predicted_class: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mean |predicted logBB - (-1)| on mistakes under actual model decisions.

    All arrays have shape (replicates, compounds). No mistakes, or missing
    logBB predictions anywhere in a replicate, makes that distance undefined.
    No compounds are dropped and class probabilities are never treated as logBB.
    """
    y_class = np.asarray(y_class, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    predicted_class = np.asarray(predicted_class, dtype=int)
    if not (y_class.shape == prediction.shape == predicted_class.shape):
        raise ValueError("Wrong-side distance arrays have different shapes.")
    true_np = y_class == 1
    incorrect = predicted_class != y_class
    distance = np.abs(prediction - MARGIN)
    complete = np.isfinite(prediction).all(axis=1)

    def masked_mean(mask):
        value = safe_divide(np.sum(np.where(mask, distance, 0.0), axis=1),
                            np.sum(mask, axis=1))
        return np.where(complete, value, np.nan)

    return {
        "wrong_side_mean_distance": masked_mean(incorrect),
        "np_wrong_side_mean_distance": masked_mean(incorrect & true_np),
        "penetrant_wrong_side_mean_distance": masked_mean(incorrect & ~true_np),
    }

def wrong_side_distance_counts(*, y_class, predicted_class) -> dict[str, int]:
    """Count actual mistakes, including mistakes without a logBB prediction."""
    y_class, predicted_class = np.asarray(y_class), np.asarray(predicted_class)
    if y_class.shape != predicted_class.shape:
        raise ValueError("Wrong-side count arrays have different shapes.")
    wrong = y_class != predicted_class
    return {
        "wrong_side_mean_distance": int(wrong.sum()),
        "np_wrong_side_mean_distance": int((wrong & (y_class == 1)).sum()),
        "penetrant_wrong_side_mean_distance": int((wrong & (y_class == 0)).sum()),
    }

def bootstrap_regression_metrics(
    *, y_true, y_class, prediction, predicted_class, bootstrap_index, chunk_size,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Calculate paired regression bootstrap samples in double precision."""
    observed_rows = regression_metrics_rows_with_groups(
        y_true=y_true, y_class=y_class, prediction_rows=prediction[None, :],
        predicted_class_rows=predicted_class[None, :],
    )
    observed = {k: float(v[0]) for k, v in observed_rows.items()}
    distributions = {m: np.empty(len(bootstrap_index), dtype=float)
                     for m in REGRESSION_METRICS}
    for start in range(0, len(bootstrap_index), chunk_size):
        index = bootstrap_index[start:start + chunk_size]
        y, c, pred = y_true[index], y_class[index], prediction[index]
        values = regression_metrics_rows(y, pred)
        values["np_mae"] = safe_divide(
            np.sum(np.where(c == 1, np.abs(pred - y), 0), axis=1),
            np.sum(c == 1, axis=1),
        )
        values.update(wrong_side_distance_rows(
            y_class=c, prediction=pred, predicted_class=predicted_class[index]))
        for metric, value in values.items():
            distributions[metric][start:start + len(index)] = value
    return observed, distributions

def average_precision_rows(
    score: np.ndarray,
    positive: np.ndarray,
) -> np.ndarray:
    """
    Tie-aware average precision for rows of samples.

    This reproduces the threshold-group definition used by average precision:
    each distinct score contributes its recall increment multiplied by precision
    at the end of that tied score group. A constant dummy score therefore gives
    AP equal to the positive prevalence.
    """
    order = np.argsort(-score, axis=1, kind="stable")
    sorted_score = np.take_along_axis(score, order, axis=1)
    sorted_positive = np.take_along_axis(
        positive.astype(np.int8), order, axis=1
    )
    cumulative_tp = np.cumsum(sorted_positive, axis=1).astype(float)
    ranks = np.arange(1, score.shape[1] + 1, dtype=float)[None, :]

    starts = np.ones_like(sorted_positive, dtype=bool)
    starts[:, 1:] = sorted_score[:, 1:] != sorted_score[:, :-1]
    ends = np.ones_like(sorted_positive, dtype=bool)
    ends[:, :-1] = sorted_score[:, :-1] != sorted_score[:, 1:]

    positions = np.arange(score.shape[1], dtype=int)[None, :]
    start_positions = np.maximum.accumulate(
        np.where(starts, positions, 0),
        axis=1,
    )
    previous_positions = np.maximum(start_positions - 1, 0)
    previous_tp = np.take_along_axis(
        cumulative_tp, previous_positions, axis=1
    )
    previous_tp = np.where(start_positions == 0, 0.0, previous_tp)
    group_positive = cumulative_tp - previous_tp

    n_positive = np.sum(positive, axis=1).astype(float)
    precision_at_end = safe_divide(cumulative_tp, ranks)
    recall_increment = safe_divide(group_positive, n_positive[:, None])
    return np.sum(
        np.where(ends, precision_at_end * recall_increment, 0.0),
        axis=1,
    )


def classification_metrics_rows(
    *,
    y_class: np.ndarray,
    predicted_class: np.ndarray,
    np_score: np.ndarray,
    logbb_prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Classification performance plus distance on actual misclassifications."""
    y_class = np.asarray(y_class, dtype=int)
    predicted_class = np.asarray(predicted_class, dtype=int)
    np_score = np.asarray(np_score, dtype=float)

    if not (y_class.shape == predicted_class.shape == np_score.shape):
        raise ValueError("Classification metric arrays have different shapes.")

    positive = y_class == 1
    negative = ~positive
    predicted_positive = predicted_class == 1
    predicted_negative = ~predicted_positive

    tp = np.sum(positive & predicted_positive, axis=1).astype(float)
    fn = np.sum(positive & predicted_negative, axis=1).astype(float)
    fp = np.sum(negative & predicted_positive, axis=1).astype(float)
    tn = np.sum(negative & predicted_negative, axis=1).astype(float)

    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    predicted_positive_count = tp + fp
    precision = np.where(
        predicted_positive_count == 0,
        0.0,
        safe_divide(tp, predicted_positive_count),
    )
    f1_denominator = precision + recall
    f1 = np.where(
        np.isfinite(recall) & (f1_denominator == 0),
        0.0,
        safe_divide(2.0 * precision * recall, f1_denominator),
    )
    balanced_accuracy = 0.5 * (recall + specificity)

    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = np.where(
        mcc_denominator == 0,
        0.0,
        safe_divide(mcc_numerator, mcc_denominator),
    )

    n_samples = tp + fn + fp + tn

    observed_agreement = safe_divide(
        tp + tn,
        n_samples,
    )

    expected_agreement = safe_divide(
        (tp + fn) * (tp + fp)
        + (tn + fp) * (tn + fn),
        n_samples**2,
    )

    cohens_kappa = safe_divide(
        observed_agreement - expected_agreement,
        1.0 - expected_agreement,
)

    score_ranks = rankdata(np_score, axis=1, method="average")
    n_positive = tp + fn
    n_negative = tn + fp
    positive_rank_sum = np.sum(score_ranks * positive, axis=1)
    roc_auc = safe_divide(
        positive_rank_sum - n_positive * (n_positive + 1.0) / 2.0,
        n_positive * n_negative,
    )
    average_precision = average_precision_rows(np_score, positive)

    result = {
        "balanced_accuracy": balanced_accuracy,
        "np_recall": recall,
        "np_precision": precision,
        "np_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc_nonpenetrant": average_precision,
        "mcc": mcc,
        "cohens_kappa": cohens_kappa,
        "wrong_side_count": fp + fn,
        "np_wrong_side_count": fn,
    }
    result.update(wrong_side_distance_rows(y_class=y_class,
        prediction=logbb_prediction, predicted_class=predicted_class))
    return result


def observed_classification_metrics(*, y_class, predicted_class, np_score, logbb_prediction):
    values = classification_metrics_rows(
        y_class=np.asarray(y_class)[None, :],
        predicted_class=np.asarray(predicted_class)[None, :],
        np_score=np.asarray(np_score)[None, :],
        logbb_prediction=np.asarray(logbb_prediction)[None, :],
    )
    return {key: float(value[0]) for key, value in values.items()}

def bootstrap_classification_metrics(
    *, y_class, predicted_class, np_score, logbb_prediction, bootstrap_index, chunk_size,
):
    observed = observed_classification_metrics(
        y_class=y_class, predicted_class=predicted_class, np_score=np_score,
        logbb_prediction=logbb_prediction)
    distributions = {metric: np.empty(len(bootstrap_index), dtype=float)
                     for metric in CLASSIFICATION_METRICS}
    for start in range(0, len(bootstrap_index), chunk_size):
        index = bootstrap_index[start:start + chunk_size]
        values = classification_metrics_rows(
            y_class=y_class[index], predicted_class=predicted_class[index],
            np_score=np_score[index], logbb_prediction=logbb_prediction[index])
        for metric, value in values.items():
            distributions[metric][start:start + len(index)] = value
    return observed, distributions

def bootstrap_difference_summary(
    values: np.ndarray,
    observed_difference: float,
    confidence: float,
) -> dict[str, float | int | bool]:
    """Summarize a paired bootstrap distribution of model A minus model B."""
    if not np.isfinite(observed_difference):
        return {
            "difference_a_minus_b": float(observed_difference),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_ci_excludes_zero": False,
            "n_bootstrap_valid": 0,
        }

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "difference_a_minus_b": float(observed_difference),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_ci_excludes_zero": False,
            "n_bootstrap_valid": 0,
        }

    alpha = 1.0 - confidence
    ci_lower, ci_upper = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return {
        "difference_a_minus_b": float(observed_difference),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_ci_excludes_zero": bool(
            (ci_lower > 0.0) or (ci_upper < 0.0)
        ),
        "n_bootstrap_valid": int(len(values)),
    }


def regression_metrics_rows_with_groups(
    *,
    y_true: np.ndarray,
    y_class: np.ndarray,
    prediction_rows: np.ndarray,
    predicted_class_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the prespecified regression metrics by permutation row."""
    y_true = np.asarray(y_true, dtype=float)
    y_class = np.asarray(y_class, dtype=int)
    prediction_rows = np.asarray(prediction_rows, dtype=float)
    y_rows = np.broadcast_to(y_true, prediction_rows.shape)

    metrics = regression_metrics_rows(y_rows, prediction_rows)
    np_mask = y_class == 1
    metrics["np_mae"] = (np.mean(
        np.abs(prediction_rows[:, np_mask] - y_rows[:, np_mask]), axis=1
    ) if np_mask.any() else np.full(len(prediction_rows), np.nan))
    metrics.update(
        wrong_side_distance_rows(
            y_class=np.broadcast_to(y_class, prediction_rows.shape),
            prediction=prediction_rows,
            predicted_class=predicted_class_rows,
        )
    )
    return metrics


def _initialize_permutation_counts(
    metrics: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {metric: 0 for metric in metrics},
        {metric: 0 for metric in metrics},
    )


def _update_two_sided_permutation_counts(
    *,
    permuted_a: dict[str, np.ndarray],
    permuted_b: dict[str, np.ndarray],
    observed_difference: dict[str, float],
    metrics: list[str],
    extreme_counts: dict[str, int],
    valid_counts: dict[str, int],
) -> None:
    """Accumulate |permuted difference| >= |observed difference| counts."""
    for metric in metrics:
        if not np.isfinite(observed_difference[metric]):
            continue
        difference = np.asarray(
            permuted_a[metric] - permuted_b[metric], dtype=float
        )
        valid = np.isfinite(difference)
        valid_counts[metric] += int(np.sum(valid))
        if np.any(valid):
            threshold = abs(float(observed_difference[metric]))
            extreme_counts[metric] += int(
                np.sum(np.abs(difference[valid]) >= threshold - 1e-15)
            )


def _finalize_two_sided_permutation_pvalues(
    *,
    metrics: list[str],
    extreme_counts: dict[str, int],
    valid_counts: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        n_valid = valid_counts[metric]
        p_value = (
            np.nan
            if n_valid == 0
            else (extreme_counts[metric] + 1.0) / (n_valid + 1.0)
        )
        results[metric] = {
            "permutation_p_two_sided": float(p_value),
            "n_valid_permutations": int(n_valid),
            "n_extreme_permutations": int(extreme_counts[metric]),
        }
    return results


def paired_regression_permutation_pvalues(
    *,
    y_true: np.ndarray,
    y_class: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    class_prediction_a: np.ndarray,
    class_prediction_b: np.ndarray,
    observed_difference: dict[str, float],
    metrics: list[str],
    n_permutations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    """Paired two-sided prediction-swap permutation test for regression."""
    prediction_a = np.asarray(prediction_a, dtype=float)
    prediction_b = np.asarray(prediction_b, dtype=float)
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Regression prediction arrays have different shapes.")

    rng = np.random.default_rng(seed)
    extreme_counts, valid_counts = _initialize_permutation_counts(metrics)
    n_samples = len(prediction_a)

    for start in range(0, n_permutations, chunk_size):
        current = min(chunk_size, n_permutations - start)
        swap = rng.integers(
            0, 2, size=(current, n_samples), dtype=np.int8
        ).astype(bool)
        permuted_prediction_a = np.where(
            swap, prediction_b[None, :], prediction_a[None, :]
        )
        permuted_prediction_b = np.where(
            swap, prediction_a[None, :], prediction_b[None, :]
        )
        permuted_a = regression_metrics_rows_with_groups(
            y_true=y_true,
            y_class=y_class,
            prediction_rows=permuted_prediction_a,
            predicted_class_rows=np.where(swap, class_prediction_b, class_prediction_a),
        )
        permuted_b = regression_metrics_rows_with_groups(
            y_true=y_true,
            y_class=y_class,
            prediction_rows=permuted_prediction_b,
            predicted_class_rows=np.where(swap, class_prediction_a, class_prediction_b),
        )
        _update_two_sided_permutation_counts(
            permuted_a=permuted_a,
            permuted_b=permuted_b,
            observed_difference=observed_difference,
            metrics=metrics,
            extreme_counts=extreme_counts,
            valid_counts=valid_counts,
        )

    return _finalize_two_sided_permutation_pvalues(
        metrics=metrics,
        extreme_counts=extreme_counts,
        valid_counts=valid_counts,
    )


def paired_classification_permutation_pvalues(
    *,
    y_class: np.ndarray,
    class_prediction_a: np.ndarray,
    class_prediction_b: np.ndarray,
    np_score_a: np.ndarray,
    np_score_b: np.ndarray,
    logbb_prediction_a: np.ndarray,
    logbb_prediction_b: np.ndarray,
    observed_difference: dict[str, float],
    metrics: list[str],
    n_permutations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    """Paired two-sided prediction-swap test for selected classification metrics."""
    y_class = np.asarray(y_class, dtype=int)
    class_prediction_a = np.asarray(class_prediction_a, dtype=int)
    class_prediction_b = np.asarray(class_prediction_b, dtype=int)
    np_score_a = np.asarray(np_score_a, dtype=float)
    np_score_b = np.asarray(np_score_b, dtype=float)

    expected_shape = y_class.shape
    for name, array in [
        ("class_prediction_a", class_prediction_a),
        ("class_prediction_b", class_prediction_b),
        ("np_score_a", np_score_a),
        ("np_score_b", np_score_b),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {expected_shape}."
            )

    rng = np.random.default_rng(seed)
    extreme_counts, valid_counts = _initialize_permutation_counts(metrics)
    n_samples = len(y_class)

    for start in range(0, n_permutations, chunk_size):
        current = min(chunk_size, n_permutations - start)
        swap = rng.integers(
            0, 2, size=(current, n_samples), dtype=np.int8
        ).astype(bool)

        permuted_class_a = np.where(
            swap, class_prediction_b[None, :], class_prediction_a[None, :]
        )
        permuted_class_b = np.where(
            swap, class_prediction_a[None, :], class_prediction_b[None, :]
        )
        permuted_score_a = np.where(
            swap, np_score_b[None, :], np_score_a[None, :]
        )
        permuted_score_b = np.where(
            swap, np_score_a[None, :], np_score_b[None, :]
        )

        y_rows = np.broadcast_to(y_class, permuted_class_a.shape)
        permuted_a = classification_metrics_rows(
            y_class=y_rows,
            predicted_class=permuted_class_a,
            np_score=permuted_score_a,
            logbb_prediction=np.where(swap, logbb_prediction_b, logbb_prediction_a),
        )
        permuted_b = classification_metrics_rows(
            y_class=y_rows,
            predicted_class=permuted_class_b,
            np_score=permuted_score_b,
            logbb_prediction=np.where(swap, logbb_prediction_a, logbb_prediction_b),
        )
        _update_two_sided_permutation_counts(
            permuted_a=permuted_a,
            permuted_b=permuted_b,
            observed_difference=observed_difference,
            metrics=metrics,
            extreme_counts=extreme_counts,
            valid_counts=valid_counts,
        )

    return _finalize_two_sided_permutation_pvalues(
        metrics=metrics,
        extreme_counts=extreme_counts,
        valid_counts=valid_counts,
    )


def parse_boolean_series(
    series: pd.Series,
    column_name: str,
) -> np.ndarray:
    """Parse a CSV boolean column without treating nonempty strings as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
        if not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError(
                f"{column_name!r} contains numeric values other than 0/1."
            )
        return numeric.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(
            f"{column_name!r} contains unrecognized boolean values: "
            f"{unknown[:20]}"
        )
    return normalized.map(mapping).to_numpy(bool)


def resolve_predictions_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


STANDARD_PREDICTION_HEADER = {
    "sample_id",
    "row_index",
    "smiles",
    "outer_fold",
    "true_logBB",
    "bbb_class",
    "has_logBB",
    "classification_only",
    "model",
    "model_task",
    "predicted_logBB",
    "score_nonpenetrant",
    "predicted_bbb_class",
    "classification_threshold",
}

PRETRAINED_PREDICTION_HEADER = {
    "smiles",
    "outer_fold",
    "logBB",
    "bbb_class",
    "model",
    "predicted_logBB",
    "nonpenetrant_score",
    "inner_tuned_threshold",
    "predicted_class_tuned",
    "native_threshold",
    "predicted_class_native",
}


OPTIONAL_PREDICTION_COLUMNS = {
    "absolute_error",
    "squared_error",
    "wrong_side",
    "_source_row",
    "_has_logbb",
    "has_logbb",
}


def read_prediction_csv(path: Path) -> pd.DataFrame:
    """Read only recognized prediction columns with stable dtype inference."""
    recognized = (
        STANDARD_PREDICTION_HEADER
        | PRETRAINED_PREDICTION_HEADER
        | OPTIONAL_PREDICTION_COLUMNS
    )
    return pd.read_csv(
        path,
        usecols=lambda column: column in recognized,
        low_memory=False,
    )


def _normalize_pretrained_rows(source: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Expand frozen-encoder rows into native and altered decisions."""
    base = source.copy()
    if "row_index" not in base or base["row_index"].isna().all():
        if "_source_row" in base:
            base["row_index"] = base["_source_row"]
        else:
            base["row_index"] = np.arange(len(base), dtype=int)
    if "sample_id" not in base or base["sample_id"].isna().all():
        base["sample_id"] = base["row_index"].map(lambda value: f"row_{value}")

    true_logbb = pd.to_numeric(base["logBB"], errors="coerce")
    if "has_logBB" in base and base["has_logBB"].notna().all():
        has_logbb = parse_boolean_series(base["has_logBB"], "has_logBB")
    elif "has_logbb" in base and base["has_logbb"].notna().all():
        has_logbb = parse_boolean_series(base["has_logbb"], "has_logbb")
    elif "_has_logbb" in base and base["_has_logbb"].notna().all():
        has_logbb = parse_boolean_series(base["_has_logbb"], "_has_logbb")
    else:
        has_logbb = true_logbb.notna().to_numpy()
    classification_only = ~np.asarray(has_logbb, dtype=bool)
    prediction = pd.to_numeric(base["predicted_logBB"], errors="raise")
    score = pd.to_numeric(base["nonpenetrant_score"], errors="raise")
    if not np.allclose(score, -prediction, atol=1e-10, rtol=1e-10):
        raise ValueError(
            f"{path} has nonpenetrant_score values inconsistent with "
            "-predicted_logBB."
        )

    normalized_variants: list[pd.DataFrame] = []
    for suffix, source_threshold, source_class in [
        ("native_threshold", "native_threshold", "predicted_class_native"),
        (
            "altered_threshold",
            "inner_tuned_threshold",
            "predicted_class_tuned",
        ),
    ]:
        variant = pd.DataFrame(
            {
                "sample_id": base["sample_id"].to_numpy(),
                "row_index": pd.to_numeric(
                    base["row_index"], errors="raise"
                ).to_numpy(int),
                "smiles": base["smiles"].astype(str).to_numpy(),
                "outer_fold": pd.to_numeric(
                    base["outer_fold"], errors="raise"
                ).to_numpy(int),
                "true_logBB": true_logbb.to_numpy(float),
                "bbb_class": pd.to_numeric(
                    base["bbb_class"], errors="raise"
                ).to_numpy(int),
                "has_logBB": np.asarray(has_logbb, dtype=bool),
                "classification_only": classification_only,
                "model": base["model"].astype(str) + f"_{suffix}",
                "model_task": "regression",
                "predicted_logBB": prediction.to_numpy(float),
                "score_nonpenetrant": score.to_numpy(float),
                "predicted_bbb_class": pd.to_numeric(
                    base[source_class], errors="raise"
                ).to_numpy(int),
                # Frozen-encoder files store score thresholds, whereas this
                # evaluator stores thresholds on the predicted-logBB scale.
                "classification_threshold": -pd.to_numeric(
                    base[source_threshold], errors="raise"
                ).to_numpy(float),
            }
        )
        measured = variant["has_logBB"].to_numpy(bool)
        error = (
            variant["predicted_logBB"].to_numpy(float)
            - variant["true_logBB"].to_numpy(float)
        )
        variant["absolute_error"] = np.where(measured, np.abs(error), np.nan)
        variant["squared_error"] = np.where(measured, error**2, np.nan)
        variant["wrong_side"] = (
            variant["predicted_bbb_class"].to_numpy(int)
            != variant["bbb_class"].to_numpy(int)
        )
        normalized_variants.append(variant)

    return pd.concat(normalized_variants, ignore_index=True)


def normalize_prediction_source(source: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalize standard, frozen-encoder, or mixed combined prediction rows."""
    columns = set(source.columns)
    has_standard_schema = STANDARD_PREDICTION_HEADER.issubset(columns)
    has_pretrained_schema = PRETRAINED_PREDICTION_HEADER.issubset(columns)
    if not has_standard_schema and not has_pretrained_schema:
        raise ValueError(
            f"Prediction file {path} does not use a supported raw prediction "
            "schema."
        )
    if has_standard_schema and not has_pretrained_schema:
        return source.copy()
    if has_pretrained_schema and not has_standard_schema:
        return _normalize_pretrained_rows(source, path)

    standard_identifier_columns = [
        "sample_id",
        "row_index",
        "model",
        "model_task",
        "score_nonpenetrant",
        "predicted_bbb_class",
        "classification_threshold",
    ]
    pretrained_identifier_columns = [
        "smiles",
        "outer_fold",
        "bbb_class",
        "model",
        "predicted_logBB",
        "nonpenetrant_score",
        "inner_tuned_threshold",
        "predicted_class_tuned",
        "native_threshold",
        "predicted_class_native",
    ]
    standard_mask = source[standard_identifier_columns].notna().all(axis=1)
    # Prefer an explicitly standardized row when both sets of source columns
    # are present; otherwise adapt the frozen-encoder row exactly once.
    pretrained_mask = (
        source[pretrained_identifier_columns].notna().all(axis=1)
        & ~standard_mask
    )
    unrecognized = ~(standard_mask | pretrained_mask)
    if unrecognized.any():
        examples = source.loc[
            unrecognized,
            [column for column in ["sample_id", "row_index", "model"] if column in source],
        ].head(20).to_dict("records")
        raise ValueError(
            f"Some rows in {path} are missing the identifiers required by "
            "both supported prediction schemas. Examples: "
            f"{examples}"
        )

    frames: list[pd.DataFrame] = []
    if standard_mask.any():
        frames.append(source.loc[standard_mask].copy())
    if pretrained_mask.any():
        frames.append(
            _normalize_pretrained_rows(
                source.loc[pretrained_mask].copy(), path
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def validated_integer_column(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Parse a required integer column and report bad rows explicitly."""
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    invalid = ~np.isfinite(numeric) | (numeric != np.floor(numeric))
    if column in {"bbb_class", "predicted_bbb_class"}:
        invalid |= ~np.isin(numeric, [0.0, 1.0])
    elif column == "row_index":
        invalid |= numeric < 0
    invalid |= numeric < -(2**63)
    invalid |= numeric >= 2**63
    if invalid.any():
        positions = np.flatnonzero(invalid)
        examples = []
        for position in positions[:20]:
            examples.append(
                {
                    "csv_data_row": int(position),
                    "model": str(frame.iloc[position].get("model", "")),
                    "sample_id": str(
                        frame.iloc[position].get("sample_id", "")
                    ),
                    column: frame.iloc[position][column],
                    "source_path": str(
                        frame.iloc[position].get("__source_path", "")
                    ),
                }
            )
        raise ValueError(
            f"Required integer column {column!r} contains missing, infinite, "
            f"non-integer, or out-of-range values. Examples: {examples}. "
            "No rows were discarded and no labels were imputed."
        )
    return numeric.astype(np.int64)


def load_raw_prediction_table(
    prediction_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """
    Load and combine long-format raw OOF prediction tables.

    The requested files may contain the baseline models, custom-MTL model, and
    independently trained LogBB_Pred comparator separately. For backward
    compatibility, a single pre-combined file is also accepted. Model names
    beyond the established baseline/custom/logBB set are retained.

    Expected columns:
        sample_id, row_index, smiles, outer_fold, true_logBB, bbb_class,
        has_logBB, classification_only, model, model_task,
        predicted_logBB, score_nonpenetrant, predicted_bbb_class,
        classification_threshold, absolute_error, squared_error, wrong_side.
    """
    if not prediction_paths:
        raise ValueError("At least one raw prediction file is required.")

    source_frames: list[pd.DataFrame] = []
    for prediction_path in prediction_paths:
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Raw prediction file not found: {prediction_path}"
            )
        source = normalize_prediction_source(
            read_prediction_csv(prediction_path), prediction_path
        )
        source["__source_path"] = str(prediction_path)
        source_frames.append(source)

    frame = pd.concat(source_frames, ignore_index=True, sort=False)
    required_columns = [
        "sample_id",
        "row_index",
        "smiles",
        "outer_fold",
        "true_logBB",
        "bbb_class",
        "has_logBB",
        "classification_only",
        "model",
        "model_task",
        "predicted_logBB",
        "score_nonpenetrant",
        "predicted_bbb_class",
        "classification_threshold",
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Raw prediction table is missing required columns: "
            f"{missing_columns}. Observed columns: {frame.columns.tolist()}"
        )

    frame = frame.copy()
    if frame["sample_id"].isna().any():
        raise ValueError("sample_id contains missing values.")
    frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
    if (frame["sample_id"] == "").any():
        raise ValueError("sample_id contains empty values.")

    # Recover genuinely absent row indices only from an existing one-to-one
    # sample mapping elsewhere in the same combined file. Frozen-encoder rows
    # normally use _source_row and are normalized before reaching this point.
    numeric_index = pd.to_numeric(frame["row_index"], errors="coerce")
    valid_index = (
        np.isfinite(numeric_index)
        & numeric_index.ge(0)
        & numeric_index.eq(np.floor(numeric_index))
        & numeric_index.lt(2**63)
    )
    known = frame.loc[valid_index, ["sample_id"]].copy()
    known["row_index"] = numeric_index.loc[valid_index]
    if (
        known.groupby("sample_id")["row_index"].nunique().gt(1).any()
        or known.groupby("row_index")["sample_id"].nunique().gt(1).any()
    ):
        raise ValueError(
            "Conflicting sample_id / row_index mapping across prediction rows. "
            "Rebuild the combined CSV from aligned original prediction files."
        )
    absent = frame["row_index"].isna()
    mapping = known.drop_duplicates("sample_id").set_index("sample_id")[
        "row_index"
    ]
    recovered = frame.loc[absent, "sample_id"].map(mapping)
    recoverable = recovered.notna()
    if recoverable.any():
        frame.loc[
            recovered.index[recoverable], "row_index"
        ] = recovered[recoverable]
        warnings.warn(
            f"Recovered {int(recoverable.sum())} missing row_index values "
            "using the existing one-to-one sample_id mapping.",
            RuntimeWarning,
        )

    for integer_column in [
        "row_index",
        "outer_fold",
        "bbb_class",
        "predicted_bbb_class",
    ]:
        frame[integer_column] = validated_integer_column(
            frame, integer_column
        )
    frame["classification_threshold"] = pd.to_numeric(
        frame["classification_threshold"], errors="raise"
    ).astype(float)
    frame["true_logBB"] = pd.to_numeric(
        frame["true_logBB"], errors="coerce"
    )
    frame["predicted_logBB"] = pd.to_numeric(
        frame["predicted_logBB"], errors="coerce"
    )
    frame["score_nonpenetrant"] = pd.to_numeric(
        frame["score_nonpenetrant"], errors="coerce"
    )
    frame["has_logBB"] = parse_boolean_series(
        frame["has_logBB"], "has_logBB"
    )
    frame["classification_only"] = parse_boolean_series(
        frame["classification_only"], "classification_only"
    )
    if set(frame["bbb_class"].unique()).difference({0, 1}):
        raise ValueError("bbb_class must contain only 0 and 1.")
    if set(frame["predicted_bbb_class"].unique()).difference({0, 1}):
        raise ValueError(
            "predicted_bbb_class must contain only 0 and 1."
        )

    original_model_name = frame["model"].astype(str)
    frame["__source_model_name"] = original_model_name
    frame["model"] = original_model_name.replace(MODEL_ALIASES)

    observed_models = sorted(
        frame["model"].dropna().astype(str).unique().tolist()
    )
    if not observed_models:
        raise ValueError("No non-empty model names were found.")
    model_order = [
        model for model in PREFERRED_MODEL_ORDER if model in observed_models
    ]
    model_order.extend(
        sorted(
            set(observed_models).difference(model_order),
            key=model_sort_key,
        )
    )

    working = frame.copy()

    duplicate_mask = working.duplicated(
        ["model", "row_index"], keep=False
    )
    if duplicate_mask.any():
        examples = (
            working.loc[
                duplicate_mask,
                ["model", "row_index", "__source_path"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Expected exactly one row per model × row_index; rows are never "
            "collapsed automatically. The native-threshold and altered-"
            "threshold LightGBM variants must use the distinct model names "
            "'lightgbm_mse_regression' and "
            "'lightgbm_mse_regression_threshold'. Duplicates within either "
            "name mean an input was appended more than once. Supply each "
            "model source exactly once. Duplicate examples: "
            f"{examples}"
        )

    # Construct canonical rows from the first requested model.
    reference_name = (
        "dummy_mean_regression"
        if "dummy_mean_regression" in model_order
        else model_order[0]
    )
    reference = (
        working[working["model"] == reference_name]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    if reference.empty:
        raise ValueError(f"No rows found for {reference_name!r}.")
    if len(reference) != reference["row_index"].nunique():
        raise ValueError("Reference model row_index is not unique.")
    expected_row_index = np.arange(len(reference), dtype=int)
    actual_row_index = reference["row_index"].to_numpy(int)
    if not np.array_equal(actual_row_index, expected_row_index):
        raise ValueError(
            "row_index must be a complete zero-based sequence after sorting. "
            f"First observed values: {actual_row_index[:20].tolist()}"
        )
    if reference["sample_id"].duplicated().any():
        duplicate_ids = (
            reference.loc[
                reference["sample_id"].duplicated(keep=False), "sample_id"
            ]
            .head(20)
            .tolist()
        )
        raise ValueError(
            "sample_id must be unique but contains duplicates. First "
            f"duplicate values: {duplicate_ids}"
        )

    measured = reference["has_logBB"].to_numpy(bool)
    class_only = reference["classification_only"].to_numpy(bool)
    if not np.array_equal(class_only, ~measured):
        mismatch = np.flatnonzero(class_only != ~measured)
        raise ValueError(
            "classification_only is not the inverse of has_logBB. "
            f"First mismatches: {mismatch[:20].tolist()}"
        )
    true_logbb = reference["true_logBB"].to_numpy(float)
    if not np.isfinite(true_logbb[measured]).all():
        raise ValueError(
            "Measured rows contain missing or nonfinite true_logBB values."
        )
    measured_class_from_logbb = (
        true_logbb[measured] < MARGIN
    ).astype(int)
    measured_stored_class = reference.loc[
        measured, "bbb_class"
    ].to_numpy(int)
    if not np.array_equal(
        measured_class_from_logbb, measured_stored_class
    ):
        measured_positions = np.flatnonzero(measured)
        mismatch = measured_positions[
            measured_class_from_logbb != measured_stored_class
        ]
        examples = [
            {
                "row_index": int(actual_row_index[index]),
                "true_logBB": float(true_logbb[index]),
                "bbb_class": int(reference.loc[index, "bbb_class"]),
                "expected_bbb_class": int(true_logbb[index] < MARGIN),
            }
            for index in mismatch[:20]
        ]
        raise ValueError(
            "Measured bbb_class values are inconsistent with "
            f"true_logBB < {MARGIN}. Examples: {examples}"
        )
    if np.isfinite(true_logbb[class_only]).any():
        warnings.warn(
            "Some classification-only rows contain true_logBB. They remain "
            "excluded from regression metrics because has_logBB is False.",
            RuntimeWarning,
        )

    canonical = pd.DataFrame(
        {
            "row_position": actual_row_index,
            "sample_id": reference["sample_id"].to_numpy(),
            "smiles": reference["smiles"].astype(str).to_numpy(),
            "outer_fold": reference["outer_fold"].to_numpy(int),
            "logBB": true_logbb,
            "bbb_class": reference["bbb_class"].to_numpy(int),
            "measured": measured,
            "classification_only": class_only,
        }
    )

    selected_models: dict[str, dict[str, Any]] = {}
    audit_rows = []
    truth_columns = [
        "sample_id",
        "row_index",
        "smiles",
        "outer_fold",
        "true_logBB",
        "bbb_class",
        "has_logBB",
        "classification_only",
    ]

    for model_name in model_order:
        group = (
            working[working["model"] == model_name]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        if len(group) != len(canonical):
            raise ValueError(
                f"{model_name!r} has {len(group)} rows; expected "
                f"{len(canonical)} complete OOF rows."
            )

        # Verify exact pairing and truth/fold consistency.
        for column in truth_columns:
            observed = group[column]
            expected = reference[column]
            if column == "true_logBB":
                equal = np.allclose(
                    observed.to_numpy(float),
                    expected.to_numpy(float),
                    equal_nan=True,
                    atol=1e-12,
                    rtol=0,
                )
            else:
                equal = np.array_equal(
                    observed.astype(str).to_numpy(),
                    expected.astype(str).to_numpy(),
                )
            if not equal:
                raise ValueError(
                    f"{model_name!r} does not match the canonical rows in "
                    f"column {column!r}."
                )

        task_values = sorted(
            group["model_task"].dropna().astype(str).str.lower().unique()
        )
        if len(task_values) != 1 or task_values[0] not in {
            "classification",
            "regression",
        }:
            raise ValueError(
                f"{model_name!r} has model_task values {task_values}; "
                "expected exactly one of 'classification' or 'regression'."
            )
        expected_task = task_values[0]

        class_prediction = group[
            "predicted_bbb_class"
        ].to_numpy(int)
        np_score = group["score_nonpenetrant"].to_numpy(float)
        if not np.isfinite(np_score).all():
            raise ValueError(
                f"{model_name!r} has missing/nonfinite "
                "score_nonpenetrant values."
            )

        threshold_values = group[
            "classification_threshold"
        ].to_numpy(float)
        if not np.isfinite(threshold_values).all():
            nonfinite = np.flatnonzero(~np.isfinite(threshold_values))
            raise ValueError(
                f"{model_name!r} has missing/nonfinite classification "
                "thresholds. First row positions: "
                f"{nonfinite[:20].tolist()}"
            )
        uses_fold_specific_threshold = not np.allclose(
            threshold_values,
            threshold_values[0],
            atol=1e-12,
            rtol=0,
        )
        if uses_fold_specific_threshold:
            thresholds_by_fold: dict[int, float] = {}
            for fold, fold_group in group.groupby("outer_fold", sort=True):
                fold_thresholds = fold_group[
                    "classification_threshold"
                ].to_numpy(float)
                if not np.allclose(
                    fold_thresholds,
                    fold_thresholds[0],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        f"{model_name!r} uses multiple thresholds "
                        f"within outer fold {fold}."
                    )
                thresholds_by_fold[int(fold)] = float(fold_thresholds[0])
            threshold_audit: float | str = json.dumps(
                thresholds_by_fold, sort_keys=True
            )
        else:
            threshold_audit = float(threshold_values[0])

        if expected_task == "regression":
            regression_prediction = group[
                "predicted_logBB"
            ].to_numpy(float)
            if not np.isfinite(regression_prediction).all():
                raise ValueError(
                    f"{model_name!r} lacks complete predicted_logBB values."
                )
            implied_class = (
                regression_prediction < threshold_values
            ).astype(int)
            if not np.array_equal(implied_class, class_prediction):
                implied_class_inclusive = (
                    regression_prediction <= threshold_values
                ).astype(int)
                if not np.array_equal(
                    implied_class_inclusive, class_prediction
                ):
                    mismatch = np.flatnonzero(
                        implied_class != class_prediction
                    )
                    raise ValueError(
                        f"{model_name!r} predicted_bbb_class disagrees with "
                        "thresholding predicted_logBB at "
                        "classification_threshold. First mismatches: "
                        f"{mismatch[:20].tolist()}"
                    )

            # score_nonpenetrant is expected to increase as logBB decreases.
            if not np.allclose(
                np_score,
                -regression_prediction,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"{model_name!r} score_nonpenetrant is not "
                    "-predicted_logBB."
                )
            distance_prediction = regression_prediction
            threshold_kind = (
                "native"
                if np.allclose(
                    threshold_values, MARGIN, atol=1e-12, rtol=0
                )
                else "altered"
            )

        else:
            # Classifier scores are expected to increase toward class 1.
            # An exact tie is assigned to class 0, matching sklearn's sorted-
            # class behavior for the prior dummy classifier.
            implied_class = (np_score > threshold_values).astype(int)
            if not np.array_equal(implied_class, class_prediction):
                mismatch = np.flatnonzero(
                    implied_class != class_prediction
                )
                raise ValueError(
                    f"{model_name!r} predicted_bbb_class disagrees with "
                    "score_nonpenetrant > classification_threshold. First "
                    f"mismatches: {mismatch[:20].tolist()}"
                )

            regression_prediction = None
            distance_prediction = None
            threshold_kind = (
                "fixed" if not uses_fold_specific_threshold else "altered"
            )

        expected_wrong = (
            class_prediction != canonical["bbb_class"].to_numpy(int)
        )
        if (
            "wrong_side" in group.columns
            and group["wrong_side"].notna().any()
        ):
            if not group["wrong_side"].notna().all():
                missing = np.flatnonzero(
                    group["wrong_side"].isna().to_numpy()
                )
                raise ValueError(
                    f"{model_name!r} has a partially populated wrong_side "
                    f"column. First missing positions: {missing[:20].tolist()}"
                )
            stored_wrong = parse_boolean_series(
                group["wrong_side"], f"{model_name} wrong_side"
            )
            if not np.array_equal(stored_wrong, expected_wrong):
                mismatch = np.flatnonzero(
                    stored_wrong != expected_wrong
                )
                raise ValueError(
                    f"{model_name!r} stored wrong_side is inconsistent with "
                    "predicted_bbb_class and bbb_class. First mismatches: "
                    f"{mismatch[:20].tolist()}"
                )

        if (
            expected_task == "regression"
            and "absolute_error" in group.columns
            and group["absolute_error"].notna().any()
        ):
            if not group.loc[measured, "absolute_error"].notna().all():
                missing = np.flatnonzero(
                    group.loc[measured, "absolute_error"].isna().to_numpy()
                )
                raise ValueError(
                    f"{model_name!r} has missing absolute_error values for "
                    "measured compounds. First measured-subset positions: "
                    f"{missing[:20].tolist()}"
                )
            stored_absolute = pd.to_numeric(
                group["absolute_error"], errors="raise"
            ).to_numpy(float)
            expected_absolute = np.abs(
                regression_prediction[measured] - true_logbb[measured]
            )
            if not np.allclose(
                stored_absolute[measured],
                expected_absolute,
                atol=1e-7,
                rtol=1e-6,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{model_name!r} stored absolute_error is inconsistent "
                    "with true_logBB and predicted_logBB."
                )

        if (
            expected_task == "regression"
            and "squared_error" in group.columns
            and group["squared_error"].notna().any()
        ):
            if not group.loc[measured, "squared_error"].notna().all():
                missing = np.flatnonzero(
                    group.loc[measured, "squared_error"].isna().to_numpy()
                )
                raise ValueError(
                    f"{model_name!r} has missing squared_error values for "
                    "measured compounds. First measured-subset positions: "
                    f"{missing[:20].tolist()}"
                )
            stored_squared = pd.to_numeric(
                group["squared_error"], errors="raise"
            ).to_numpy(float)
            expected_squared = (
                regression_prediction[measured] - true_logbb[measured]
            ) ** 2
            if not np.allclose(
                stored_squared[measured],
                expected_squared,
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{model_name!r} stored squared_error is inconsistent "
                    "with true_logBB and predicted_logBB."
                )

        source_paths = sorted(group["__source_path"].astype(str).unique())
        source_model_names = sorted(
            group["__source_model_name"].astype(str).unique()
        )
        if len(source_paths) != 1:
            raise ValueError(
                f"{model_name!r} appears in multiple prediction files: "
                f"{source_paths}. Supply exactly one source for each model."
            )

        selected_models[model_name] = {
            "model": model_name,
            "display_name": display_model_name(model_name),
            "regression_prediction": regression_prediction,
            "continuous_prediction": regression_prediction,
            "class_prediction": class_prediction,
            "np_score": np_score,
            "distance_prediction": distance_prediction,
            "threshold_kind": threshold_kind,
            "regression_metrics_represented_by": None,
            "score_metrics_represented_by": None,
            "source_path": source_paths[0],
            "source_model_name": ",".join(source_model_names),
            "score_orientation": (
                "-predicted_logBB"
                if expected_task == "regression"
                else "score_nonpenetrant"
            ),
        }
        audit_rows.append(
            {
                "model": model_name,
                "display_name": display_model_name(model_name),
                "source_path": source_paths[0],
                "source_model_name": ",".join(source_model_names),
                "model_task": expected_task,
                "n_rows": len(group),
                "n_measured": int(measured.sum()),
                "n_classification_only": int(class_only.sum()),
                "regression_available": (
                    regression_prediction is not None
                ),
                "threshold_available": True,
                "wrong_side_distance_available": (
                    distance_prediction is not None
                ),
                "classification_threshold": threshold_audit,
                "n_unique_classification_thresholds": int(
                    np.unique(threshold_values).size
                ),
                "threshold_kind": threshold_kind,
                "regression_metrics_represented_by": None,
                "score_metrics_represented_by": None,
                "status": "loaded_and_validated",
            }
        )

    audit = pd.DataFrame(audit_rows)

    # Keep every named variant, even when its continuous predictions are identical.
    # Each exact pair needs its own CI, permutation test, and Holm family.
    return canonical, selected_models, audit


def metric_catalog() -> pd.DataFrame:
    rows = []
    for metric in REGRESSION_METRICS:
        rows.append(
            {
                "subset": "measured_regression",
                "metric": metric,
                "metric_group": (
                    "measured_nonpenetrant"
                    if metric == "np_mae"
                    else "wrong_side_margin_distance"
                    if "wrong_side_mean_distance" in metric
                    else "overall"
                ),
                "direction": metric_direction(metric),
                "positive_improvement_definition": (
                    "comparator - focus"
                    if metric_direction(metric) == "lower"
                    else "focus - comparator"
                ),
                "dummy_classifier_available": False,
            }
        )

    for subset in [
        "measured_threshold",
        "all_threshold",
        "classification_only_threshold",
    ]:
        for metric in CLASSIFICATION_METRICS:
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "metric_group": "threshold_derived",
                    "direction": metric_direction(metric),
                    "positive_improvement_definition": (
                        "comparator - focus"
                        if metric_direction(metric) == "lower"
                        else "focus - comparator"
                    ),
                    "dummy_classifier_available": "wrong_side_mean_distance" not in metric,
                }
            )
    return pd.DataFrame(rows)


def add_pair_dataset_holm_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm across metrics within each unique model pair and subset."""
    result = frame.copy()
    result["permutation_p_two_sided_holm_pair_dataset"] = (
        result.groupby(
            ["model_a", "model_b", "subset"],
            group_keys=False,
        )["permutation_p_two_sided"]
        .transform(lambda values: holm_adjust(values).to_numpy())
    )
    result["holm_family_size"] = result.groupby(
        ["model_a", "model_b", "subset"]
    )["permutation_p_two_sided"].transform("count")
    result["significant_permutation_holm_0.05"] = (
        result["permutation_p_two_sided_holm_pair_dataset"] < 0.05
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--input-format", choices=["oof", "external"], default="oof",
                        help="oof: combined outer-fold predictions; external: 01_all_external_predictions.csv, one evaluated prediction per compound/model.")
    parser.add_argument(
        "--predictions-file",
        required=True,
        help=(
            "The combined long-format all_models_predictions.csv file. "
            "Relative paths are resolved under --root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./all_models/paired_permutation_results",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of paired prediction-swap permutations.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of paired stratified bootstrap replicates for CIs only.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()

    if args.permutations < 1:
        parser.error("--permutations must be at least 1.")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be strictly between 0 and 1.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1.")
    return args


def load_external_prediction_table(path):
    """Read already evaluated external predictions; never refit or retune.

    One row per compound/model is required. All models must cover exactly the
    same compounds, with matching true labels. No intersection/filtering is used.
    Class 1 and larger score_np both mean non-penetrant. Optional measured labels
    must be experimental logBB, not predictions or label-derived pseudo-values.
    """
    aliases = {
        "sample_id": ["external_compound_id", "sample_id"],
        "bbb_class": ["Class", "bbb_class"],
        "class_prediction": ["pred_np", "predicted_bbb_class"],
        "np_score": ["score_np", "score_nonpenetrant"],
        "logBB": ["true_logBB", "true_logbb", "experimental_logBB", "logBB"],
        "predicted_logBB": ["predicted_logbb", "predicted_logBB"],
    }
    columns = set(sum(aliases.values(), []) + ["model", "has_logBB", "has_logbb", "classification_only"])
    raw = pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)
    if raw.empty or "model" not in raw or raw.model.isna().any():
        raise ValueError("External predictions require nonempty rows and model IDs.")
    frame = raw[["model"]].copy()
    for target, choices in aliases.items():
        found = [c for c in choices if c in raw]
        if not found:
            if target in {"logBB", "predicted_logBB"}:
                frame[target] = np.nan
                continue
            raise ValueError(f"External predictions need one of {choices}.")
        # Fail on ambiguous conflicting copies instead of silently preferring one.
        for other in found[1:]:
            left, right = raw[found[0]], raw[other]
            equal = (left.fillna("<NA>").astype(str) == right.fillna("<NA>").astype(str))
            if not equal.all():
                try:
                    equal = np.isclose(pd.to_numeric(left), pd.to_numeric(right), equal_nan=True,
                                       atol=1e-12, rtol=0)
                except (ValueError, TypeError):
                    pass
                if not np.all(equal):
                    raise ValueError(f"Conflicting external columns {found[0]} and {other}.")
        frame[target] = raw[found[0]]
    if frame.sample_id.isna().any():
        raise ValueError("External compound IDs cannot be missing.")
    frame["sample_id"] = frame.sample_id.astype(str).str.strip()
    frame["model"] = frame.model.astype(str).str.strip().replace(MODEL_ALIASES)
    if frame.sample_id.eq("").any() or frame.model.eq("").any():
        raise ValueError("External compound/model IDs cannot be blank.")
    if frame.duplicated(["model", "sample_id"]).any():
        raise ValueError("External input must have one evaluated row per model/compound; aggregate fold predictions upstream.")
    for c in ["bbb_class", "class_prediction", "np_score", "logBB", "predicted_logBB"]:
        frame[c] = pd.to_numeric(frame[c], errors="raise")
        if np.isinf(frame[c]).any():
            raise ValueError(f"External {c} contains infinite values.")
    for c in ["bbb_class", "class_prediction"]:
        if not frame[c].isin([0, 1]).all():
            raise ValueError(f"External {c} must be complete 0/1 labels; 1 = NP.")
    if frame.np_score.isna().any():
        raise ValueError("External score_np must be complete and increase toward NP.")
    measured = frame.logBB.notna()
    for flag in ["has_logBB", "has_logbb", "classification_only"]:
        if flag in raw:
            actual = parse_boolean_series(raw[flag], flag)
            expected = (~measured if flag == "classification_only" else measured).to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f"External {flag} disagrees with available experimental logBB labels.")
    if not np.array_equal((frame.loc[measured, "logBB"] < MARGIN).astype(int),
                          frame.loc[measured, "bbb_class"].astype(int)):
        raise ValueError("External measured logBB and BBB class disagree at -1.")
    names = set(frame.model)
    order = [m for m in PREFERRED_MODEL_ORDER if m in names]
    order += sorted(names - set(order), key=model_sort_key)
    first = frame[frame.model == order[0]].sort_values("sample_id")
    canonical = first[["sample_id", "bbb_class", "logBB"]].reset_index(drop=True)
    canonical["measured"] = canonical.logBB.notna()
    # A single resampling group means class-stratified external bootstrap.
    # This is not an invented outer fold and is labelled accordingly in metadata.
    canonical["outer_fold"] = 0
    selected, audit = {}, []
    for model in order:
        group = frame[frame.model == model].sort_values("sample_id").reset_index(drop=True)
        if group.sample_id.tolist() != canonical.sample_id.tolist():
            raise ValueError(f"External {model}: compound coverage differs across models; no rows were dropped.")
        if not np.array_equal(group.bbb_class, canonical.bbb_class) or not np.allclose(
                group.logBB, canonical.logBB, equal_nan=True, atol=1e-12, rtol=0):
            raise ValueError(f"External {model}: true labels disagree across models.")
        pred = group.predicted_logBB.to_numpy(float)
        if np.isnan(pred).any() and not np.isnan(pred).all():
            raise ValueError(f"External {model}: partially missing logBB predictions; no rows were dropped.")
        prediction = None if np.isnan(pred).all() else pred
        selected[model] = {
            "regression_prediction": prediction, "continuous_prediction": prediction,
            "distance_prediction": prediction,
            "class_prediction": group.class_prediction.to_numpy(int),
            "np_score": group.np_score.to_numpy(float),
            "regression_metrics_represented_by": None, "score_metrics_represented_by": None,
        }
        audit.append({"model": model, "source_path": str(path), "n_rows": len(group),
                      "n_measured": int(canonical.measured.sum()),
                      "n_classification_only": int((~canonical.measured).sum()),
                      "regression_available": prediction is not None and canonical.measured.any(),
                      "wrong_side_distance_available": prediction is not None,
                      "regression_metrics_represented_by": None, "score_metrics_represented_by": None,
                      "status": "loaded_and_validated_external_decisions"})
    return canonical, selected, pd.DataFrame(audit)


def undefined_reason(metric, value, y_true, y_class, prediction, decisions):
    """Explain undefined statistics without turning undefined quantities into zero."""
    if np.isfinite(value):
        return "available"
    if (metric in REGRESSION_METRICS) and not np.isfinite(prediction).all():
        return "no_predicted_logBB"
    if "wrong_side_mean_distance" in metric:
        count = wrong_side_distance_counts(y_class=y_class, predicted_class=decisions)[metric]
        return "no_incorrect_compounds_in_conditioning_group" if count == 0 else "undefined_distance"
    if metric == "pearson_r":
        return "constant_true_or_predicted_logBB"
    if metric == "r2":
        return "constant_true_logBB"
    if metric in {"np_mae", "np_recall", "np_f1", "pr_auc_nonpenetrant"} and not (y_class == 1).any():
        return "no_true_nonpenetrants"
    if metric in {"balanced_accuracy", "roc_auc"}:
        return "single_true_class"
    if metric == "cohens_kappa":
        return "zero_kappa_denominator"
    return "undefined_metric"


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = resolve_predictions_path(root, args.predictions_file)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    if args.input_format == "external":
        canonical, selected_models, audit = load_external_prediction_table(prediction_path)
    else:
        canonical, selected_models, audit = load_raw_prediction_table([prediction_path])
    model_order = list(selected_models)
    if len(model_order) < 2:
        raise ValueError("At least two model variants are required for all-pair comparisons.")
    audit.to_csv(output_dir / "selected_model_input_audit.csv", index=False)
    metric_catalog().to_csv(output_dir / "metric_catalog.csv", index=False)
    y_reg = canonical.logBB.to_numpy(float)
    y_class = canonical.bbb_class.to_numpy(int)
    folds = canonical.outer_fold.to_numpy(int)
    measured = canonical.measured.to_numpy(bool)
    all_masks = {
        "measured_regression": measured,
        "measured_threshold": measured,
        "all_threshold": np.ones(len(canonical), dtype=bool),
        "classification_only_threshold": ~measured,
    }
    # External manuscript tables need the full classification set and measured regression.
    # Do not duplicate the full external set as a second classification-only table.
    selected_subsets = (list(all_masks) if args.input_format == "oof"
                        else ["all_threshold", "measured_regression"])
    masks = {s: all_masks[s] for s in selected_subsets if all_masks[s].any()}
    pd.DataFrame([{"subset": s, "n_samples": int(all_masks[s].sum()),
                   "status": "evaluated" if s in masks else "no_compounds_in_subset"}
                  for s in selected_subsets]).to_csv(output_dir / "subset_input_audit.csv", index=False)
    resampling_name = {"measured_regression": "measured", "measured_threshold": "measured",
                       "all_threshold": "all", "classification_only_threshold": "classification_only"}
    bootstrap_indices = {}
    for subset, mask in masks.items():
        name = resampling_name[subset]
        if name not in bootstrap_indices:
            bootstrap_indices[name] = make_stratified_bootstrap_matrix(
                folds[mask], y_class[mask], args.bootstrap,
                stable_seed(args.seed, "bootstrap", name))
    values = {m: {} for m in model_order}
    for model, record in selected_models.items():
        # NaN represents a classifier that does not produce logBB; it is never imputed.
        prediction = record["distance_prediction"]
        record["analysis_logbb"] = (np.full(len(canonical), np.nan)
                                    if prediction is None else np.asarray(prediction, dtype=float))
        for subset, mask in masks.items():
            pred = record["analysis_logbb"][mask]
            decision = record["class_prediction"][mask]
            shared = dict(y_class=y_class[mask], predicted_class=decision,
                          bootstrap_index=bootstrap_indices[resampling_name[subset]],
                          chunk_size=args.chunk_size)
            if subset == "measured_regression":
                observed, bootstrap = bootstrap_regression_metrics(
                    y_true=y_reg[mask], prediction=pred, **shared)
            else:
                observed, bootstrap = bootstrap_classification_metrics(
                    np_score=record["np_score"][mask], logbb_prediction=pred, **shared)
            values[model][subset] = {
                "observed": observed, "bootstrap": bootstrap,
                "counts": wrong_side_distance_counts(y_class=y_class[mask], predicted_class=decision),
                "status": {metric: undefined_reason(metric, value, y_reg[mask], y_class[mask], pred, decision)
                           for metric, value in observed.items()},
            }
        print(f"Bootstrapped {model}", flush=True)
    del bootstrap_indices

    pooled_rows = []
    for model in model_order:
        for subset, result in values[model].items():
            for metric, value in result["observed"].items():
                summary = bootstrap_difference_summary(result["bootstrap"][metric], value, args.confidence)
                pooled_rows.append({
                    "model": model, "display_name": display_model_name(model),
                    "subset": subset, "subset_label": SUBSET_LABELS[subset], "metric": metric,
                    "n_samples": int(masks[subset].sum()), "value": value,
                    "metric_status": result["status"][metric],
                    "estimate_ci_lower": summary["ci_lower"], "estimate_ci_upper": summary["ci_upper"],
                    "n_bootstrap_valid": summary["n_bootstrap_valid"],
                    "n_incorrect_for_distance_mean": result["counts"].get(metric, np.nan),
                    "direction": metric_direction(metric),
                })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output_dir / "pooled_metrics_selected_models_long.csv", index=False)
    pooled.pivot(index=["model", "display_name"], columns=["subset", "metric"], values="value").to_csv(
        output_dir / "pooled_metrics_selected_models_wide.csv")

    pairwise_rows = []
    all_pairs = list(itertools.combinations(model_order, 2))
    for pair_number, (model_a, model_b) in enumerate(all_pairs, 1):
        a, b = selected_models[model_a], selected_models[model_b]
        for subset, mask in masks.items():
            va, vb = values[model_a][subset], values[model_b][subset]
            metrics = REGRESSION_METRICS if subset == "measured_regression" else CLASSIFICATION_METRICS
            differences = {m: float(va["observed"][m] - vb["observed"][m]) for m in metrics}
            kwargs = dict(y_class=y_class[mask], class_prediction_a=a["class_prediction"][mask],
                          class_prediction_b=b["class_prediction"][mask], observed_difference=differences,
                          metrics=metrics, n_permutations=args.permutations,
                          seed=stable_seed(args.seed, "permutation", model_a, model_b, subset),
                          chunk_size=args.chunk_size)
            if not any(np.isfinite(v) for v in differences.values()):
                extreme, valid = _initialize_permutation_counts(metrics)
                permutation = _finalize_two_sided_permutation_pvalues(
                    metrics=metrics, extreme_counts=extreme, valid_counts=valid)
            elif subset == "measured_regression":
                permutation = paired_regression_permutation_pvalues(
                    y_true=y_reg[mask], prediction_a=a["analysis_logbb"][mask],
                    prediction_b=b["analysis_logbb"][mask], **kwargs)
            else:
                permutation = paired_classification_permutation_pvalues(
                    np_score_a=a["np_score"][mask], np_score_b=b["np_score"][mask],
                    logbb_prediction_a=a["analysis_logbb"][mask],
                    logbb_prediction_b=b["analysis_logbb"][mask], **kwargs)
            for metric in metrics:
                ci = bootstrap_difference_summary(
                    va["bootstrap"][metric] - vb["bootstrap"][metric], differences[metric], args.confidence)
                test = permutation[metric]
                missing = [f"{m}: {v['status'][metric]}" for m, v in [(model_a, va), (model_b, vb)]
                           if v["status"][metric] != "available"]
                if missing:
                    ci_status = p_status = "; ".join(missing)
                else:
                    ci_status = ("no_valid_bootstrap_replicates" if ci["n_bootstrap_valid"] == 0 else
                                 "conditional_on_defined_replicates" if ci["n_bootstrap_valid"] < args.bootstrap else "available")
                    p_status = ("no_valid_permutations" if test["n_valid_permutations"] == 0 else
                                "conditional_on_defined_replicates" if test["n_valid_permutations"] < args.permutations else "available")
                effect = float(metric_improvement(va["observed"][metric], vb["observed"][metric], metric))
                pairwise_rows.append({
                    "model_a": model_a, "model_a_display_name": display_model_name(model_a),
                    "model_b": model_b, "model_b_display_name": display_model_name(model_b),
                    "subset": subset, "subset_label": SUBSET_LABELS[subset], "metric": metric,
                    "n_paired": int(mask.sum()), "model_a_value": va["observed"][metric],
                    "model_b_value": vb["observed"][metric],
                    "model_a_status": va["status"][metric], "model_b_status": vb["status"][metric],
                    "model_a_n_incorrect_for_distance_mean": va["counts"].get(metric, np.nan),
                    "model_b_n_incorrect_for_distance_mean": vb["counts"].get(metric, np.nan),
                    **ci, "ci_status": ci_status, "p_status": p_status,
                    "better_direction": metric_direction(metric), "effect_in_favor_of_a": effect,
                    "observed_better_model": ("undefined" if not np.isfinite(effect) else
                                               model_a if effect > 0 else model_b if effect < 0 else "tie"),
                    **test, "n_permutations_requested": args.permutations,
                    "n_bootstrap_requested": args.bootstrap,
                })
        print(f"Tested pair {pair_number}/{len(all_pairs)}: {model_a} vs {model_b}", flush=True)

    results = add_pair_dataset_holm_adjustment(pd.DataFrame(pairwise_rows))
    results = results.sort_values(["model_a", "model_b", "subset", "metric"])
    # Every named model pair has every metric row, including explicit undefined cases.
    expected = len(all_pairs) * sum(len(REGRESSION_METRICS if s == "measured_regression"
                                       else CLASSIFICATION_METRICS) for s in masks)
    if len(results) != expected or results.duplicated(["model_a", "model_b", "subset", "metric"]).any():
        raise RuntimeError("Pairwise coverage is incomplete or duplicated.")
    results.to_csv(output_dir / "paired_two_sided_permutation_holm_by_pair_and_dataset.csv", index=False)
    for subset in SUBSET_LABELS:
        results[results.subset == subset].to_csv(output_dir / f"paired_permutation_{subset}.csv", index=False)
    audit_columns = ["model_a", "model_b", "subset", "metric", "model_a_status", "model_b_status",
                     "ci_status", "p_status", "n_bootstrap_valid", "n_valid_permutations"]
    unavailable = ~(np.isfinite(results.ci_lower) & np.isfinite(results.ci_upper)
                    & np.isfinite(results.permutation_p_two_sided_holm_pair_dataset))
    results.loc[unavailable, audit_columns].to_csv(output_dir / "unavailable_pairwise_statistics.csv", index=False)
    coverage = results.assign(ci_available=results.ci_lower.notna() & results.ci_upper.notna(),
                              p_available=results.permutation_p_two_sided_holm_pair_dataset.notna())
    coverage = coverage.groupby(["model_a", "model_b", "subset"], sort=False).agg(
        metric_rows=("metric", "size"), finite_CIs=("ci_available", "sum"), finite_p_values=("p_available", "sum")).reset_index()
    coverage["all_metric_rows_present"] = True
    coverage.to_csv(output_dir / "pairwise_coverage_audit.csv", index=False)

    stratification = "original outer fold × BBB class" if args.input_format == "oof" else "BBB class (external compounds)"
    distance_definition = (
        "Distance is mean |predicted logBB + 1| among incorrectly classified compounds, "
        "using each model's actual saved decisions (including tuned thresholds). The margin "
        "is fixed at -1; classification uses the table's classification population and regression "
        "uses measured-logBB compounds. Means with no mistakes or no predicted logBB are undefined."
    )
    metadata = {
        "schema_version": 2, "input_format": args.input_format,
        "prediction_source_files": [str(prediction_path)], "models": model_order,
        "n_unique_model_pairs": len(all_pairs), "n_expected_pairwise_metric_rows": expected,
        "n_actual_pairwise_metric_rows": len(results), "all_pairwise_metrics_exported": True,
        "permutation_test": "paired two-sided within-compound swap of model outputs; class decision, score, and predicted logBB move together",
        "permutation_replicates": args.permutations, "bootstrap_replicates": args.bootstrap,
        "bootstrap_purpose": "percentile confidence intervals for estimates and paired metric differences; not p-values",
        "bootstrap_stratification": stratification,
        "undefined_resample_policy": "Exclude only undefined metric replicates, never compounds; valid counts are exported and conditional inference is flagged.",
        "distance_definition": distance_definition,
        "wrong_side_count_definition": "Total classification errors: false positives + false negatives, using each model's actual saved decisions within the evaluated subset.",
        "np_wrong_side_count_definition": "Non-penetrant false negatives: true BBB class 1 predicted as class 0, using each model's actual saved decisions within the evaluated subset.",
        "zero_denominator_convention": "NP precision, NP F1 (when recall is defined), and MCC are zero for zero denominators; constant-input Pearson correlation and empty conditional means are undefined.",
        "holm_family": "all finite metric p-values within each unique model pair and subset; no adjustment across pairs or for later comparator selection",
        "confidence": args.confidence, "seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme = f"""# Complete paired model comparisons

All {len(all_pairs)} unique named model pairs are exported, including native/tuned
and pretrained variants with identical continuous predictions. Each unordered pair
is stored once. `difference_a_minus_b` and its CI always refer to model A minus B.
Reversing a comparison negates the difference and reverses/negates CI endpoints;
the two-sided p-value stays unchanged.

P-values use {args.permutations:,} within-compound paired prediction-swap permutations
with the finite-simulation correction (extreme + 1)/(valid + 1). Model outputs move
together. Class labels and evaluation populations never change. No thresholds are
fit on evaluation data. Holm correction covers all finite metrics within each
model pair/subset, including the new distance metrics. Consequently, the new
adjusted p-values may differ from older runs with smaller metric families.

{args.bootstrap:,} shared paired bootstrap draws, stratified by {stratification},
give {args.confidence:.1%} percentile CIs. Pooled `estimate_ci_*` columns describe an
individual model; pairwise `ci_lower/ci_upper` describe A minus B. The manuscript
notebook uses the pairwise intervals. Arithmetic and saved differences use float64.

{distance_definition}
`wrong_side_count` counts all classification errors (FP + FN).
`np_wrong_side_count` counts true non-penetrants predicted as penetrant (FN).
Both counts use each model's actual saved decisions, including tuned thresholds;
both receive estimate/difference CIs and paired permutation tests.
The NP/penetrant distance metrics restrict those mistakes by true class. Different
models can have different sets of mistakes. Mean distance is descriptive of those
sets; it should be read together with the mistake count.

Undefined metric replicates are excluded from that metric's resampling distribution
and flagged `conditional_on_defined_replicates`; their valid counts are exported.
No logBB prediction, constant-input Pearson correlation, or no mistakes can make
a metric and its comparison undefined. Such cases remain NA with an explicit
reason. Identical defined outputs yield delta 0, CI (0, 0), and p=1.

Files:
- `pooled_metrics_selected_models_long.csv`: every model/subset/metric and estimate CI.
- `paired_two_sided_permutation_holm_by_pair_and_dataset.csv`: all pairwise difference
  CIs, raw p-values, and `permutation_p_two_sided_holm_pair_dataset`.
- `pairwise_coverage_audit.csv`: complete metric-row and finite-statistic counts.
- `unavailable_pairwise_statistics.csv`: exact reasons for remaining unavailable tests.
- `subset_input_audit.csv`: evaluated or empty populations; no compounds are filtered.

For external predictions run this same script with `--input-format external` and
`--predictions-file path/to/01_all_external_predictions.csv`, choosing a separate
`--output-dir`. Predictions must already be aggregated to one row per model/compound
and scored so that higher score_np means NP. Optional experimental `true_logBB`
enables external regression; class labels alone cannot supply regression statistics.

Run notebook 19 with INPUT_MODE='results' and format='09' for each updated output
folder. An existing compiled CSV contains old statistics and must be regenerated.
The notebook retains custom MTL versus the better native MSE/logBB_pred comparator
per metric and places tuned MSE versus repeated custom MTL in separate tables.
The original per-pair Holm procedure does not correct across all model pairs or
for choosing the better comparator using these evaluation results.
"""
    (output_dir / "README_RESULTS.md").write_text(readme, encoding="utf-8")
    print(f"Complete: {len(model_order)} models, {len(all_pairs)} pairs, {len(results)} metric comparisons.")
    print(f"Undefined CI/p rows: {int(unavailable.sum())}; see unavailable_pairwise_statistics.csv.")
    print(f"Outputs written to: {output_dir}")

if __name__ == "__main__":
    main()
