#!/usr/bin/env python3
"""Paired two-sided permutation testing for two BBB models.

The input is a long-format prediction table with one row per compound/model
combination. Classification is evaluated on compounds with complete binary
labels, hard predictions, and scores from both models. Regression is evaluated
separately on compounds with a measured continuous logBB value and complete
continuous predictions from both models.

Only one inferential test is performed:

    paired within-compound two-sided permutation test

For each permutation, Model A and Model B outputs are swapped together within
each compound. This preserves pairing. Holm correction is then applied across
all metrics within each task separately for the selected model pair:

    * one Holm family containing 8 prespecified classification metrics
    * one Holm family containing 5 prespecified regression metrics

The retained manuscript metrics are:

Classification: balanced accuracy, non-penetrant recall, non-penetrant
precision, non-penetrant F1, ROC AUC, PR AUC, MCC, and non-penetrant
wrong-side count.

Regression: MAE, MSE, Pearson r, R2, and non-penetrant MAE.

The raw metric difference is always Model A minus Model B. The
``effect_in_favor_of_a`` column is oriented so positive values favor Model A
when a metric has a meaningful better direction.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PREDICTIONS_CSV = Path(
    "./external_test_set/01_eval/01_all_external_predictions.csv"
)
OUTPUT_DIR = Path("./external_test_set/04_eval_significance_evaluation_dummyr")
OUTPUT_STEM = "external_significance_permutation_only"

# This identifies the dataset/subset being evaluated. If the prediction table
# does not contain SUBSET_COL, the full table is used and this value is retained
# only as an output label.
SUBSET_NAME = "all_nonoverlapping"

MODEL_A = "custom_mtl_original"
MODEL_B = "dummy_mean_regression"

ID_COL = "external_compound_id"
SUBSET_COL = "subset"
MODEL_COL = "model"

# Classification columns.
TRUE_CLASS_COL = "Class"
PRED_CLASS_COL = "pred_np"
SCORE_CLASS_COL = "score_np"

# Regression columns. Change these to match the merged prediction CSV.
TRUE_REG_COL = "logBB"
PRED_REG_COL = "predicted_logbb"

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Metric direction
# ---------------------------------------------------------------------

# +1: larger is better; -1: smaller is better; 0: neither signed direction is
# inherently better. Direction does not affect the two-sided p-value; it is
# used only to make ``effect_in_favor_of_a`` interpretable.
CLASSIFICATION_DIRECTION = {
    "balanced_accuracy": 1,
    "np_recall": 1,
    "np_precision": 1,
    "np_f1": 1,
    "roc_auc": 1,
    "pr_auc": 1,
    "mcc": 1,
    "np_wrong_side_count": -1,
}

REGRESSION_DIRECTION = {
    "mae": -1,
    "mse": -1,
    "pearson_r": 1,
    "r2": 1,
    "mae_nonpenetrant": -1,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or not np.isfinite(score).all():
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or not np.isfinite(score).all():
        return np.nan
    return float(average_precision_score(y_true, score))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if (
        len(y_true) < 2
        or np.allclose(y_true, y_true[0])
        or np.allclose(y_pred, y_pred[0])
    ):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pearsonr(y_true, y_pred).statistic)



def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm family-wise error correction, preserving the original index."""
    pvalues = pd.to_numeric(pvalues, errors="coerce")
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()

    m = len(valid)
    running_max = 0.0
    for rank, (index, raw_p) in enumerate(valid.items()):
        candidate = min(1.0, (m - rank) * float(raw_p))
        running_max = max(running_max, candidate)
        adjusted.loc[index] = running_max

    return adjusted


def metric_direction_fields(
    raw_difference: float,
    direction: int,
) -> tuple[str, float]:
    """Return a readable direction and an effect oriented in favor of A."""
    if direction == 1:
        return "higher", raw_difference
    if direction == -1:
        return "lower", -raw_difference
    return "two_sided_only", np.nan


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics with non-penetrant encoded as 1."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "np_recall": float(recall),
        "np_precision": float(precision),
        "np_f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, score),
        "pr_auc": safe_pr_auc(y_true, score),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "np_wrong_side_count": float(fn),
    }


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bbb_class: np.ndarray,
) -> dict[str, float]:
    """Calculate overall and class-stratified regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bbb_class = np.asarray(bbb_class, dtype=int)

    nonpenetrant_mask = bbb_class == 1

    mse = float(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "pearson_r": safe_pearson(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "mae_nonpenetrant": (
            float(
                mean_absolute_error(
                    y_true[nonpenetrant_mask],
                    y_pred[nonpenetrant_mask],
                )
            )
            if nonpenetrant_mask.any()
            else np.nan
        ),
    }


# ---------------------------------------------------------------------
# Paired two-sided permutation test
# ---------------------------------------------------------------------


def paired_two_sided_permutation_test(
    *,
    metric_function,
    fixed_arguments: tuple[np.ndarray, ...],
    swappable_arguments_a: tuple[np.ndarray, ...],
    swappable_arguments_b: tuple[np.ndarray, ...],
    metric_direction: dict[str, int],
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run paired within-compound two-sided permutation tests for all metrics.

    Fixed arguments, such as true labels, are supplied unchanged to both
    models. Corresponding Model A and Model B outputs are swapped together
    within each compound with probability 0.5.
    """
    if len(swappable_arguments_a) != len(swappable_arguments_b):
        raise ValueError("A/B swappable argument counts differ.")
    if not swappable_arguments_a:
        raise ValueError("At least one swappable model output is required.")

    n = len(swappable_arguments_a[0])
    all_arguments = (
        *fixed_arguments,
        *swappable_arguments_a,
        *swappable_arguments_b,
    )
    if any(len(argument) != n for argument in all_arguments):
        raise ValueError("Permutation-test arguments have inconsistent lengths.")

    observed_a = metric_function(*fixed_arguments, *swappable_arguments_a)
    observed_b = metric_function(*fixed_arguments, *swappable_arguments_b)
    metric_names = list(observed_a)

    observed_differences = {
        metric: observed_a[metric] - observed_b[metric]
        for metric in metric_names
    }

    rng = np.random.default_rng(random_seed)
    null_differences = {
        metric: np.full(n_permutations, np.nan, dtype=float)
        for metric in metric_names
    }

    for iteration in range(n_permutations):
        swap = rng.random(n) < 0.5

        permuted_a = tuple(
            np.where(swap, values_b, values_a)
            for values_a, values_b in zip(
                swappable_arguments_a,
                swappable_arguments_b,
            )
        )
        permuted_b = tuple(
            np.where(swap, values_a, values_b)
            for values_a, values_b in zip(
                swappable_arguments_a,
                swappable_arguments_b,
            )
        )

        metrics_a = metric_function(*fixed_arguments, *permuted_a)
        metrics_b = metric_function(*fixed_arguments, *permuted_b)

        for metric in metric_names:
            null_differences[metric][iteration] = (
                metrics_a[metric] - metrics_b[metric]
            )

    rows: list[dict[str, float | int | str]] = []
    for metric in metric_names:
        observed = observed_differences[metric]
        finite_null = null_differences[metric][
            np.isfinite(null_differences[metric])
        ]

        if not np.isfinite(observed) or not len(finite_null):
            p_two_sided = np.nan
        else:
            p_two_sided = float(
                (1 + np.sum(np.abs(finite_null) >= abs(observed)))
                / (len(finite_null) + 1)
            )

        better_direction, effect_in_favor_of_a = metric_direction_fields(
            observed,
            metric_direction[metric],
        )

        rows.append(
            {
                "metric": metric,
                "model_a_value": observed_a[metric],
                "model_b_value": observed_b[metric],
                "difference_a_minus_b": observed,
                "better_direction": better_direction,
                "effect_in_favor_of_a": effect_in_favor_of_a,
                "permutation_p_two_sided": p_two_sided,
                "n_valid_permutations": int(len(finite_null)),
                "n_permutations_requested": int(n_permutations),
            }
        )

    results = pd.DataFrame(rows)

    # This function is called separately for classification and regression.
    # Therefore, Holm correction is confined to the metrics within that task
    # for this model pair and this dataset/subset.
    results["permutation_p_two_sided_holm"] = holm_adjust(
        results["permutation_p_two_sided"]
    )
    return results


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------


def load_prediction_table() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(PREDICTIONS_CSV)

    predictions = pd.read_csv(PREDICTIONS_CSV, low_memory=False)

    required_before_filter = {ID_COL, MODEL_COL}
    missing_before_filter = required_before_filter - set(predictions.columns)
    if missing_before_filter:
        raise ValueError(
            f"Missing required columns: {sorted(missing_before_filter)}. "
            f"Available columns: {list(predictions.columns)}"
        )

    if SUBSET_COL in predictions.columns:
        predictions = predictions.loc[
            predictions[SUBSET_COL].astype(str) == SUBSET_NAME
        ].copy()

    predictions = predictions.loc[
        predictions[MODEL_COL].astype(str).isin([MODEL_A, MODEL_B])
    ].copy()

    required_columns = {
        ID_COL,
        MODEL_COL,
        TRUE_CLASS_COL,
        PRED_CLASS_COL,
        SCORE_CLASS_COL,
        TRUE_REG_COL,
        PRED_REG_COL,
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(predictions.columns)}"
        )

    duplicates = predictions.duplicated(
        subset=[ID_COL, MODEL_COL],
        keep=False,
    )
    if duplicates.any():
        duplicate_rows = predictions.loc[
            duplicates,
            [ID_COL, MODEL_COL],
        ]
        raise ValueError(
            "Duplicate compound/model predictions were found:\n"
            f"{duplicate_rows.head(20)}"
        )

    numeric_columns = [
        TRUE_CLASS_COL,
        PRED_CLASS_COL,
        SCORE_CLASS_COL,
        TRUE_REG_COL,
        PRED_REG_COL,
    ]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    class_label_counts = predictions.groupby(ID_COL)[TRUE_CLASS_COL].nunique(
        dropna=True
    )
    if (class_label_counts > 1).any():
        raise ValueError("Some compounds have conflicting true class labels.")

    regression_label_counts = predictions.groupby(ID_COL)[TRUE_REG_COL].nunique(
        dropna=True
    )
    if (regression_label_counts > 1).any():
        raise ValueError("Some compounds have conflicting true logBB values.")

    available_models = set(predictions[MODEL_COL].astype(str).unique())
    missing_models = {MODEL_A, MODEL_B} - available_models
    if missing_models:
        raise ValueError(
            f"Selected models are missing after filtering: {sorted(missing_models)}. "
            f"Available models: {sorted(available_models)}"
        )

    return predictions


def build_wide_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.pivot(
        index=ID_COL,
        columns=MODEL_COL,
        values=[
            TRUE_CLASS_COL,
            PRED_CLASS_COL,
            SCORE_CLASS_COL,
            TRUE_REG_COL,
            PRED_REG_COL,
        ],
    )


def classification_arrays(
    wide: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    needed = [
        (TRUE_CLASS_COL, MODEL_A),
        (TRUE_CLASS_COL, MODEL_B),
        (PRED_CLASS_COL, MODEL_A),
        (PRED_CLASS_COL, MODEL_B),
        (SCORE_CLASS_COL, MODEL_A),
        (SCORE_CLASS_COL, MODEL_B),
    ]
    paired = wide.dropna(subset=needed).copy()

    y_true = paired[(TRUE_CLASS_COL, MODEL_A)].to_numpy(dtype=int)
    y_true_b = paired[(TRUE_CLASS_COL, MODEL_B)].to_numpy(dtype=int)
    if not np.array_equal(y_true, y_true_b):
        raise ValueError("True class labels differ between model rows.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Classification labels must be encoded as 0/1.")

    pred_a = paired[(PRED_CLASS_COL, MODEL_A)].to_numpy(dtype=int)
    pred_b = paired[(PRED_CLASS_COL, MODEL_B)].to_numpy(dtype=int)
    if not set(np.unique(pred_a)).issubset({0, 1}):
        raise ValueError(f"{MODEL_A} hard predictions are not all 0/1.")
    if not set(np.unique(pred_b)).issubset({0, 1}):
        raise ValueError(f"{MODEL_B} hard predictions are not all 0/1.")

    score_a = paired[(SCORE_CLASS_COL, MODEL_A)].to_numpy(dtype=float)
    score_b = paired[(SCORE_CLASS_COL, MODEL_B)].to_numpy(dtype=float)

    finite = np.isfinite(score_a) & np.isfinite(score_b)
    if not finite.all():
        paired = paired.iloc[np.flatnonzero(finite)]
        y_true = y_true[finite]
        pred_a = pred_a[finite]
        pred_b = pred_b[finite]
        score_a = score_a[finite]
        score_b = score_b[finite]

    return y_true, pred_a, pred_b, score_a, score_b, paired.index


def regression_arrays(
    wide: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    needed = [
        (TRUE_CLASS_COL, MODEL_A),
        (TRUE_CLASS_COL, MODEL_B),
        (TRUE_REG_COL, MODEL_A),
        (TRUE_REG_COL, MODEL_B),
        (PRED_REG_COL, MODEL_A),
        (PRED_REG_COL, MODEL_B),
    ]
    paired = wide.dropna(subset=needed).copy()

    y_true = paired[(TRUE_REG_COL, MODEL_A)].to_numpy(dtype=float)
    y_true_b = paired[(TRUE_REG_COL, MODEL_B)].to_numpy(dtype=float)
    if not np.allclose(y_true, y_true_b, equal_nan=False):
        raise ValueError("True logBB values differ between model rows.")

    bbb_class = paired[(TRUE_CLASS_COL, MODEL_A)].to_numpy(dtype=int)
    bbb_class_b = paired[(TRUE_CLASS_COL, MODEL_B)].to_numpy(dtype=int)
    if not np.array_equal(bbb_class, bbb_class_b):
        raise ValueError("True class labels differ between regression model rows.")

    pred_a = paired[(PRED_REG_COL, MODEL_A)].to_numpy(dtype=float)
    pred_b = paired[(PRED_REG_COL, MODEL_B)].to_numpy(dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    if not finite.all():
        paired = paired.iloc[np.flatnonzero(finite)]
        y_true = y_true[finite]
        bbb_class = bbb_class[finite]
        pred_a = pred_a[finite]
        pred_b = pred_b[finite]

    return y_true, bbb_class, pred_a, pred_b, paired.index


# ---------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------


def add_analysis_metadata(
    results: pd.DataFrame,
    *,
    task: str,
    n_paired: int,
    n_penetrant: int,
    n_nonpenetrant: int,
) -> pd.DataFrame:
    results = results.copy()
    results.insert(0, "dataset", SUBSET_NAME)
    results.insert(1, "model_a", MODEL_A)
    results.insert(2, "model_b", MODEL_B)
    results.insert(3, "task", task)
    results.insert(4, "n_paired", n_paired)
    results.insert(5, "n_penetrant", n_penetrant)
    results.insert(6, "n_nonpenetrant", n_nonpenetrant)
    return results


def run_classification_analysis(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> pd.DataFrame:
    results = paired_two_sided_permutation_test(
        metric_function=calculate_classification_metrics,
        fixed_arguments=(y_true,),
        swappable_arguments_a=(pred_a, score_a),
        swappable_arguments_b=(pred_b, score_b),
        metric_direction=CLASSIFICATION_DIRECTION,
        n_permutations=N_PERMUTATIONS,
        random_seed=RANDOM_SEED,
    )

    return add_analysis_metadata(
        results,
        task="classification",
        n_paired=len(y_true),
        n_penetrant=int(np.sum(y_true == 0)),
        n_nonpenetrant=int(np.sum(y_true == 1)),
    )


def run_regression_analysis(
    y_true: np.ndarray,
    bbb_class: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> pd.DataFrame:
    # calculate_regression_metrics expects (y_true, y_pred, bbb_class), while
    # the generic permutation function places fixed arguments before swappable
    # model outputs. This wrapper preserves the required argument order.
    def metric_wrapper(
        fixed_y_true: np.ndarray,
        fixed_bbb_class: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, float]:
        return calculate_regression_metrics(
            fixed_y_true,
            predicted,
            fixed_bbb_class,
        )

    results = paired_two_sided_permutation_test(
        metric_function=metric_wrapper,
        fixed_arguments=(y_true, bbb_class),
        swappable_arguments_a=(pred_a,),
        swappable_arguments_b=(pred_b,),
        metric_direction=REGRESSION_DIRECTION,
        n_permutations=N_PERMUTATIONS,
        random_seed=RANDOM_SEED + 1,
    )

    return add_analysis_metadata(
        results,
        task="regression",
        n_paired=len(y_true),
        n_penetrant=int(np.sum(bbb_class == 0)),
        n_nonpenetrant=int(np.sum(bbb_class == 1)),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    predictions = load_prediction_table()
    wide = build_wide_table(predictions)

    (
        class_y,
        class_pred_a,
        class_pred_b,
        class_score_a,
        class_score_b,
        classification_ids,
    ) = classification_arrays(wide)

    (
        reg_y,
        reg_class,
        reg_pred_a,
        reg_pred_b,
        regression_ids,
    ) = regression_arrays(wide)

    if len(class_y) == 0:
        raise ValueError("No compounds are evaluable for classification.")
    if len(reg_y) == 0:
        raise ValueError(
            "No compounds are evaluable for regression. Check TRUE_REG_COL "
            "and PRED_REG_COL and confirm measured logBB values are present."
        )

    print(f"Dataset/subset: {SUBSET_NAME}")
    print(f"Models: {MODEL_A} versus {MODEL_B}")
    print(
        "Classification paired compounds: "
        f"{len(class_y)} "
        f"(penetrant={np.sum(class_y == 0)}, "
        f"non-penetrant={np.sum(class_y == 1)})"
    )
    print(
        "Regression paired compounds: "
        f"{len(reg_y)} "
        f"(penetrant={np.sum(reg_class == 0)}, "
        f"non-penetrant={np.sum(reg_class == 1)})"
    )

    classification_results = run_classification_analysis(
        class_y,
        class_pred_a,
        class_pred_b,
        class_score_a,
        class_score_b,
    )
    regression_results = run_regression_analysis(
        reg_y,
        reg_class,
        reg_pred_a,
        reg_pred_b,
    )

    combined = pd.concat(
        [classification_results, regression_results],
        ignore_index=True,
        sort=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classification_path = OUTPUT_DIR / (
        OUTPUT_STEM + "_classification.csv"
    )
    regression_path = OUTPUT_DIR / (
        OUTPUT_STEM + "_regression.csv"
    )
    combined_path = OUTPUT_DIR / (
        OUTPUT_STEM + "_combined.csv"
    )
    audit_path = OUTPUT_DIR / (
        OUTPUT_STEM + "_paired_compounds.csv"
    )

    classification_results.to_csv(classification_path, index=False)
    regression_results.to_csv(regression_path, index=False)
    combined.to_csv(combined_path, index=False)

    audit = pd.concat(
        [
            pd.DataFrame(
                {
                    ID_COL: classification_ids.astype(str),
                    "dataset": SUBSET_NAME,
                    "model_a": MODEL_A,
                    "model_b": MODEL_B,
                    "task": "classification",
                }
            ),
            pd.DataFrame(
                {
                    ID_COL: regression_ids.astype(str),
                    "dataset": SUBSET_NAME,
                    "model_a": MODEL_A,
                    "model_b": MODEL_B,
                    "task": "regression",
                }
            ),
        ],
        ignore_index=True,
    )
    audit.to_csv(audit_path, index=False)

    display_columns = [
        "dataset",
        "task",
        "metric",
        "model_a_value",
        "model_b_value",
        "difference_a_minus_b",
        "permutation_p_two_sided",
        "permutation_p_two_sided_holm",
    ]

    print("\nPaired two-sided permutation results:")
    print(combined[display_columns].to_string(index=False))

    print("\nHolm correction families:")
    print(
        "  classification: 8 prespecified metrics for this dataset and model pair"
    )
    print(
        "  regression: 5 prespecified metrics for this dataset and model pair"
    )

    print("\nOutputs:")
    for path in [
        classification_path,
        regression_path,
        combined_path,
        audit_path,
    ]:
        print(path.resolve())


if __name__ == "__main__":
    main()
