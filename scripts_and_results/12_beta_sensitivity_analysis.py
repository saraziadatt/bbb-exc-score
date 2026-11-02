
#!/usr/bin/env python3
"""
Run a beta sensitivity analysis for the custom multi-task LightGBM BBB model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
try:
    import optuna
except ImportError:
    optuna = None
import pandas as pd
import sklearn
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
from sklearn.model_selection import StratifiedKFold


DEFAULT_TARGET_COL = "logBB"
DEFAULT_CLASS_COL = "bbb_class"
DEFAULT_FOLD_COL = "outer_fold"
DEFAULT_SMILES_COL = "smiles"
DEFAULT_ID_COL = "sample_id"

MODEL_NAME = "custom_mtl_bab"
MODEL_TASKS = {MODEL_NAME: "regression"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value



def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()



def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])



def regression_to_classification(
    predicted_logbb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_logbb = np.asarray(predicted_logbb, dtype=float)
    return (predicted_logbb < threshold).astype(int), -predicted_logbb


# ---------------------------------------------------------------------------
# Metrics

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_class: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_class_array = (
        np.full(len(y_true), np.nan)
        if y_class is None
        else np.asarray(y_class, dtype=float)
    )
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    yc = y_class_array[mask]
    if len(yt) == 0:
        return {
            "n_evaluable": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "median_absolute_error": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "np_mae": np.nan,
            "penetrant_mae": np.nan,
        }
    mse = float(mean_squared_error(yt, yp))
    output = {
        "n_evaluable": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": float(median_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "pearson_r": safe_pearson(yt, yp),
        "np_mae": np.nan,
        "penetrant_mae": np.nan,
    }
    np_mask = yc == 1
    p_mask = yc == 0
    if np.any(np_mask):
        output["np_mae"] = float(mean_absolute_error(yt[np_mask], yp[np_mask]))
    if np.any(p_mask):
        output["penetrant_mae"] = float(
            mean_absolute_error(yt[p_mask], yp[p_mask])
        )
    return output



def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_nonpenetrant: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score_nonpenetrant = np.asarray(score_nonpenetrant, dtype=float)
    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(score_nonpenetrant)
    )
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    ys = score_nonpenetrant[mask]
    if len(yt) == 0:
        return {
            "n_evaluable": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "np_recall": np.nan,
            "np_precision": np.nan,
            "np_f1": np.nan,
            "penetrant_specificity": np.nan,
            "roc_auc": np.nan,
            "pr_auc_nonpenetrant": np.nan,
            "cohen_kappa": np.nan,
            "mcc": np.nan,
            "wrong_side_count": 0,
            "wrong_side_rate": np.nan,
            "np_wrong_side_count": 0,
            "penetrant_wrong_side_count": 0,
        }
    wrong = yp != yt
    np_mask = yt == 1
    p_mask = yt == 0
    has_both = len(np.unique(yt)) == 2
    return {
        "n_evaluable": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "np_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "np_precision": float(
            precision_score(yt, yp, pos_label=1, zero_division=0)
        ),
        "np_f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "penetrant_specificity": (
            float(np.mean(yp[p_mask] == 0)) if np.any(p_mask) else np.nan
        ),
        "roc_auc": float(roc_auc_score(yt, ys)) if has_both else np.nan,
        "pr_auc_nonpenetrant": (
            float(average_precision_score(yt, ys)) if has_both else np.nan
        ),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "wrong_side_count": int(np.sum(wrong)),
        "wrong_side_rate": float(np.mean(wrong)),
        "np_wrong_side_count": int(np.sum(wrong & np_mask)),
        "penetrant_wrong_side_count": int(np.sum(wrong & p_mask)),
    }


# ---------------------------------------------------------------------------
# Data loading and folds
# ---------------------------------------------------------------------------


def infer_descriptor_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    class_col: str,
    fold_col: str,
    smiles_col: str,
    id_col: str,
) -> list[str]:
    metadata = {
        target_col,
        class_col,
        fold_col,
        smiles_col,
        id_col,
        "has_logbb",
        "logbb_is_measured",
        "classification_only",
        "outer_stratum",
        "row_index",
    }
    metadata = {str(column).strip().lower() for column in metadata}
    excluded_prefixes = (
        "inner_fold_outer_",
        "pred_",
        "prediction",
        "oof_",
        "shap_",
    )
    columns: list[str] = []
    for column in df.columns:
        name = str(column)
        lower = name.strip().lower()
        if lower in metadata:
            continue
        if lower.startswith(excluded_prefixes):
            continue
        if lower.startswith("unnamed:"):
            continue
        if lower in {"index", "level_0", "fold", "target", "label", "class"}:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric molecular descriptor columns were found.")
    return columns



def check_feature_target_correlation(
    X: pd.DataFrame,
    y_reg: np.ndarray,
) -> pd.DataFrame:
    measured = np.isfinite(y_reg)
    rows: list[dict[str, Any]] = []
    if np.sum(measured) < 3:
        return pd.DataFrame(columns=["feature", "pearson_r", "abs_pearson_r"])
    target = y_reg[measured]
    for column in X.columns:
        values = pd.to_numeric(X.loc[measured, column], errors="coerce").to_numpy(
            float
        )
        mask = np.isfinite(values) & np.isfinite(target)
        correlation = (
            safe_pearson(values[mask], target[mask]) if np.sum(mask) >= 3 else np.nan
        )
        rows.append(
            {
                "feature": column,
                "pearson_r": correlation,
                "abs_pearson_r": (
                    abs(correlation) if np.isfinite(correlation) else np.nan
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_pearson_r", ascending=False, na_position="last")
        .reset_index(drop=True)
    )



@dataclass
class LoadedData:
    frame: pd.DataFrame
    X: pd.DataFrame
    feature_columns: list[str]
    y_reg: np.ndarray
    y_class: np.ndarray
    outer_fold: np.ndarray
    sample_id: np.ndarray
    smiles: np.ndarray



def load_master_data(args: argparse.Namespace) -> LoadedData:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path, low_memory=False)
    required = {
        args.target_col,
        args.class_col,
        args.fold_col,
        args.smiles_col,
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if args.id_col not in df.columns:
        df[args.id_col] = np.arange(len(df), dtype=int)

    y_reg = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)
    y_class = pd.to_numeric(df[args.class_col], errors="raise").to_numpy(int)
    outer_fold = pd.to_numeric(df[args.fold_col], errors="raise").to_numpy(int)
    if not set(np.unique(y_class)).issubset({0, 1}):
        raise ValueError(
            f"{args.class_col} must use 0=penetrant and 1=non-penetrant."
        )

    feature_columns = infer_descriptor_columns(
        df,
        target_col=args.target_col,
        class_col=args.class_col,
        fold_col=args.fold_col,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
    )

    forbidden_features = {
        "has_logbb",
        "logbb_is_measured",
        "classification_only",
        "outer_fold",
        "outer_stratum",
        "row_index",
        "logbb",
        "bbb_class",
    }

    leaked_features = [
        column
        for column in feature_columns
        if str(column).strip().lower() in forbidden_features
    ]

    if leaked_features:
        raise RuntimeError(
            f"Metadata or target columns entered the descriptor set: "
            f"{leaked_features}"
        )

    print(f"Using {len(feature_columns)} molecular descriptors.")
    print(f"First descriptors: {feature_columns[:10]}")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        warnings.warn(
            f"Dropping {len(all_nan)} globally all-NaN descriptors. "
            f"Examples: {all_nan[:10]}"
        )
        X = X.drop(columns=all_nan)
        feature_columns = X.columns.tolist()

    measured = np.isfinite(y_reg)
    implied = (y_reg[measured] < args.logbb_threshold).astype(int)
    disagreement = int(np.sum(implied != y_class[measured]))
    if disagreement:
        message = (
            f"{disagreement} measured rows disagree between {args.class_col} "
            f"and logBB threshold {args.logbb_threshold}."
        )
        if args.allow_label_disagreement:
            warnings.warn(message)
        else:
            raise ValueError(
                message + " Use --allow-label-disagreement only if intentional."
            )

    return LoadedData(
        frame=df,
        X=X,
        feature_columns=feature_columns,
        y_reg=y_reg,
        y_class=y_class,
        outer_fold=outer_fold,
        sample_id=df[args.id_col].to_numpy(),
        smiles=df[args.smiles_col].astype(str).to_numpy(),
    )



def saved_inner_splits(
    df: pd.DataFrame,
    *,
    outer_fold: int,
    fit_indices: np.ndarray,
) -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
    column = f"inner_fold_outer_{outer_fold}"
    if column not in df.columns:
        return None
    assignments = pd.to_numeric(df.loc[fit_indices, column], errors="coerce")
    if assignments.isna().any():
        return None
    values = assignments.to_numpy(int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for value in sorted(np.unique(values)):
        validation = np.flatnonzero(values == value)
        training = np.flatnonzero(values != value)
        if len(training) and len(validation):
            splits.append((training, validation))
    return splits if len(splits) >= 2 else None



def classification_inner_splits(
    y_class: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = pd.Series(y_class).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Insufficient samples per class for {n_splits}-fold CV: "
            f"{counts.to_dict()}"
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros((len(y_class), 1)), y_class))


def select_pareto_trial(
    pareto_trials: list[optuna.trial.FrozenTrial],
) -> tuple[optuna.trial.FrozenTrial, dict[str, float]]:
    """Select one Pareto-optimal trial using equal-weight normalized distance.

    Objective 0 is mean validation MSE (lower is better).
    Objective 1 is mean validation balanced accuracy (higher is better).

    Both objectives are min-max normalized across the Pareto front, then the
    trial closest to the ideal point (MSE=best, BA=best) is selected by
    Euclidean distance. This avoids combining raw MSE and balanced accuracy,
    which are on different numerical scales.
    """
    if not pareto_trials:
        raise RuntimeError("Optuna returned an empty Pareto front.")

    mse = np.asarray([float(t.values[0]) for t in pareto_trials], dtype=float)
    ba = np.asarray([float(t.values[1]) for t in pareto_trials], dtype=float)

    mse_min, mse_max = float(np.min(mse)), float(np.max(mse))
    ba_min, ba_max = float(np.min(ba)), float(np.max(ba))

    if mse_max > mse_min:
        mse_distance = (mse - mse_min) / (mse_max - mse_min)
    else:
        mse_distance = np.zeros_like(mse)

    if ba_max > ba_min:
        ba_distance = (ba_max - ba) / (ba_max - ba_min)
    else:
        ba_distance = np.zeros_like(ba)

    distance = np.sqrt(mse_distance ** 2 + ba_distance ** 2)
    best_index = int(np.argmin(distance))
    selected = pareto_trials[best_index]

    selection_info = {
        "pareto_distance_to_ideal": float(distance[best_index]),
        "pareto_mse_min": mse_min,
        "pareto_mse_max": mse_max,
        "pareto_ba_min": ba_min,
        "pareto_ba_max": ba_max,
        "n_pareto_trials": int(len(pareto_trials)),
    }
    return selected, selection_info


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def custom_multi_loss(
    y_train_class: np.ndarray,
    y_reg_with_nans: np.ndarray,
    *,
    alpha: float,
    beta: float,
    margin: float = -1.0,
    margin_buffer: float = 0.1,
    k: float = 10.0,
):
    del k
    y_train_class = np.asarray(y_train_class, dtype=int)
    y_reg_with_nans = np.asarray(y_reg_with_nans, dtype=float)

    def objective(predictions: np.ndarray, train_data: lgb.Dataset):
        del train_data
        
        predictions = np.asarray(predictions, dtype=float)
        has_regression = np.isfinite(y_reg_with_nans)

        # regression
        grad_reg = np.zeros_like(predictions)
        hess_reg = np.zeros_like(predictions)
        grad_reg[has_regression] = alpha * (predictions[has_regression] - y_reg_with_nans[has_regression])
        hess_reg[has_regression] = alpha

        # classification
        grad_class = np.zeros_like(predictions)
        hess_class = np.zeros_like(predictions)
        penetrant_target = margin + margin_buffer
        nonpenetrant_target = margin - margin_buffer
        n_penetrant = int(np.sum(y_train_class == 0))
        n_nonpenetrant = int(np.sum(y_train_class == 1))
        weight_penetrant = 1.0 / n_penetrant if n_penetrant else 0.0
        weight_nonpenetrant = 1.0 / n_nonpenetrant if n_nonpenetrant else 0.0

        wrong_nonpenetrant = ((y_train_class == 1) & (predictions > nonpenetrant_target))
        difference = predictions[wrong_nonpenetrant] - nonpenetrant_target
        grad_class[wrong_nonpenetrant] = beta * weight_nonpenetrant * 2.0 * difference
        hess_class[wrong_nonpenetrant] = beta * weight_nonpenetrant * 2.0

        wrong_penetrant = ((y_train_class == 0) & (predictions < penetrant_target))
        difference = predictions[wrong_penetrant] - penetrant_target
        grad_class[wrong_penetrant] = beta * weight_penetrant * 2.0 * difference
        hess_class[wrong_penetrant] = beta * weight_penetrant * 2.0

        return grad_reg + grad_class, hess_reg + hess_class

    return objective



def prepare_fold_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preserve descriptor order and NaNs for native LightGBM handling."""
    feature_names = X_train.columns.tolist()
    return (
        X_train.loc[:, feature_names].to_numpy(dtype=float),
        X_test.loc[:, feature_names].to_numpy(dtype=float),
        feature_names,
    )


def save_optuna_trials(study, trials_dir: Path, outer_fold: int, selected_number=None):
    """Export all trials and the current Pareto front after each trial."""
    trials_dir.mkdir(parents=True, exist_ok=True)
    pareto_numbers = {trial.number for trial in study.best_trials}
    rows = []
    for trial in study.trials:
        values = trial.values
        rows.append({
            "outer_fold": int(outer_fold),
            "trial_number": trial.number,
            "state": trial.state.name,
            "mean_inner_mse": values[0] if values is not None else None,
            "mean_inner_balanced_accuracy": values[1] if values is not None else None,
            "is_pareto": trial.number in pareto_numbers,
            "is_selected": trial.number == selected_number,
            **{f"param_{key}": value for key, value in trial.params.items()},
            "user_attrs_json": json.dumps(json_safe(trial.user_attrs), sort_keys=True),
        })
    frame = pd.DataFrame(rows)
    for name, table in [
        (f"outer_{outer_fold}_trials.csv", frame),
        (f"outer_{outer_fold}_pareto.csv", frame.loc[frame["is_pareto"]]),
    ]:
        target = trials_dir / name
        temporary = target.with_suffix(".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(target)
    # Keeps the Study object for optuna.visualization.plot_pareto_front.
    target = trials_dir / f"outer_{outer_fold}_study.joblib"
    temporary = target.with_suffix(".tmp")
    joblib.dump(study, temporary)
    temporary.replace(target)


def tune_and_fit_custom_mtl(
    *,
    X_train: np.ndarray,
    y_reg_train: np.ndarray,
    y_class_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_trials: int,
    boost_rounds: int,
    margin: float,
    trials_dir: Path,
    outer_fold: int,
) -> tuple[lgb.Booster, dict[str, Any], dict[str, float]]:
    """Tune MTL + LightGBM hyperparameters on MSE and balanced accuracy.

    Inner-CV objectives
    -------------------
    1. Minimize mean MSE on validation observations with measured logBB.
    2. Maximize mean balanced accuracy on all validation observations.

    Balanced accuracy is calculated after converting predicted logBB to
    BBB class using the fixed biologically defined `margin` threshold.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        fold_mse: list[float] = []
        fold_balanced_accuracy: list[float] = []

        # Custom-loss hyperparameters
        beta = trial.suggest_float("beta", 0.5, 5.0, step=0.25)
        margin_buffer = trial.suggest_float(
            "margin_buffer",
            0.15,
            0.35,
            step=0.02,
        )

        # LightGBM hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        )

        for inner_number, (fit_rel, val_rel) in enumerate(inner_splits):
            # The callable objective uses the original NaN-containing targets
            # captured in its closure. Keep Dataset labels finite so LightGBM
            # never interprets missing regression targets itself. Values used
            # here for classification-only rows are inert and are not target
            # imputation or additional regression supervision.
            dataset_label = np.where(
                np.isfinite(y_reg_train[fit_rel]),
                y_reg_train[fit_rel],
                0.0,
            )
            train_data = lgb.Dataset(
                X_train[fit_rel],
                label=dataset_label,
                free_raw_data=False,
            )

            inner_seed = seed + 100 * inner_number

            parameters = {
                "objective": custom_multi_loss(
                    y_class_train[fit_rel],
                    y_reg_train[fit_rel],
                    alpha=1.0,
                    beta=beta,
                    margin=margin,
                    margin_buffer=margin_buffer,
                ),
                "metric": "None",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "subsample": subsample,
                "bagging_freq": 1,
                "colsample_bytree": colsample_bytree,
                "seed": inner_seed,
                "feature_fraction_seed": inner_seed,
                "bagging_seed": inner_seed,
                "drop_seed": inner_seed,
                "data_random_seed": inner_seed,
                "extra_seed": inner_seed,
                "force_col_wise": True,
                "num_threads": 1,
            }

            model = lgb.train(
                parameters,
                train_data,
                num_boost_round=boost_rounds,
            )

            validation_prediction = np.asarray(
                model.predict(X_train[val_rel]),
                dtype=float,
            )

            # Regression objective: only validation rows with measured logBB.
            measured_validation = np.isfinite(y_reg_train[val_rel])
            if np.any(measured_validation):
                fold_mse.append(
                    float(
                        mean_squared_error(
                            y_reg_train[val_rel][measured_validation],
                            validation_prediction[measured_validation],
                        )
                    )
                )

            # Classification objective: ALL validation rows, including rows
            # without measured logBB, because all rows have bbb_class labels.
            validation_class, _ = regression_to_classification(
                validation_prediction,
                margin,
            )
            fold_balanced_accuracy.append(
                float(
                    balanced_accuracy_score(
                        y_class_train[val_rel],
                        validation_class,
                    )
                )
            )

        if not fold_mse:
            raise RuntimeError(
                "No measured validation observations were available during "
                "custom-MTL tuning."
            )
        if not fold_balanced_accuracy:
            raise RuntimeError(
                "No validation observations were available for balanced "
                "accuracy during custom-MTL tuning."
            )

        mean_mse = float(np.mean(fold_mse))
        mean_ba = float(np.mean(fold_balanced_accuracy))

        # Save readable attributes in the Optuna trial table.
        trial.set_user_attr("mean_inner_mse", mean_mse)
        trial.set_user_attr("mean_inner_balanced_accuracy", mean_ba)

        return mean_mse, mean_ba

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[lambda study, trial: save_optuna_trials(study, trials_dir, outer_fold)],
    )

    # Multi-objective studies do not have one automatic `best_trial`.
    # Choose one point from the Pareto front using equal-weight normalized
    # distance to the ideal point (lowest MSE, highest balanced accuracy).
    selected_trial, selection_info = select_pareto_trial(study.best_trials)
    save_optuna_trials(study, trials_dir, outer_fold, selected_trial.number)
    selected = dict(selected_trial.params)

    beta = float(selected.pop("beta"))
    margin_buffer = float(selected.pop("margin_buffer"))

    # Fit final model on all outer-training data using hyperparameters
    # selected entirely within the inner CV.
    final_parameters = {
        "objective": custom_multi_loss(
            y_class_train,
            y_reg_train,
            alpha=1.0,
            beta=beta,
            margin=margin,
            margin_buffer=margin_buffer,
        ),
        "metric": "None",
        "boosting_type": "gbdt",
        "verbosity": -1,
        **selected,
        "bagging_freq": 1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "extra_seed": seed,
        "force_col_wise": True,
        "num_threads": 1,
    }

    final_dataset_label = np.where(
        np.isfinite(y_reg_train),
        y_reg_train,
        0.0,
    )
    final_data = lgb.Dataset(
        X_train,
        label=final_dataset_label,
        free_raw_data=False,
    )

    final_model = lgb.train(
        final_parameters,
        final_data,
        num_boost_round=boost_rounds,
    )

    reported_parameters = {
        **selected,
        "margin_buffer": margin_buffer,
        "beta": beta,
        "alpha": 1.0,
        "num_boost_round": boost_rounds,
    }

    tuning_scores = {
        "best_inner_mse": float(selected_trial.values[0]),
        "best_inner_balanced_accuracy": float(selected_trial.values[1]),
        "selected_trial_number": int(selected_trial.number),
        **selection_info,
    }

    return final_model, reported_parameters, tuning_scores



# ---------------------------------------------------------------------------
# Fixed-beta sensitivity fitting
# ---------------------------------------------------------------------------


def fit_custom_mtl_fixed_beta(
    *,
    X_train: np.ndarray,
    y_reg_train: np.ndarray,
    y_class_train: np.ndarray,
    selected_parameters: dict[str, Any],
    beta: float,
    margin: float,
    seed: int,
) -> lgb.Booster:
    """Refit the final outer-fold model at one fixed beta.

    All non-beta settings are held exactly at the values selected by the
    nested-CV HPO for that outer fold. This makes the sensitivity sweep a
    one-factor-at-a-time analysis of beta.
    """
    parameters = dict(selected_parameters)

    margin_buffer = float(parameters.pop("margin_buffer"))
    parameters.pop("beta", None)
    alpha = float(parameters.pop("alpha", 1.0))
    boost_rounds = int(parameters.pop("num_boost_round"))

    final_parameters = {
        "objective": custom_multi_loss(
            y_class_train,
            y_reg_train,
            alpha=alpha,
            beta=float(beta),
            margin=margin,
            margin_buffer=margin_buffer,
        ),
        "metric": "None",
        "boosting_type": "gbdt",
        "verbosity": -1,
        **parameters,
        "bagging_freq": 1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "extra_seed": seed,
        "force_col_wise": True,
        "num_threads": 1,
    }

    train_data = lgb.Dataset(
        X_train,
        # Finite placeholders only; the custom loss retains the original NaNs.
        label=np.where(np.isfinite(y_reg_train), y_reg_train, 0.0),
        free_raw_data=False,
    )

    return lgb.train(
        final_parameters,
        train_data,
        num_boost_round=boost_rounds,
    )


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------


def prediction_frame(
    *,
    data: LoadedData,
    test_indices: np.ndarray,
    model: str,
    model_task: str,
    predicted_logbb: Optional[np.ndarray],
    predicted_class: np.ndarray,
    score_nonpenetrant: np.ndarray,
    threshold: Optional[float],
) -> pd.DataFrame:
    indices = np.asarray(test_indices, dtype=int)
    logbb_values = (
        np.full(len(indices), np.nan)
        if predicted_logbb is None
        else np.asarray(predicted_logbb, dtype=float)
    )
    result = pd.DataFrame(
        {
            "sample_id": data.sample_id[indices],
            "row_index": indices,
            "smiles": data.smiles[indices],
            "outer_fold": data.outer_fold[indices],
            "true_logBB": data.y_reg[indices],
            "bbb_class": data.y_class[indices],
            "has_logBB": np.isfinite(data.y_reg[indices]),
            "classification_only": ~np.isfinite(data.y_reg[indices]),
            "model": model,
            "model_task": model_task,
            "predicted_logBB": logbb_values,
            "score_nonpenetrant": np.asarray(score_nonpenetrant, dtype=float),
            "predicted_bbb_class": np.asarray(predicted_class, dtype=int),
            "classification_threshold": threshold,
        }
    )
    result["absolute_error"] = np.where(
        result["has_logBB"],
        np.abs(result["true_logBB"] - result["predicted_logBB"]),
        np.nan,
    )
    result["squared_error"] = np.where(
        result["has_logBB"],
        (result["true_logBB"] - result["predicted_logBB"]) ** 2,
        np.nan,
    )
    result["wrong_side"] = (
        result["predicted_bbb_class"].astype(int)
        != result["bbb_class"].astype(int)
    )
    return result



# ---------------------------------------------------------------------------
# Summary tables and outputs
# ---------------------------------------------------------------------------


def evaluate_prediction_group(
    group: pd.DataFrame,
    *,
    model_task: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_task == "regression":
        rows.append(
            {
                "evaluation_subset": "measured_regression",
                "evaluation_task": "regression",
                **regression_metrics(
                    group["true_logBB"].to_numpy(float),
                    group["predicted_logBB"].to_numpy(float),
                    group["bbb_class"].to_numpy(float),
                ),
            }
        )

    subsets = {
        "classification_all": np.ones(len(group), dtype=bool),
        "classification_measured": group["has_logBB"].to_numpy(bool),
        "classification_only": group["classification_only"].to_numpy(bool),
    }
    for subset_name, mask in subsets.items():
        subset = group.loc[mask]
        rows.append(
            {
                "evaluation_subset": subset_name,
                "evaluation_task": "classification",
                **classification_metrics(
                    subset["bbb_class"].to_numpy(float),
                    subset["predicted_bbb_class"].to_numpy(float),
                    subset["score_nonpenetrant"].to_numpy(float),
                ),
            }
        )
    return rows


def build_beta_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-fold, pooled, and mean±SD metrics separately for each beta."""
    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []

    for (model, beta, outer_fold), group in predictions.groupby(
        ["model", "beta", "outer_fold"],
        sort=True,
    ):
        model_task = MODEL_TASKS[model]
        for values in evaluate_prediction_group(group, model_task=model_task):
            fold_rows.append(
                {
                    "model": model,
                    "model_task": model_task,
                    "beta": float(beta),
                    "outer_fold": int(outer_fold),
                    **values,
                }
            )

    for (model, beta), group in predictions.groupby(
        ["model", "beta"],
        sort=True,
    ):
        model_task = MODEL_TASKS[model]
        for values in evaluate_prediction_group(group, model_task=model_task):
            pooled_rows.append(
                {
                    "model": model,
                    "model_task": model_task,
                    "beta": float(beta),
                    **values,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    pooled_metrics = pd.DataFrame(pooled_rows)

    identifier_columns = {
        "model",
        "model_task",
        "beta",
        "outer_fold",
        "evaluation_subset",
        "evaluation_task",
    }
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in identifier_columns
    ]

    summary_rows: list[dict[str, Any]] = []
    for (model, model_task, beta, subset, task), group in fold_metrics.groupby(
        [
            "model",
            "model_task",
            "beta",
            "evaluation_subset",
            "evaluation_task",
        ],
        sort=True,
    ):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            sem = float(sd / np.sqrt(len(values))) if len(values) > 1 else np.nan
            summary_rows.append(
                {
                    "model": model,
                    "model_task": model_task,
                    "beta": float(beta),
                    "evaluation_subset": subset,
                    "evaluation_task": task,
                    "metric": metric,
                    "n_outer_folds": int(len(values)),
                    "mean": mean,
                    "sd": sd,
                    "sem": sem,
                    "mean_sd": (
                        f"{mean:.6f} ± {sd:.6f}"
                        if np.isfinite(sd)
                        else f"{mean:.6f}"
                    ),
                }
            )

    return (
        fold_metrics,
        pooled_metrics,
        pd.DataFrame(summary_rows),
    )


def build_plot_data(mean_sd: pd.DataFrame) -> pd.DataFrame:
    """Create one wide, plotting-ready row per beta."""
    wanted = {
        ("classification_all", "np_recall"): "np_recall",
        ("classification_all", "np_precision"): "np_precision",
        ("classification_all", "np_f1"): "np_f1",
        ("classification_all", "penetrant_specificity"): "penetrant_specificity",
        ("classification_all", "balanced_accuracy"): "balanced_accuracy",
        ("classification_all", "roc_auc"): "roc_auc",
        ("classification_all", "pr_auc_nonpenetrant"): "pr_auc_nonpenetrant",
        ("measured_regression", "mae"): "mae",
        ("measured_regression", "mse"): "mse",
        ("measured_regression", "rmse"): "rmse",
        ("measured_regression", "r2"): "r2",
        ("measured_regression", "pearson_r"): "pearson_r",
        ("measured_regression", "np_mae"): "np_mae",
        ("measured_regression", "penetrant_mae"): "penetrant_mae",
    }

    rows: list[dict[str, Any]] = []
    for beta in sorted(mean_sd["beta"].dropna().unique()):
        beta_rows = mean_sd.loc[np.isclose(mean_sd["beta"], beta)]
        row: dict[str, Any] = {"beta": float(beta)}

        for (subset, metric), output_name in wanted.items():
            match = beta_rows.loc[
                (beta_rows["evaluation_subset"] == subset)
                & (beta_rows["metric"] == metric)
            ]
            if match.empty:
                row[f"{output_name}_mean"] = np.nan
                row[f"{output_name}_sd"] = np.nan
                row[f"{output_name}_sem"] = np.nan
            else:
                first = match.iloc[0]
                row[f"{output_name}_mean"] = float(first["mean"])
                row[f"{output_name}_sd"] = float(first["sd"])
                row[f"{output_name}_sem"] = float(first["sem"])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def write_excel_workbook(
    path: Path,
    *,
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    mean_sd: pd.DataFrame,
    plot_data: pd.DataFrame,
    best_parameters: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            predictions.to_excel(writer, sheet_name="predictions", index=False)
            pooled_metrics.to_excel(writer, sheet_name="pooled_metrics", index=False)
            fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
            mean_sd.to_excel(writer, sheet_name="mean_sd", index=False)
            plot_data.to_excel(writer, sheet_name="plot_data", index=False)
            best_parameters.to_excel(
                writer,
                sheet_name="hpo_parameters",
                index=False,
            )
            pd.DataFrame(
                {
                    "key": list(manifest.keys()),
                    "value": [
                        json.dumps(json_safe(value))
                        if isinstance(value, (dict, list))
                        else value
                        for value in manifest.values()
                    ],
                }
            ).to_excel(writer, sheet_name="run_manifest", index=False)
    except ImportError as exc:
        warnings.warn(
            "openpyxl is not installed, so the XLSX workbook was not written. "
            f"CSV outputs are complete. Original error: {exc}"
        )


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
    resume: bool,
) -> tuple[Path, Path, Path]:
    output_dir = output_dir.resolve()
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        elif not resume:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite to start over, "
                "--resume to continue a compatible checkpoint, or choose a "
                "new directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "fitted_models"
    checkpoint_dir = output_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, model_dir, checkpoint_dir


def save_checkpoint(
    checkpoint_dir: Path,
    prediction_parts: list[pd.DataFrame],
    parameter_rows: list[dict[str, Any]],
) -> None:
    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            checkpoint_dir / "beta_sensitivity_predictions.partial.csv",
            index=False,
        )
    if parameter_rows:
        pd.DataFrame(parameter_rows).to_csv(
            checkpoint_dir / "beta_sensitivity_hpo_parameters.partial.csv",
            index=False,
        )


# ---------------------------------------------------------------------------
# Arguments and main
# ---------------------------------------------------------------------------


IMPLEMENTATION = "training_script_v4_saved_final_parameters_native_missing_v1"
REFERENCE_PARAMETER_KEYS = {
    "alpha", "beta", "margin_buffer", "learning_rate", "num_leaves",
    "max_depth", "min_child_samples", "subsample", "colsample_bytree",
    "num_boost_round",
}


def load_reference_parameters(args: argparse.Namespace):
    """Load script-04 selections without changing or re-optimizing them."""
    if args.reference_parameters is None:
        if args.reference_manifest is not None:
            raise ValueError("--reference-manifest requires --reference-parameters.")
        return {}, None
    path = args.reference_parameters.resolve()
    manifest_path = (args.reference_manifest or path.with_name("run_manifest.json")).resolve()
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Need the reference CSV ({path}) and its run manifest ({manifest_path}).")
    manifest = json.loads(manifest_path.read_text())
    required_manifest = {"input_sha256", "model", "seed", "logbb_threshold", "outer_folds", "mtl_boost_rounds"}
    missing = required_manifest.difference(manifest)
    if missing:
        raise ValueError(f"Reference manifest is missing {sorted(missing)}.")
    if manifest["model"] != MODEL_NAME:
        raise ValueError(f"Expected reference model {MODEL_NAME!r}, got {manifest['model']!r}.")
    if manifest["input_sha256"] != sha256_file(args.input.resolve()):
        raise ValueError("Reference parameters were trained on a different input file. Use the matching dataset and split CSV.")
    for name, source_key in (("seed", "seed"), ("logbb_threshold", "logbb_threshold"), ("mtl_boost_rounds", "mtl_boost_rounds")):
        supplied = getattr(args, name)
        original = manifest[source_key]
        if not np.isfinite(float(original)):
            raise ValueError(f"Reference manifest has invalid {source_key}.")
        if name in {"seed", "mtl_boost_rounds"} and float(original) != int(original):
            raise ValueError(f"Reference manifest needs an integer {source_key}.")
        if supplied is not None and supplied != original:
            raise ValueError(f"--{name.replace('_', '-')}={supplied} conflicts with the final model's saved value {original}.")
        setattr(args, name, float(original) if name == "logbb_threshold" else int(original))
    if args.mtl_boost_rounds < 1:
        raise ValueError("Reference boosting rounds must be positive.")
    frame = pd.read_csv(path)
    if not {"outer_fold", "parameters_json"}.issubset(frame):
        raise ValueError("Reference CSV requires outer_fold and parameters_json columns.")
    if "model" in frame and not frame.model.eq(MODEL_NAME).all():
        raise ValueError("Reference CSV must contain only the final custom_mtl_bab model.")
    folds = pd.to_numeric(frame.outer_fold, errors="raise").to_numpy(float)
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError("Reference outer_fold values must be finite integers.")
    if len(set(folds)) != len(folds):
        raise ValueError("Reference CSV must have exactly one parameter row per outer fold.")
    if set(folds) != set(manifest["outer_folds"]):
        raise ValueError("Reference CSV fold set differs from the reference manifest.")
    rows = {}
    for row in frame.to_dict("records"):
        parameters = json.loads(row["parameters_json"])
        if not isinstance(parameters, dict) or set(parameters) != REFERENCE_PARAMETER_KEYS:
            raise ValueError("Reference parameters must use the script-04 parameter schema; unexpected or missing settings cannot be silently changed.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in parameters.values()):
            raise ValueError("Reference parameter values must be finite numbers.")
        for name in ("num_leaves", "max_depth", "min_child_samples", "num_boost_round"):
            if parameters[name] != int(parameters[name]):
                raise ValueError(f"Reference {name} must be an integer.")
        if parameters["alpha"] != 1.0 or parameters["beta"] < 0 or parameters["margin_buffer"] < 0:
            raise ValueError("Reference loss settings must match script 04: alpha=1, beta>=0, margin_buffer>=0.")
        if parameters["num_boost_round"] != args.mtl_boost_rounds:
            raise ValueError("Reference boosting rounds differ between the CSV and manifest.")
        rows[int(row["outer_fold"])] = row
    provenance = {
        "parameters_file": str(path), "parameters_sha256": sha256_file(path),
        "manifest_file": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }
    return rows, provenance


def validate_reference_data(rows, provenance, data: LoadedData) -> None:
    if provenance is None:
        return
    if set(rows) != set(np.unique(data.outer_fold)):
        raise ValueError("Input dataset and reference parameter CSV have different outer folds.")
    manifest = provenance["manifest"]
    for key, actual in (("n_samples", len(data.frame)), ("n_features_loaded", len(data.feature_columns)),
                        ("n_measured_logbb", int(np.isfinite(data.y_reg).sum()))):
        if key in manifest and manifest[key] != actual:
            raise ValueError(f"Reference {key}={manifest[key]} differs from the loaded input ({actual}).")
    for fold, row in rows.items():
        if "n_selected_features" in row and row["n_selected_features"] != len(data.feature_columns):
            raise ValueError(f"Fold {fold}: reference feature count does not match this script's feature matrix.")


def make_run_signature(args, reference) -> dict[str, Any]:
    """The beta grid may be extended; the dataset and reference cannot change."""
    return {
        "implementation": IMPLEMENTATION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "mode": "saved_final_parameters" if reference else "nested_hpo",
        "reference_parameters_sha256": reference["parameters_sha256"] if reference else None,
        "reference_manifest_sha256": reference["manifest_sha256"] if reference else None,
        "seed": args.seed, "logbb_threshold": args.logbb_threshold,
        "mtl_boost_rounds": args.mtl_boost_rounds,
        "mtl_trials": None if reference else args.mtl_trials,
        "inner_folds": None if reference else args.inner_folds,
        "columns": {key: getattr(args, key) for key in ("target_col", "class_col", "fold_col", "smiles_col", "id_col")},
        "save_sensitivity_models": args.save_sensitivity_models,
        "skip_model_saving": args.skip_model_saving,
        "versions": {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "optuna": getattr(optuna, "__version__", None) if reference is None else None},
    }


def validate_resume_signature(output_dir: Path, signature: dict[str, Any]) -> None:
    path = output_dir / "beta_sweep_implementation.json"
    if not path.is_file():
        raise ValueError("No compatible beta-sweep signature found. Start in a new output directory.")
    previous = json.loads(path.read_text())
    if previous.get("run_signature") != signature:
        raise ValueError("Incompatible beta-sweep checkpoint: code, input, reference parameters, or settings differ. Start in a new output directory; older 30-trial sweeps cannot be resumed in saved-parameter mode.")


DEFAULT_BETA_VALUES = (
    "0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,"
    "12.5,15,17.5,20"
)


def parse_beta_values(value: str) -> list[float]:
    try:
        betas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--beta-values must be a comma-separated list of numbers."
        ) from exc

    if not betas:
        raise argparse.ArgumentTypeError("--beta-values cannot be empty.")
    if any((not np.isfinite(beta)) or beta < 0 for beta in betas):
        raise argparse.ArgumentTypeError(
            "Every beta value must be finite and >= 0."
        )

    # Deduplicate while preserving numerical order.
    betas = sorted(set(betas))
    return betas


def beta_tag(beta: float) -> str:
    text = f"{beta:g}"
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("custom_mtl_beta_sensitivity_results"),
    )
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--class-col", default=DEFAULT_CLASS_COL)
    parser.add_argument("--fold-col", default=DEFAULT_FOLD_COL)
    parser.add_argument("--smiles-col", default=DEFAULT_SMILES_COL)
    parser.add_argument("--id-col", default=DEFAULT_ID_COL)
    parser.add_argument("--reference-parameters", type=Path, default=None,
                        help="Final script-04 custom_mtl_best_parameters.csv. When supplied, skip HPO and change only beta.")
    parser.add_argument("--reference-manifest", type=Path, default=None,
                        help="Original run_manifest.json; defaults to the file next to --reference-parameters.")
    parser.add_argument("--seed", type=int, default=None, help="HPO default: 15. Saved mode uses the original seed; conflicting overrides are rejected.")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--logbb-threshold", type=float, default=None, help="HPO default: -1. Saved mode uses the original margin.")
    parser.add_argument("--mtl-trials", type=int, default=30)
    parser.add_argument("--mtl-boost-rounds", type=int, default=None, help="HPO default: 1000. Saved mode uses the original boosting rounds.")
    parser.add_argument(
        "--beta-values",
        type=parse_beta_values,
        default=parse_beta_values(DEFAULT_BETA_VALUES),
        help=(
            "Comma-separated fixed beta values used for the sensitivity sweep. "
            "The default is dense from 0 to 10 and coarser from 10 to 20."
        ),
    )
    parser.add_argument(
        "--save-sensitivity-models",
        action="store_true",
        help=(
            "Save every fixed-beta outer-fold model. Off by default because "
            "the full sweep can create many model files."
        ),
    )
    parser.add_argument("--skip-model-saving", action="store_true")
    parser.add_argument("--allow-label-disagreement", action="store_true")
    parser.add_argument(
        "--allow-high-target-correlation",
        action="store_true",
        help=(
            "Allow descriptors with |Pearson r| >= 0.98 against measured logBB."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a compatible run produced by this version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta_values = [float(beta) for beta in args.beta_values]
    reference_rows, reference = load_reference_parameters(args)
    if reference is None:
        args.seed = 15 if args.seed is None else args.seed
        args.logbb_threshold = -1.0 if args.logbb_threshold is None else args.logbb_threshold
        args.mtl_boost_rounds = 1000 if args.mtl_boost_rounds is None else args.mtl_boost_rounds
        if optuna is None:
            raise ImportError("Optuna is required for HPO. Supply --reference-parameters to use saved final selections without Optuna.")
    else:
        print(f"Using final parameters from {reference['parameters_file']}; skipping HPO.")
    data = load_master_data(args)
    validate_reference_data(reference_rows, reference, data)
    signature = make_run_signature(args, reference)
    if args.resume:
        validate_resume_signature(args.output_dir.resolve(), signature)
    protected_sources = [args.input.resolve()]
    if reference:
        protected_sources += [Path(reference['parameters_file']), Path(reference['manifest_file'])]
    if args.overwrite and any(path.is_relative_to(args.output_dir.resolve()) for path in protected_sources):
        raise ValueError("--overwrite output directory contains a source input/reference file. Use a separate output directory.")

    output_dir, model_dir, checkpoint_dir = prepare_output_directory(
        args.output_dir,
        args.overwrite,
        args.resume,
    )
    version_path = output_dir / "beta_sweep_implementation.json"
    version_path.write_text(json.dumps({"implementation": IMPLEMENTATION, "run_signature": signature}, indent=2))

    correlation_check = check_feature_target_correlation(data.X, data.y_reg)
    correlation_check.to_csv(
        output_dir / "feature_target_correlation_check.csv",
        index=False,
    )

    suspicious = correlation_check.loc[
        correlation_check["abs_pearson_r"] >= 0.98
    ]
    if len(suspicious) and not args.allow_high_target_correlation:
        examples = suspicious[["feature", "pearson_r"]].head(10)
        raise RuntimeError(
            "Potential target leakage detected. At least one descriptor has "
            "|Pearson r| >= 0.98 against measured logBB:\n"
            + examples.to_string(index=False)
            + "\nUse --allow-high-target-correlation only after checking."
        )

    outer_folds = sorted(np.unique(data.outer_fold).tolist())

    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    pred_ckpt = checkpoint_dir / "beta_sensitivity_predictions.partial.csv"
    param_ckpt = checkpoint_dir / "beta_sensitivity_hpo_parameters.partial.csv"

    if pred_ckpt.exists():
        existing_predictions = pd.read_csv(pred_ckpt)
        if "beta" in existing_predictions.columns:
            # Keep only beta values requested in the current run.
            keep = np.zeros(len(existing_predictions), dtype=bool)
            for beta in beta_values:
                keep |= np.isclose(
                    existing_predictions["beta"].to_numpy(float),
                    beta,
                )
            existing_predictions = existing_predictions.loc[keep].copy()
            if len(existing_predictions):
                prediction_parts = [existing_predictions]

    if param_ckpt.exists():
        parameter_rows = pd.read_csv(param_ckpt).to_dict("records")
        saved_folds = [int(row["outer_fold"]) for row in parameter_rows]
        if len(saved_folds) != len(set(saved_folds)) or not set(saved_folds).issubset(outer_folds):
            raise ValueError("Checkpoint parameter rows contain duplicate or unexpected outer folds.")
        if reference:
            for row in parameter_rows:
                original = reference_rows[int(row["outer_fold"])]
                if json.loads(row["parameters_json"]) != json.loads(original["parameters_json"]):
                    raise ValueError("Checkpoint parameters disagree with the saved final model. Use a new output directory.")

    existing_pairs: set[tuple[int, float]] = set()
    if prediction_parts:
        existing = prediction_parts[0]
        for (fold, beta), group in existing.groupby(
            ["outer_fold", "beta"],
            sort=False,
        ):
            # Only call the pair complete if every row in that outer test fold
            # is present exactly once.
            expected_indices = np.flatnonzero(data.outer_fold == int(fold))
            expected_rows = len(expected_indices)
            observed_indices = pd.to_numeric(group["row_index"], errors="raise").to_numpy(float)
            if not np.isfinite(observed_indices).all() or not np.equal(observed_indices, np.floor(observed_indices)).all():
                raise ValueError("Checkpoint contains invalid row indices.")
            if group["row_index"].duplicated().any() or not set(observed_indices).issubset(set(expected_indices)):
                raise ValueError("Checkpoint row indices do not match their outer fold.")
            order = observed_indices.astype(int)
            if not np.array_equal(pd.to_numeric(group["bbb_class"]).to_numpy(int), data.y_class[order]):
                raise ValueError("Checkpoint labels do not match the input dataset.")
            if not np.isfinite(pd.to_numeric(group["predicted_logBB"], errors="raise")).all():
                raise ValueError("Checkpoint contains nonfinite predictions.")
            if (
                len(group) == expected_rows
                and not group["row_index"].duplicated().any()
            ):
                existing_pairs.add((int(fold), float(beta)))

    print("Sensitivity beta values:")
    print(beta_values)
    if existing_pairs:
        print(f"Resuming with {len(existing_pairs)} completed fold/beta pairs.")

    for fold_position, outer_fold in enumerate(outer_folds):
        print(f"\nouter fold: {outer_fold}")

        train_mask = data.outer_fold != outer_fold
        test_mask = data.outer_fold == outer_fold
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)

        X_train_df = data.X.iloc[train_indices]
        X_test_df = data.X.iloc[test_indices]
        seed = args.seed + 1000 * fold_position

        X_train, X_test, feature_names = prepare_fold_features(
            X_train_df,
            X_test_df,
        )
        y_reg_train = data.y_reg[train_indices]
        y_class_train = data.y_class[train_indices]

        # Reuse HPO results when resuming this fold, if available.
        existing_parameter_row = next(
            (
                row
                for row in parameter_rows
                if int(row["outer_fold"]) == int(outer_fold)
            ),
            None,
        )

        selected_model: Optional[lgb.Booster] = None

        if existing_parameter_row is not None:
            parameters = json.loads(existing_parameter_row["parameters_json"])
            hpo_selected_beta = float(parameters["beta"])
            print(
                "  reusing saved nested-CV parameters; "
                f"HPO-selected beta={hpo_selected_beta:g}"
            )
        elif reference is not None:
            source_row = reference_rows[int(outer_fold)]
            parameters = json.loads(source_row["parameters_json"])
            hpo_selected_beta = float(parameters["beta"])
            parameter_rows.append({
                **source_row,
                "model": MODEL_NAME,
                "outer_fold": int(outer_fold),
                "hpo_selected_beta": hpo_selected_beta,
                "parameters_json": json.dumps(json_safe(parameters), sort_keys=True),
                "n_selected_features": len(feature_names),
                "training_location": "saved_final_parameters",
                "reference_parameters_file": reference["parameters_file"],
                "reference_parameters_sha256": reference["parameters_sha256"],
            })
            save_checkpoint(checkpoint_dir, prediction_parts, parameter_rows)
        else:
            inner_cv = saved_inner_splits(
                data.frame,
                outer_fold=outer_fold,
                fit_indices=train_indices,
            )
            if inner_cv is None:
                inner_cv = classification_inner_splits(
                    y_class_train,
                    n_splits=args.inner_folds,
                    seed=seed,
                )

            selected_model, parameters, tuning_scores = tune_and_fit_custom_mtl(
                X_train=X_train,
                y_reg_train=y_reg_train,
                y_class_train=y_class_train,
                inner_splits=inner_cv,
                seed=seed,
                n_trials=args.mtl_trials,
                boost_rounds=args.mtl_boost_rounds,
                margin=args.logbb_threshold,
                trials_dir=output_dir / "optuna_trials",
                outer_fold=int(outer_fold),
            )

            hpo_selected_beta = float(parameters["beta"])

            parameter_row = {
                "model": MODEL_NAME,
                "outer_fold": outer_fold,
                "hpo_selected_beta": hpo_selected_beta,
                **tuning_scores,
                "parameters_json": json.dumps(
                    json_safe(parameters),
                    sort_keys=True,
                ),
                "n_selected_features": len(feature_names),
                "training_location": "this_script",
            }
            parameter_rows.append(parameter_row)

            # Save the HPO result immediately so an interrupted beta sweep does
            # not require repeating the inner-CV search on resume.
            save_checkpoint(
                checkpoint_dir,
                prediction_parts,
                parameter_rows,
            )

            if not args.skip_model_saving:
                selected_model.save_model(
                    str(
                        model_dir
                        / f"{MODEL_NAME}_hpo_selected_outer_{outer_fold}.txt"
                    )
                )
                joblib.dump(
                    {
                        "feature_names": feature_names,
                        "parameters": parameters,
                        "threshold": args.logbb_threshold,
                        "hpo_selected_beta": hpo_selected_beta,
                    },
                    model_dir
                    / f"{MODEL_NAME}_hpo_selected_outer_{outer_fold}_preprocessor.joblib",
                    compress=3,
                )

        print(f"  HPO-selected beta for this fold: {hpo_selected_beta:g}")

        for beta in beta_values:
            pair_exists = any(
                fold == int(outer_fold) and np.isclose(saved_beta, beta)
                for fold, saved_beta in existing_pairs
            )
            if pair_exists:
                print(f"  beta={beta:g}: already completed; skipping")
                continue

            print(f"  beta={beta:g}")

            # Reuse the already fitted HPO-selected model only when its beta is
            # exactly one of the requested sensitivity values in this fresh run.
            if (
                selected_model is not None
                and np.isclose(beta, hpo_selected_beta)
            ):
                model = selected_model
            else:
                model = fit_custom_mtl_fixed_beta(
                    X_train=X_train,
                    y_reg_train=y_reg_train,
                    y_class_train=y_class_train,
                    selected_parameters=parameters,
                    beta=beta,
                    margin=args.logbb_threshold,
                    seed=seed,
                )

            predicted_logbb = np.asarray(
                model.predict(X_test),
                dtype=float,
            )
            predicted_class, score_nonpenetrant = (
                regression_to_classification(
                    predicted_logbb,
                    args.logbb_threshold,
                )
            )

            fold_predictions = prediction_frame(
                data=data,
                test_indices=test_indices,
                model=MODEL_NAME,
                model_task="regression",
                predicted_logbb=predicted_logbb,
                predicted_class=predicted_class,
                score_nonpenetrant=score_nonpenetrant,
                threshold=args.logbb_threshold,
            )
            fold_predictions["beta"] = float(beta)
            fold_predictions["hpo_selected_beta"] = hpo_selected_beta
            fold_predictions["is_hpo_selected_beta"] = bool(
                np.isclose(beta, hpo_selected_beta)
            )
            prediction_parts.append(fold_predictions)
            existing_pairs.add((int(outer_fold), float(beta)))

            if args.save_sensitivity_models:
                tag = beta_tag(beta)
                model.save_model(
                    str(
                        model_dir
                        / f"{MODEL_NAME}_beta_{tag}_outer_{outer_fold}.txt"
                    )
                )
                joblib.dump(
                    {
                        "feature_names": feature_names,
                        "parameters": {
                            **parameters,
                            "beta": float(beta),
                        },
                        "threshold": args.logbb_threshold,
                        "hpo_selected_beta": hpo_selected_beta,
                    },
                    model_dir
                    / f"{MODEL_NAME}_beta_{tag}_outer_{outer_fold}_preprocessor.joblib",
                    compress=3,
                )

            save_checkpoint(
                checkpoint_dir,
                prediction_parts,
                parameter_rows,
            )

    if not prediction_parts:
        raise RuntimeError("No predictions were generated.")

    predictions = pd.concat(prediction_parts, ignore_index=True)

    # If a run was resumed, protect against accidental duplicate checkpoint rows.
    predictions = predictions.drop_duplicates(
        subset=["row_index", "beta"],
        keep="last",
    )
    predictions = predictions.sort_values(
        ["beta", "row_index"],
        kind="stable",
    ).reset_index(drop=True)

    expected = len(data.frame) * len(beta_values)
    if len(predictions) != expected:
        counts = (
            predictions.groupby("beta")["row_index"]
            .nunique()
            .sort_index()
            .to_dict()
        )
        raise RuntimeError(
            "Beta sensitivity predictions are incomplete. "
            f"Expected {expected} rows but found {len(predictions)}. "
            f"Unique rows per beta: {counts}"
        )

    if predictions.duplicated(subset=["row_index", "beta"]).any():
        raise RuntimeError(
            "Duplicate row_index/beta combinations remain in predictions."
        )

    fold_metrics, pooled_metrics, mean_sd = build_beta_metric_tables(
        predictions
    )
    plot_data = build_plot_data(mean_sd)

    best_parameters = pd.DataFrame(parameter_rows).sort_values(
        ["outer_fold"],
        kind="stable",
    )

    manifest = {
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "output_directory": str(output_dir),
        "n_samples": int(len(data.frame)),
        "n_measured_logbb": int(np.sum(np.isfinite(data.y_reg))),
        "n_classification_only": int(np.sum(~np.isfinite(data.y_reg))),
        "n_features_loaded": int(len(data.feature_columns)),
        "descriptor_preprocessing": "no imputation; native LightGBM missing-value handling",
        "outer_folds": outer_folds,
        "model": MODEL_NAME,
        "class_encoding": {"0": "BBB penetrant", "1": "BBB non-penetrant"},
        "logbb_threshold": args.logbb_threshold,
        "seed": args.seed,
        "parameter_source": "saved_final_parameters" if reference else "nested_hpo",
        "reference": reference,
        "run_signature": signature,
        "inner_folds": reference["manifest"].get("inner_folds") if reference else args.inner_folds,
        "mtl_trials": 0 if reference else args.mtl_trials,
        "reference_mtl_trials": reference["manifest"].get("mtl_trials") if reference else args.mtl_trials,
        "mtl_boost_rounds": args.mtl_boost_rounds,
        "beta_sensitivity_values": beta_values,
        "beta_sensitivity_design": (
            "For each outer fold, load the final model's saved parameters without rerunning HPO. "
            "Hold all non-beta settings, the original seed, margin, descriptor order, and boosting "
            "rounds fixed; refit on the same outer-training partition at each requested beta and "
            "evaluate only on its held-out outer fold."
        ) if reference else (
            "For each outer fold, nested-CV HPO is run with the requested settings. "
            "The selected non-beta hyperparameters, including "
            "margin_buffer, are then held fixed while beta alone is replaced "
            "by each prespecified sensitivity value and the model is retrained "
            "on the full outer-training fold. Performance is evaluated only on "
            "the held-out outer fold."
        ),
        "hyperparameter_optimization": {
            "performed_in_this_run": reference is None,
            "type": "multi_objective",
            "regression_objective": (
                "mean inner-fold MSE on measured-logBB rows; minimize"
            ),
            "classification_objective": (
                "mean inner-fold balanced accuracy on all rows; maximize"
            ),
            "classification_threshold": args.logbb_threshold,
            "beta_search_in_reference_hpo": "0.5 to 5.0 in steps of 0.25",
            "pareto_selection": (
                "equal-weight Euclidean distance to the ideal point after "
                "min-max normalization across the Pareto front"
            ),
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "optuna": getattr(optuna, "__version__", None),
            "joblib": joblib.__version__,
        },
    }

    predictions_path = (
        output_dir / "custom_mtl_beta_sensitivity_predictions.csv"
    )
    pooled_path = (
        output_dir / "custom_mtl_beta_sensitivity_pooled_metrics.csv"
    )
    fold_path = (
        output_dir / "custom_mtl_beta_sensitivity_fold_metrics.csv"
    )
    mean_sd_path = (
        output_dir / "custom_mtl_beta_sensitivity_mean_sd.csv"
    )
    plot_data_path = (
        output_dir / "custom_mtl_beta_sensitivity_plot_data.csv"
    )
    parameters_path = (
        output_dir / "custom_mtl_beta_sensitivity_hpo_parameters.csv"
    )

    predictions.to_csv(predictions_path, index=False)
    pooled_metrics.to_csv(pooled_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    mean_sd.to_csv(mean_sd_path, index=False)
    plot_data.to_csv(plot_data_path, index=False)
    best_parameters.to_csv(parameters_path, index=False)

    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2)
    )

    write_excel_workbook(
        output_dir / "custom_mtl_beta_sensitivity_results.xlsx",
        predictions=predictions,
        pooled_metrics=pooled_metrics,
        fold_metrics=fold_metrics,
        mean_sd=mean_sd,
        plot_data=plot_data,
        best_parameters=best_parameters,
        manifest=manifest,
    )

    print("\nCompleted successfully.")
    print("Predictions:", predictions_path)
    print("Fold metrics:", fold_path)
    print("Mean ± SD:", mean_sd_path)
    print("Plot-ready data:", plot_data_path)
    print("HPO parameters:", parameters_path)
    if args.save_sensitivity_models:
        print("Sensitivity models:", model_dir)


if __name__ == "__main__":
    main()
