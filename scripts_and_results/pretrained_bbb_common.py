#!/usr/bin/env python3
"""Shared training/evaluation code for frozen pretrained BBB experiments."""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset


TAU = -1.0


def prepare_output_directory(output_dir, result_files, result_dirs, overwrite=False):
    """Prevent a rerun from mixing old checkpoints with newly written tables.

    Explicit overwrite replaces only this trainer's known outputs. Embedding
    caches and unrelated files are preserved.
    """
    output_dir = Path(output_dir)
    names = list(result_files) + list(result_dirs)
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Training output names must be simple basenames.")
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Training results already exist in {output_dir}. Use a new --output-dir or --overwrite.")
    if overwrite:
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_bbb_dataframe(path: Path) -> pd.DataFrame:
    return validate_bbb_dataframe(pd.read_csv(path, low_memory=False))


def parse_bool_column(series: pd.Series) -> np.ndarray:
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{series.name} must contain explicit True/False or 1/0 values.")
    return parsed.to_numpy(dtype=bool)


def validate_bbb_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate without dropping rows, imputing targets, or changing class labels."""
    df = df.copy().reset_index(drop=True)
    if not df.columns.is_unique:
        raise ValueError("Input columns must have unique names, including after any column renaming.")
    required = {"smiles", "bbb_class", "outer_fold", "logBB"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The input dataset is empty.")
    df["bbb_class"] = pd.to_numeric(df["bbb_class"], errors="raise")
    df["logBB"] = pd.to_numeric(df["logBB"], errors="raise")
    folds = pd.to_numeric(df["outer_fold"], errors="raise")
    if not np.isfinite(folds).all() or (folds != np.floor(folds)).any() or folds.nunique() < 2:
        raise ValueError("outer_fold must contain at least two finite integer fold labels.")
    df["outer_fold"] = folds.astype(int)
    if df["smiles"].isna().any() or df["smiles"].astype(str).str.strip().eq("").any() or df["bbb_class"].isna().any():
        raise ValueError("SMILES and bbb_class must be present for every row.")
    bad_labels = set(df["bbb_class"].unique()).difference({0, 1})
    if bad_labels:
        raise ValueError(f"bbb_class must use 0=penetrant, 1=non-penetrant; found {bad_labels}")
    df["bbb_class"] = df["bbb_class"].astype(int)
    if np.isinf(df["logBB"].to_numpy(float)).any():
        raise ValueError("logBB may be missing but cannot be infinite.")
    measured = df["logBB"].notna().to_numpy()
    for column in ("has_logbb", "has_logBB", "logbb_is_measured", "classification_only"):
        if column in df:
            supplied = parse_bool_column(df[column])
            expected = ~measured if column == "classification_only" else measured
            if not np.array_equal(supplied, expected):
                raise ValueError(f"{column} disagrees with measured logBB availability; fix the input explicitly.")
    if not measured.any():
        raise ValueError("Training requires measured logBB values.")
    if ((df.loc[measured, "logBB"] < TAU).astype(int) != df.loc[measured, "bbb_class"]).any():
        raise ValueError("Measured logBB and bbb_class disagree at the dataset threshold -1.")
    df["_has_logbb"] = measured
    if "sample_id" not in df:
        df["sample_id"] = np.arange(len(df))
    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique and nonmissing.")
    if "row_index" in df:
        rows = pd.to_numeric(df["row_index"], errors="raise")
        if rows.isna().any() or not np.isfinite(rows).all() or (rows != np.floor(rows)).any() or rows.duplicated().any():
            raise ValueError("row_index must contain unique, nonmissing finite integers.")
        df["row_index"] = rows.astype(np.int64)
    else:
        df["row_index"] = np.arange(len(df), dtype=int)
    # Export identity is independent of positional indices used for CV/slicing.
    df["_source_row"] = df["row_index"].to_numpy()
    return df


def make_inner_splits(classes, has_logbb, inner_folds, seed, assignments=None):
    """Return relative indices; saved assignments take precedence over random CV."""
    classes, has_logbb = np.asarray(classes), np.asarray(has_logbb, dtype=bool)
    if inner_folds < 2:
        raise ValueError("At least two inner folds are required.")
    if assignments is not None:
        values = pd.to_numeric(pd.Series(assignments), errors="raise").to_numpy(float)
        if len(values) != len(classes) or not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError("Saved inner-fold assignments must be complete, finite integers.")
        if len(np.unique(values)) != inner_folds:
            raise ValueError("Saved inner-fold count differs from --inner-folds.")
        splits = [(np.flatnonzero(values != f), np.flatnonzero(values == f)) for f in np.unique(values)]
    else:
        strata = np.array([f"{c}_{int(h)}" for c, h in zip(classes, has_logbb)])
        if pd.Series(strata).value_counts().min() < inner_folds:
            raise ValueError("Too few rows per class/measured stratum for the requested inner folds.")
        splits = list(StratifiedKFold(inner_folds, shuffle=True, random_state=seed).split(classes, strata))
    for tr, va in splits:
        for indices in (tr, va):
            if not len(indices) or not has_logbb[indices].any() or len(np.unique(classes[indices])) != 2:
                raise ValueError("Every inner training/validation partition needs measured logBB and both classes.")
    return splits


def dataframe_inner_splits(df, fold, indices, inner_folds, seed):
    column = f"inner_fold_outer_{fold}"
    if column not in df and any(str(c).startswith("inner_fold_outer_") for c in df):
        raise ValueError(f"Missing {column}; refusing to replace a partially saved fold design with random splits.")
    assignments = df.iloc[indices][column].to_numpy() if column in df else None
    return make_inner_splits(df.iloc[indices]["bbb_class"].to_numpy(int),
                             df.iloc[indices]["_has_logbb"].to_numpy(bool), inner_folds, seed, assignments)


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Standardization requires a nonempty finite embedding matrix.")
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32)


def normalization_reference(
    x: np.ndarray, has_logbb: np.ndarray, model_kind: str
) -> np.ndarray:
    """Fit regression preprocessing only on rows that actually supervise it."""
    return x[has_logbb] if model_kind == "regression" else x


class LinearTransferHead(nn.Module):
    """Dropout plus a single linear output, matching the simplest published heads."""

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(x)).squeeze(-1)


def squared_mtl_loss(output, classes, logbb, has_logbb, beta, margin_buffer,
                     *, alpha=1.0, tau=TAU, class_counts=None):
    """Exact LightGBM scalar objective (no component-wise or global rescaling).

    alpha/2 * sum_measured(error^2) + beta * sum_c(sum_class_c(violation^2)/N_c).
    For accumulated graph batches, N_c is from the entire fitting partition.
    Missing targets are indexed out before subtraction; they never supervise regression.
    """
    if not all(math.isfinite(float(v)) for v in (alpha, beta, tau, margin_buffer)) or alpha < 0 or beta < 0 or margin_buffer < 0:
        raise ValueError("Loss parameters must be finite with nonnegative weights and buffer.")
    if not (output.ndim == classes.ndim == logbb.ndim == has_logbb.ndim == 1 and
            len(output) == len(classes) == len(logbb) == len(has_logbb)):
        raise ValueError("Loss inputs must be matching one-dimensional tensors.")
    if not torch.all((classes == 0) | (classes == 1)):
        raise ValueError("Classes must be 0 or 1.")
    if not torch.isfinite(output).all() or not torch.isfinite(logbb[has_logbb]).all():
        raise FloatingPointError("Nonfinite prediction or measured regression target.")
    regression = 0.5 * alpha * (output[has_logbb] - logbb[has_logbb]).square().sum()
    counts = (torch.bincount(classes.long(), minlength=2).to(output)
              if class_counts is None else torch.as_tensor(class_counts, device=output.device, dtype=output.dtype))
    if counts.shape != (2,) or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("class_counts must contain two finite nonnegative counts.")
    if torch.any(counts < torch.bincount(classes.long(), minlength=2).to(output)):
        raise ValueError("Reference class counts cannot be smaller than the current batch counts.")
    violation = torch.where(classes == 1, F.relu(output - (tau - margin_buffer)),
                            F.relu((tau + margin_buffer) - output))
    classification = (violation.square() / counts.clamp_min(1)[classes.long()]).sum()
    return regression + beta * classification


def compute_loss(
    output: torch.Tensor,
    classes: torch.Tensor,
    logbb: torch.Tensor,
    has_logbb: torch.Tensor,
    model_kind: str,
    beta: float,
    margin_buffer: float,
) -> torch.Tensor:
    if model_kind == "regression":
        if not has_logbb.any():
            raise RuntimeError("A regression minibatch contained no measured logBB values.")
        return F.mse_loss(output[has_logbb], logbb[has_logbb])

    if model_kind != "mtl":
        raise ValueError(f"Unknown model kind: {model_kind}")
    return squared_mtl_loss(output, classes, logbb, has_logbb, beta, margin_buffer)


def make_loader(
    x: np.ndarray,
    classes: np.ndarray,
    logbb: np.ndarray,
    has_logbb: np.ndarray,
    batch_size: int,
    shuffle: bool,
    model_kind: str,
) -> DataLoader:
    if model_kind == "regression":
        x, classes, logbb, has_logbb = x[has_logbb], classes[has_logbb], logbb[has_logbb], has_logbb[has_logbb]
    if not len(x):
        raise ValueError("No training/evaluation rows for this objective.")
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(classes.astype(np.int64)),
        torch.from_numpy(logbb.astype(np.float32)),
        torch.from_numpy(has_logbb.astype(bool)),
    )
    # Linear heads are small: one exact full-partition objective per update.
    # batch_size controls prediction memory only, not the loss normalization.
    effective_batch = len(dataset)
    return DataLoader(dataset, batch_size=effective_batch, shuffle=shuffle)


@torch.no_grad()
def predict_raw(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size)
    for xb in loader:
        out.append(model(xb.to(device)).detach().cpu().numpy())
    prediction = np.concatenate(out).astype(float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite frozen-head predictions.")
    return prediction


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float


def fit_head(
    x_train: np.ndarray,
    cls_train: np.ndarray,
    logbb_train: np.ndarray,
    has_train: np.ndarray,
    x_valid: Optional[np.ndarray],
    cls_valid: Optional[np.ndarray],
    logbb_valid: Optional[np.ndarray],
    has_valid: Optional[np.ndarray],
    model_kind: str,
    params: Dict[str, Any],
    beta: float,
    margin_buffer: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    fixed_epochs: Optional[int] = None,
) -> FitResult:
    if max_epochs < 1 or batch_size < 1 or (fixed_epochs is not None and fixed_epochs < 1):
        raise ValueError("Epochs and batch size must be positive.")
    if x_valid is not None and patience < 1:
        raise ValueError("Early-stopping patience must be positive.")
    seed_everything(seed)
    model = LinearTransferHead(x_train.shape[1], float(params["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    train_loader = make_loader(
        x_train, cls_train, logbb_train, has_train, batch_size, True, model_kind
    )
    valid_loader = None
    if x_valid is not None:
        valid_loader = make_loader(
            x_valid, cls_valid, logbb_valid, has_valid, batch_size, False, model_kind
        )

    epochs_to_run = int(max_epochs if fixed_epochs is None else fixed_epochs)
    best_loss = float("inf")
    best_epoch = epochs_to_run
    best_state = None
    stale = 0
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        for xb, cb, yb, hb in train_loader:
            xb, cb, yb, hb = xb.to(device), cb.to(device), yb.to(device), hb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(xb), cb, yb, hb, model_kind, beta, margin_buffer)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite head training loss.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite head gradients.")
            optimizer.step()

        if valid_loader is None:
            continue
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, cb, yb, hb in valid_loader:
                xb, cb, yb, hb = xb.to(device), cb.to(device), yb.to(device), hb.to(device)
                losses.append(float(compute_loss(
                    model(xb), cb, yb, hb, model_kind, beta, margin_buffer
                ).item()))
        val_loss = float(np.mean(losses))
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite head validation loss.")
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model=model, best_epoch=best_epoch, best_val_loss=best_loss)


def candidate_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
    }


def select_pareto_trial(study):
    candidates = sorted((t for t in study.best_trials if np.isfinite(t.values).all()), key=lambda t: t.number)
    if not candidates:
        raise RuntimeError("No finite Pareto-optimal trial.")
    values = np.array([t.values for t in candidates])
    distance = values.copy()
    distance[:, 1] *= -1
    span = np.ptp(distance, axis=0)
    normalized = (distance - distance.min(axis=0)) / np.where(span > 0, span, 1)
    return candidates[int(np.argmin(np.linalg.norm(normalized, axis=1)))]


def tune_head(
    x: np.ndarray,
    classes: np.ndarray,
    logbb: np.ndarray,
    has_logbb: np.ndarray,
    model_kind: str,
    beta: float,
    margin_buffer: float,
    trials: int,
    inner_folds: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    splits=None,
    tune_loss: bool = False,
    audit_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], int]:
    splits = make_inner_splits(classes, has_logbb, inner_folds, seed) if splits is None else splits

    def objective(trial: optuna.Trial):
        params = candidate_params(trial)
        trial_beta = trial.suggest_float("beta", 0.5, 5.0, step=0.25) if tune_loss and model_kind == "mtl" else beta
        trial_buffer = trial.suggest_float("margin_buffer", 0.15, 0.35, step=0.02) if tune_loss and model_kind == "mtl" else margin_buffer
        mse_values, ba_values = [], []
        epochs = []
        for inner, (tr, va) in enumerate(splits):
            mean, scale = standardize_fit(
                normalization_reference(x[tr], has_logbb[tr], model_kind)
            )
            result = fit_head(
                standardize_apply(x[tr], mean, scale), classes[tr], logbb[tr], has_logbb[tr],
                standardize_apply(x[va], mean, scale), classes[va], logbb[va], has_logbb[va],
                model_kind, params, trial_beta, trial_buffer, max_epochs, patience,
                batch_size, seed + inner, device,
            )
            prediction = predict_raw(result.model, standardize_apply(x[va], mean, scale), device, batch_size)
            mse_values.append(mean_squared_error(logbb[va][has_logbb[va]], prediction[has_logbb[va]]))
            ba_values.append(balanced_accuracy_score(classes[va], (prediction < TAU).astype(int)))
            epochs.append(result.best_epoch)
        trial.set_user_attr("median_best_epoch", int(np.median(epochs)))
        trial.set_user_attr("inner_best_epochs", epochs)
        return float(np.mean(mse_values)), float(np.mean(ba_values))

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    selected = select_pareto_trial(study)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        table = study.trials_dataframe()
        table["selected"] = table["number"] == selected.number
        table.to_csv(audit_path, index=False)
    return dict(selected.params), int(selected.user_attrs["median_best_epoch"])


def oof_scores_for_threshold(
    x: np.ndarray,
    classes: np.ndarray,
    logbb: np.ndarray,
    has_logbb: np.ndarray,
    model_kind: str,
    params: Dict[str, Any],
    final_epochs: int,
    beta: float,
    margin_buffer: float,
    inner_folds: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    splits=None,
) -> np.ndarray:
    scores = np.full(len(x), np.nan, dtype=float)
    splits = make_inner_splits(classes, has_logbb, inner_folds, seed) if splits is None else splits
    for inner, (tr, va) in enumerate(splits):
        mean, scale = standardize_fit(
            normalization_reference(x[tr], has_logbb[tr], model_kind)
        )
        fit = fit_head(
            standardize_apply(x[tr], mean, scale), classes[tr], logbb[tr], has_logbb[tr],
            None, None, None, None, model_kind, params, beta, margin_buffer,
            final_epochs, 0, batch_size, seed + 100 + inner, device,
            fixed_epochs=final_epochs,
        )
        raw = predict_raw(fit.model, standardize_apply(x[va], mean, scale), device, batch_size)
        scores[va] = -raw
    if not np.isfinite(scores).all():
        raise FloatingPointError("Inner out-of-fold scores are incomplete or nonfinite.")
    return scores


def choose_balanced_accuracy_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    if not len(score) or not np.isfinite(score).all() or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection needs finite scores and both classes.")
    values = np.unique(score)
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2
    candidates = np.unique(np.r_[np.nextafter(values[0], -np.inf), midpoints,
                                  np.nextafter(values[-1], np.inf), -TAU])
    best = (float("-inf"), float("-inf"), float("-inf"))
    selected = -TAU
    for threshold in candidates:
        ba = balanced_accuracy_score(y_true, (score > threshold).astype(int))
        key = (float(ba), -abs(float(threshold) + TAU), -float(threshold))
        if key > best:
            best, selected = key, float(threshold)
    return selected


def safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float, pred=None) -> Dict[str, Any]:
    pred = (score > threshold).astype(int) if pred is None else np.asarray(pred, dtype=int)
    both_classes = len(np.unique(y)) == 2
    return {
        "n_evaluable": int(len(y)),
        "accuracy": safe_metric(accuracy_score, y, pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y, pred) if both_classes else np.nan,
        "np_recall": safe_metric(recall_score, y, pred, pos_label=1, zero_division=0),
        "np_precision": safe_metric(precision_score, y, pred, pos_label=1, zero_division=0),
        "np_f1": safe_metric(f1_score, y, pred, pos_label=1, zero_division=0),
        "penetrant_specificity": float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else np.nan,
        "roc_auc": safe_metric(roc_auc_score, y, score) if both_classes else np.nan,
        "pr_auc_nonpenetrant": safe_metric(average_precision_score, y, score) if both_classes else np.nan,
        "cohen_kappa": safe_metric(cohen_kappa_score, y, pred),
        "mcc": safe_metric(matthews_corrcoef, y, pred),
        "wrong_side_count": int((pred != y).sum()),
        "wrong_side_rate": float((pred != y).mean()),
        "np_wrong_side_count": int(((y == 1) & (pred == 0)).sum()),
        "penetrant_wrong_side_count": int(((y == 0) & (pred == 1)).sum()),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
    return {
        "n_evaluable": int(len(y)),
        "mae": safe_metric(mean_absolute_error, y, pred),
        "mse": safe_metric(mean_squared_error, y, pred),
        "rmse": math.sqrt(safe_metric(mean_squared_error, y, pred)),
        "median_absolute_error": float(np.median(np.abs(y - pred))),
        "r2": safe_metric(r2_score, y, pred),
        "pearson_r": safe_metric(lambda a, b: np.corrcoef(a, b)[0, 1], y, pred),
        "np_mae": safe_metric(mean_absolute_error, y[classes == 1], pred[classes == 1]),
        "penetrant_mae": safe_metric(mean_absolute_error, y[classes == 0], pred[classes == 0]),
    }


def run_nested_outer_folds(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    encoder_name: str,
    output_dir: Path,
    model_kinds: Sequence[str],
    beta: float,
    margin_buffer: float,
    trials: int,
    inner_folds: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device_name: str,
    tune_loss: bool = False,
    overwrite: bool = False,
    encoder_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    df = validate_bbb_dataframe(df)
    if len(df) != len(embeddings):
        raise ValueError(f"Data rows ({len(df)}) != embedding rows ({len(embeddings)}).")
    if embeddings.ndim != 2 or not embeddings.shape[1] or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite two-dimensional matrix.")
    if trials < 1 or inner_folds < 2 or max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("Invalid training/CV settings.")
    if not model_kinds or len(set(model_kinds)) != len(model_kinds):
        raise ValueError("Request at least one model, without duplicate model kinds.")
    if not set(model_kinds).issubset({"regression", "mtl"}):
        raise ValueError("Unknown model kind.")
    if not np.isfinite([beta, margin_buffer]).all() or beta < 0 or margin_buffer < 0:
        raise ValueError("beta and margin_buffer must be finite and nonnegative.")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = df["bbb_class"].to_numpy(dtype=int)
    logbb = df["logBB"].to_numpy(dtype=float)
    has = df["_has_logbb"].to_numpy(dtype=bool)
    fold_values = sorted(df["outer_fold"].unique().tolist())
    # Validate every split before training or replacing previous outputs.
    splits_by_fold = {fold: dataframe_inner_splits(df, fold, np.flatnonzero(df["outer_fold"].to_numpy() != fold),
                       inner_folds, seed + pos * 1000) for pos, fold in enumerate(fold_values)}
    previous_manifest = output_dir / "manifest.json"
    if previous_manifest.exists():
        previous_encoder = json.loads(previous_manifest.read_text()).get("encoder")
        if previous_encoder != encoder_name:
            raise ValueError("This output directory belongs to a different encoder; use a separate --output-dir.")
    prepare_output_directory(output_dir,
        ["outer_fold_predictions.csv", "all_models_predictions.csv", "per_fold_metrics.csv", "aggregate_metrics.csv",
         "selected_hyperparameters.json", "manifest.json"],
        [f"{encoder_name}_regression", f"{encoder_name}_custom_mtl"], overwrite)
    all_predictions = []
    fold_metrics = []
    hyperparameters: Dict[str, Any] = {}

    for model_kind in model_kinds:
        if model_kind not in {"regression", "mtl"}:
            raise ValueError(f"Unknown model kind: {model_kind}")
        model_name = f"{encoder_name}_{'regression' if model_kind == 'regression' else 'custom_mtl'}"
        for fold_position, fold in enumerate(fold_values):
            print(f"\n[{model_name}] outer fold {fold}", flush=True)
            test_idx = np.flatnonzero(df["outer_fold"].to_numpy() == fold)
            train_idx = np.flatnonzero(df["outer_fold"].to_numpy() != fold)
            splits = splits_by_fold[fold]
            best_params, final_epochs = tune_head(
                embeddings[train_idx], classes[train_idx], logbb[train_idx], has[train_idx],
                model_kind, beta, margin_buffer, trials, inner_folds, max_epochs,
                patience, batch_size, seed + fold_position * 1000, device,
                splits=splits, tune_loss=tune_loss,
                audit_path=output_dir / model_name / f"outer_fold_{fold}" / "hpo_trials.csv",
            )
            selected_beta = float(best_params.get("beta", beta))
            selected_buffer = float(best_params.get("margin_buffer", margin_buffer))
            threshold_scores = oof_scores_for_threshold(
                embeddings[train_idx], classes[train_idx], logbb[train_idx], has[train_idx],
                model_kind, best_params, final_epochs, selected_beta, selected_buffer, inner_folds,
                batch_size, seed + fold_position * 1000, device,
                splits=splits,
            )
            tuned_threshold = choose_balanced_accuracy_threshold(classes[train_idx], threshold_scores)

            mean, scale = standardize_fit(
                normalization_reference(
                    embeddings[train_idx], has[train_idx], model_kind
                )
            )
            fit = fit_head(
                standardize_apply(embeddings[train_idx], mean, scale),
                classes[train_idx], logbb[train_idx], has[train_idx],
                None, None, None, None, model_kind, best_params, selected_beta, selected_buffer,
                final_epochs, 0, batch_size, seed + fold_position * 1000 + 777,
                device, fixed_epochs=final_epochs,
            )
            raw = predict_raw(
                fit.model, standardize_apply(embeddings[test_idx], mean, scale), device, batch_size
            )
            pred_logbb = raw
            score = -pred_logbb
            native_threshold = -TAU

            pred_rows = df.iloc[test_idx].copy()
            pred_rows["model"] = model_name
            pred_rows["raw_output"] = raw
            pred_rows["nonpenetrant_score"] = score
            pred_rows["predicted_logBB"] = pred_logbb
            pred_rows["inner_tuned_threshold"] = tuned_threshold
            pred_rows["predicted_class_tuned"] = (score > tuned_threshold).astype(int)
            pred_rows["native_threshold"] = native_threshold
            pred_rows["predicted_class_native"] = (score > native_threshold).astype(int)
            all_predictions.append(pred_rows)

            for threshold_type, threshold in (("inner_tuned", tuned_threshold), ("native", native_threshold)):
                metrics = classification_metrics(classes[test_idx], score, threshold)
                metrics.update({
                    "model": model_name,
                    "outer_fold": int(fold),
                    "evaluation_task": "classification",
                    "threshold_type": threshold_type,
                    "threshold": float(threshold),
                })
                fold_metrics.append(metrics)
            if has[test_idx].any():
                measured = has[test_idx]
                metrics = regression_metrics(
                    logbb[test_idx][measured], pred_logbb[measured], classes[test_idx][measured]
                )
                metrics.update({
                    "model": model_name,
                    "outer_fold": int(fold),
                    "evaluation_task": "regression",
                    "threshold_type": "not_applicable",
                    "threshold": np.nan,
                })
                fold_metrics.append(metrics)

            fold_dir = output_dir / model_name / f"outer_fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": {k: v.detach().cpu() for k, v in fit.model.state_dict().items()},
                "embedding_mean": mean,
                "embedding_scale": scale,
                "model_kind": model_kind,
                "encoder_name": encoder_name,
                "encoder_metadata": jsonable(encoder_metadata or {}),
                "input_dim": int(embeddings.shape[1]),
                "params": best_params,
                "epochs": final_epochs,
                "fit_seed": seed + fold_position * 1000 + 777,
                "inner_tuned_threshold": tuned_threshold,
                "native_threshold": native_threshold,
                "beta": selected_beta,
                "margin_buffer": selected_buffer,
                "tau": TAU,
                "loss_version": "half_sum_squared_error_plus_class_mean_squared_hinges_v1",
                "score_threshold_comparison": ">",
            }, fold_dir / "model.pt")
            hyperparameters.setdefault(model_name, {})[str(fold)] = {
                **best_params,
                "epochs": final_epochs,
                "inner_tuned_threshold": tuned_threshold,
                "native_threshold": native_threshold,
                "beta": selected_beta,
                "margin_buffer": selected_buffer,
            }
            pd.DataFrame({"sample_id": df.iloc[train_idx]["sample_id"].to_numpy(),
                          "nonpenetrant_score": threshold_scores}).to_csv(fold_dir / "inner_oof_threshold_scores.csv", index=False)
            assignment = np.full(len(train_idx), -1, dtype=int)
            for inner, (_, va) in enumerate(splits):
                assignment[va] = inner
            pd.DataFrame({"sample_id": df.iloc[train_idx]["sample_id"].to_numpy(), "inner_fold": assignment}).to_csv(fold_dir / "inner_folds_used.csv", index=False)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "outer_fold_predictions.csv", index=False)
    # Preserve the original wide file and also supply the evaluator's long schema.
    standardized = []
    for kind, pred_col, threshold_col in (("native", "predicted_class_native", "native_threshold"),
                                           ("inner_tuned", "predicted_class_tuned", "inner_tuned_threshold")):
        frame = predictions.copy()
        if kind == "inner_tuned":
            frame["model"] = frame["model"] + "_threshold"
        frame["model_task"] = "regression"
        frame["row_index"] = frame["_source_row"]
        frame["true_logBB"] = frame["logBB"]
        frame["has_logBB"] = frame["_has_logbb"]
        frame["classification_only"] = ~frame["_has_logbb"]
        frame["score_nonpenetrant"] = frame["nonpenetrant_score"]
        frame["predicted_bbb_class"] = frame[pred_col]
        frame["classification_threshold"] = -frame[threshold_col]  # logBB units
        frame["threshold_type"] = kind
        frame["wrong_side"] = frame["predicted_bbb_class"] != frame["bbb_class"]
        standardized.append(frame)
    pd.concat(standardized, ignore_index=True).to_csv(output_dir / "all_models_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "per_fold_metrics.csv", index=False)

    aggregate = []
    for model_name, group in predictions.groupby("model", sort=False):
        for threshold_type, pred_col in (
            ("inner_tuned", "predicted_class_tuned"),
            ("native", "predicted_class_native"),
        ):
            y = group["bbb_class"].to_numpy(dtype=int)
            score = group["nonpenetrant_score"].to_numpy(dtype=float)
            pred = group[pred_col].to_numpy(dtype=int)
            metrics = classification_metrics(y, score, 0.5, pred=pred)
            metrics.update({
                "model": model_name,
                "evaluation_task": "classification",
                "threshold_type": threshold_type,
            })
            aggregate.append(metrics)
        measured_group = group[group["_has_logbb"].astype(bool)]
        metrics = regression_metrics(
            measured_group["logBB"].to_numpy(dtype=float),
            measured_group["predicted_logBB"].to_numpy(dtype=float),
            measured_group["bbb_class"].to_numpy(dtype=int),
        )
        metrics.update({
            "model": model_name,
            "evaluation_task": "regression",
            "threshold_type": "not_applicable",
        })
        aggregate.append(metrics)
    pd.DataFrame(aggregate).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    with open(output_dir / "selected_hyperparameters.json", "w", encoding="utf-8") as handle:
        json.dump(jsonable(hyperparameters), handle, indent=2)
    manifest = {
        "encoder": encoder_name,
        "encoder_metadata": jsonable(encoder_metadata or {}),
        "experiment": "frozen pretrained encoder with conventional logBB regression versus custom MTL",
        "models": list(model_kinds),
        "class_convention": "0=penetrant, 1=non-penetrant",
        "tau": TAU,
        "beta": beta,
        "margin_buffer": margin_buffer,
        "loss_parameters_tuned": tune_loss,
        "loss": "0.5*sum_measured(error^2) + beta*sum_class_means(squared_margin_violation)",
        "loss_scaling": "none",
        "hpo_objectives": ["mean_inner_measured_mse:minimize", "mean_inner_all_balanced_accuracy:maximize"],
        "hpo_selection": "normalized Pareto distance to the ideal point",
        "threshold_units": "native_threshold and inner_tuned_threshold use -logBB; classification_threshold uses logBB",
        "classification_rule": "predicted_logBB < logBB_threshold (equivalently score > score_threshold)",
        "trials_per_model_per_outer_fold": trials,
        "inner_folds": inner_folds,
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "prediction_batch_size": batch_size,
        "optimizer_updates_per_epoch": 1,
        "device": str(device),
        "note": "This is frozen-encoder transfer, not end-to-end fine-tuning.",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(jsonable(manifest), handle, indent=2)
