#!/usr/bin/env python3
"""Evaluate frozen BBB models on an external Inxight validation set.

This version adds explicit ID-mismatch diagnostics and guarded repair.
By default, prediction IDs must still exactly match external IDs.
Use --allow-row-order-id-repair only when predictions were generated from
the same 990 compounds in the same row order but temporary/generated IDs
were written to the prediction CSV.


External CSV requirements:
  external_compound_id
  bbb_class                 0 = penetrant, 1 = non-penetrant
Optional:
  max_training_tanimoto

Prediction CSV requirements (one or many files):
  external_compound_id
  score_np                  larger = more likely non-penetrant
  pred_np                   frozen hard prediction, 0 or 1
Optional:
  model                     defaults to prediction filename stem
  probability_np            calibrated P(non-penetrant), for Brier score
  predicted_logbb
  threshold

Example:
for i in {1..5}; do 

python ./10_evaluate_inxight_predictions_id_safe.py \
    --external ../datasets/not_druglike_b3db_labelled.csv \
    --predictions external_test_set/all_selected_models.csv \
    --output-dir external_test_set/01_eval/ \
    --n-bootstrap 500 --allow-row-order-id-repair

--primary-model xgboost_aft_logistic_clean \
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PAIR_METRICS = [
    "balanced_accuracy",
    "mcc",
    "np_recall",
    "np_precision",
    "np_f1",
    "roc_auc",
    "pr_auc",
]
BOOT_METRICS = PAIR_METRICS + ["penetrant_specificity", "wrong_side_rate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--external", required=True, type=Path)
    p.add_argument("--predictions", required=True, nargs="+", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
    )
    p.add_argument("--id-col", default="external_compound_id")
    p.add_argument("--label-col", default="Class")
    p.add_argument("--primary-model", default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument(
        "--novelty-thresholds",
        nargs="*",
        type=float,
        default=[0.8, 0.6],
    )
    p.add_argument(
        "--allow-row-order-id-repair",
        action="store_true",
        help=(
            "When a model has exactly the same number of unique predictions "
            "as external compounds, but zero ID overlap, replace that model's "
            "prediction IDs with external IDs in row order. Use only when the "
            "prediction file and external file are known to have identical "
            "compound ordering."
        ),
    )
    p.add_argument(
        "--id-map",
        type=Path,
        default=None,
        help=(
            "Optional explicit ID mapping CSV. It must contain "
            "`prediction_external_compound_id` and the external ID column "
            "(default `external_compound_id`). Explicit mapping is safer than "
            "row-order repair."
        ),
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: set[str], path: Path) -> None:
    missing = sorted(cols - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Available: {list(df.columns)}"
        )


def read_external(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    require_columns(df, {id_col, label_col}, path)
    df = df.copy()
    df[id_col] = df[id_col].map(normalize_identifier)
    df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    if not set(df[label_col].unique()).issubset({0, 1}):
        raise ValueError("External labels must be 0/1.")
    if df[id_col].duplicated().any():
        raise ValueError("External compound IDs are not unique.")
    return df


def read_prediction(path: Path, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    require_columns(df, {id_col, "score_np", "pred_np"}, path)
    df = df.copy()
    if "model" not in df.columns:
        df["model"] = path.stem
    df[id_col] = df[id_col].map(normalize_identifier)
    df["model"] = df["model"].astype(str).str.strip()
    df["score_np"] = pd.to_numeric(df["score_np"], errors="raise")
    df["pred_np"] = pd.to_numeric(df["pred_np"], errors="raise").astype(int)
    if not set(df["pred_np"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: pred_np must be 0/1.")
    if "probability_np" in df.columns:
        df["probability_np"] = pd.to_numeric(
            df["probability_np"], errors="coerce"
        )
        bad = df["probability_np"].notna() & ~df["probability_np"].between(0, 1)
        if bad.any():
            raise ValueError(f"{path}: probability_np outside [0,1].")
    if df.duplicated(["model", id_col]).any():
        raise ValueError(f"{path}: duplicate model/compound predictions.")
    df["prediction_source_file"] = str(path)
    return df



def normalize_identifier(value: object) -> str:
    """
    Normalize common CSV-induced ID formatting without changing real text IDs.

    Examples:
      123.0 -> 123
      " ABC " -> ABC
    """
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def apply_explicit_id_map(
    predictions: pd.DataFrame,
    *,
    mapping_path: Path,
    prediction_id_col: str,
    external_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(mapping_path, low_memory=False)
    prediction_map_col = "prediction_external_compound_id"
    require_columns(
        mapping,
        {prediction_map_col, external_id_col},
        mapping_path,
    )
    mapping = mapping[[prediction_map_col, external_id_col]].copy()
    mapping[prediction_map_col] = (
        mapping[prediction_map_col].map(normalize_identifier)
    )
    mapping[external_id_col] = (
        mapping[external_id_col].map(normalize_identifier)
    )

    if mapping[prediction_map_col].duplicated().any():
        raise ValueError(
            f"{mapping_path}: duplicate prediction IDs in explicit map."
        )
    if mapping[external_id_col].duplicated().any():
        raise ValueError(
            f"{mapping_path}: duplicate external IDs in explicit map."
        )

    original_ids = predictions[prediction_id_col].copy()
    mapped = predictions.merge(
        mapping,
        left_on=prediction_id_col,
        right_on=prediction_map_col,
        how="left",
        validate="many_to_one",
        suffixes=("", "_mapped"),
    )
    if mapped[external_id_col + "_mapped"].isna().any():
        missing = mapped.loc[
            mapped[external_id_col + "_mapped"].isna(),
            prediction_id_col,
        ].drop_duplicates().head(20).tolist()
        raise ValueError(
            f"{mapping_path}: no mapping for prediction IDs {missing}"
        )

    mapped[prediction_id_col] = mapped[external_id_col + "_mapped"]
    mapped = mapped.drop(
        columns=[prediction_map_col, external_id_col + "_mapped"]
    )

    audit = pd.DataFrame(
        {
            "model": predictions["model"].to_numpy(),
            "prediction_row": np.arange(len(predictions), dtype=int),
            "original_prediction_id": original_ids.to_numpy(),
            "repaired_external_id": mapped[prediction_id_col].to_numpy(),
            "repair_method": "explicit_id_map",
        }
    )
    return mapped, audit


def repair_ids_by_row_order(
    predictions: pd.DataFrame,
    external: pd.DataFrame,
    *,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace completely disjoint prediction IDs by external IDs in row order.

    Guardrails:
    - each model must have exactly one prediction per external row;
    - prediction IDs must be unique within model;
    - overlap must be exactly zero, not partial;
    - all models must have the same prediction row count.
    """
    external_ids = external[id_col].tolist()
    external_set = set(external_ids)
    repaired_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []

    for model, group in predictions.groupby("model", sort=False):
        group = group.copy()
        if group[id_col].duplicated().any():
            raise ValueError(
                f"{model}: duplicate prediction IDs prevent row-order repair."
            )

        prediction_ids = group[id_col].tolist()
        prediction_set = set(prediction_ids)
        overlap = len(prediction_set & external_set)

        if prediction_set == external_set:
            repaired_parts.append(group)
            continue

        if overlap != 0:
            raise ValueError(
                f"{model}: partial ID overlap ({overlap}) cannot be safely "
                "repaired by row order."
            )
        if len(group) != len(external):
            raise ValueError(
                f"{model}: has {len(group)} predictions but external contains "
                f"{len(external)} compounds."
            )

        original_ids = group[id_col].to_numpy(copy=True)
        group[id_col] = external_ids

        audit_parts.append(
            pd.DataFrame(
                {
                    "model": model,
                    "prediction_row": np.arange(len(group), dtype=int),
                    "original_prediction_id": original_ids,
                    "repaired_external_id": external_ids,
                    "repair_method": "explicit_row_order",
                }
            )
        )
        repaired_parts.append(group)

    repaired = pd.concat(repaired_parts, ignore_index=True, sort=False)
    audit = (
        pd.concat(audit_parts, ignore_index=True)
        if audit_parts
        else pd.DataFrame(
            columns=[
                "model",
                "prediction_row",
                "original_prediction_id",
                "repaired_external_id",
                "repair_method",
            ]
        )
    )
    return repaired, audit


def id_examples(
    external: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    id_col: str,
) -> pd.DataFrame:
    rows = []
    external_examples = external[id_col].head(10).tolist()
    for model, group in predictions.groupby("model", sort=True):
        prediction_examples = group[id_col].head(10).tolist()
        rows.append(
            {
                "model": model,
                "external_id_examples": json.dumps(external_examples),
                "prediction_id_examples": json.dumps(prediction_examples),
            }
        )
    return pd.DataFrame(rows)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def metrics(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "n": int(len(y)),
        "n_penetrant": int((y == 0).sum()),
        "n_nonpenetrant": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "np_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "np_precision": float(
            precision_score(y, pred, pos_label=1, zero_division=0)
        ),
        "np_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "penetrant_specificity": safe_div(tn, tn + fp),
        "np_false_negative_rate": safe_div(fn, tp + fn),
        "penetrant_false_positive_rate": safe_div(fp, tn + fp),
        "wrong_side_count": int((pred != y).sum()),
        "wrong_side_rate": float((pred != y).mean()),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    if probability is not None and np.isfinite(probability).all():
        out["brier_score"] = float(brier_score_loss(y, probability))
    else:
        out["brier_score"] = np.nan
    return out


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if len(idx):
            parts.append(rng.choice(idx, size=len(idx), replace=True))
    sample = np.concatenate(parts)
    rng.shuffle(sample)
    return sample


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_cis(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    observed = metrics(y, pred, score)
    rng = np.random.default_rng(seed)
    values = {name: np.full(n_boot, np.nan) for name in BOOT_METRICS}
    for b in range(n_boot):
        s = stratified_indices(y, rng)
        m = metrics(y[s], pred[s], score[s])
        for name in BOOT_METRICS:
            values[name][b] = m[name]
    rows = []
    for name in BOOT_METRICS:
        low, high = ci(values[name])
        rows.append(
            {
                "metric": name,
                "estimate": observed[name],
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap": n_boot,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_pvalue(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if not len(diffs):
        return np.nan
    p_lo = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
    p_hi = (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)
    return float(min(1.0, 2 * min(p_lo, p_hi)))


def paired_difference(
    y: np.ndarray,
    pred_a: np.ndarray,
    score_a: np.ndarray,
    pred_b: np.ndarray,
    score_b: np.ndarray,
    metric_name: str,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    m_a = metrics(y, pred_a, score_a)[metric_name]
    m_b = metrics(y, pred_b, score_b)[metric_name]
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        s = stratified_indices(y, rng)
        a = metrics(y[s], pred_a[s], score_a[s])[metric_name]
        bb = metrics(y[s], pred_b[s], score_b[s])[metric_name]
        diffs[b] = a - bb
    low, high = ci(diffs)
    return {
        "metric_a": m_a,
        "metric_b": m_b,
        "difference_a_minus_b": m_a - m_b,
        "ci_low": low,
        "ci_high": high,
        "p_bootstrap_two_sided": bootstrap_pvalue(diffs),
        "n_bootstrap": n_boot,
    }


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int]:
    correct_a = pred_a == y
    correct_b = pred_b == y
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p = 1.0 if discordant == 0 else float(
        binomtest(
            min(a_correct_b_wrong, a_wrong_b_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": discordant,
        "net_errors_avoided_by_a": a_wrong_b_correct - a_correct_b_wrong,
        "mcnemar_exact_p": p,
    }


def holm(pvalues: pd.Series) -> pd.Series:
    pvalues = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    running = 0.0
    m = len(valid)
    for rank, (idx, raw) in enumerate(valid.items()):
        adjusted = min(1.0, (m - rank) * float(raw))
        running = max(running, adjusted)
        out.loc[idx] = running
    return out


def subset_masks(df: pd.DataFrame, thresholds: list[float]) -> dict[str, pd.Series]:
    out = {"all_nonoverlapping": pd.Series(True, index=df.index)}
    if "max_training_tanimoto" in df.columns:
        sim = pd.to_numeric(df["max_training_tanimoto"], errors="coerce")
        for threshold in thresholds:
            name = f"max_tanimoto_lt_{threshold:.2f}".replace(".", "p")
            out[name] = sim < threshold
    return out


def main() -> None:
    args = parse_args()
    for path in [args.external, *args.predictions]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    external = read_external(args.external, args.id_col, args.label_col)
    predictions = pd.concat(
        [read_prediction(p, args.id_col) for p in args.predictions],
        ignore_index=True,
        sort=False,
    )
    if predictions.duplicated(["model", args.id_col]).any():
        raise ValueError("Duplicate model/compound predictions across files.")

    id_examples(
        external,
        predictions,
        id_col=args.id_col,
    ).to_csv(
        args.output_dir / "00_id_examples_before_repair.csv",
        index=False,
    )

    repair_audits: list[pd.DataFrame] = []

    if args.id_map is not None:
        if not args.id_map.exists():
            raise FileNotFoundError(args.id_map)
        predictions, explicit_audit = apply_explicit_id_map(
            predictions,
            mapping_path=args.id_map,
            prediction_id_col=args.id_col,
            external_id_col=args.id_col,
        )
        repair_audits.append(explicit_audit)

    external_ids = set(external[args.id_col])
    coverage_rows = []
    for model, group in predictions.groupby("model", sort=True):
        ids = set(group[args.id_col])
        coverage_rows.append(
            {
                "model": model,
                "n_external": len(external_ids),
                "n_predictions": len(group),
                "n_matching_external": len(ids & external_ids),
                "n_missing": len(external_ids - ids),
                "n_extra": len(ids - external_ids),
                "complete_coverage": ids == external_ids,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        args.output_dir / "00_prediction_coverage_before_repair.csv",
        index=False,
    )

    if (
        not coverage["complete_coverage"].all()
        and args.allow_row_order_id_repair
    ):
        predictions, row_audit = repair_ids_by_row_order(
            predictions,
            external,
            id_col=args.id_col,
        )
        repair_audits.append(row_audit)

    if repair_audits:
        pd.concat(
            repair_audits,
            ignore_index=True,
            sort=False,
        ).to_csv(
            args.output_dir / "00_id_repair_audit.csv",
            index=False,
        )

    external_ids = set(external[args.id_col])
    coverage_rows = []
    for model, group in predictions.groupby("model", sort=True):
        ids = set(group[args.id_col])
        coverage_rows.append(
            {
                "model": model,
                "n_external": len(external_ids),
                "n_predictions": len(group),
                "n_matching_external": len(ids & external_ids),
                "n_missing": len(external_ids - ids),
                "n_extra": len(ids - external_ids),
                "complete_coverage": ids == external_ids,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        args.output_dir / "00_prediction_coverage.csv",
        index=False,
    )

    if not coverage["complete_coverage"].all():
        raise ValueError(
            "Every model must predict exactly all external compounds. "
            "See 00_prediction_coverage.csv and "
            "00_id_examples_before_repair.csv. Your files may contain the "
            "same rows under different IDs. Use --id-map for an explicit "
            "mapping, or --allow-row-order-id-repair only when row order is "
            "known to be identical."
        )

    merged = predictions.merge(
        external,
        on=args.id_col,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_external"),
    )
    merged.to_csv(args.output_dir / "01_all_external_predictions.csv", index=False)

    metric_rows = []
    ci_frames = []
    masks = subset_masks(merged, args.novelty_thresholds)
    for subset_name, mask in masks.items():
        subset = merged.loc[mask].copy()
        for model, group in subset.groupby("model", sort=True):
            group = group.sort_values(args.id_col)
            y = group[args.label_col].to_numpy(int)
            pred = group["pred_np"].to_numpy(int)
            score = group["score_np"].to_numpy(float)
            probability = None
            if "probability_np" in group and group["probability_np"].notna().all():
                probability = group["probability_np"].to_numpy(float)
            metric_rows.append(
                {"subset": subset_name, "model": model, **metrics(y, pred, score, probability)}
            )
            frame = bootstrap_cis(
                y,
                pred,
                score,
                args.n_bootstrap,
                args.seed + abs(hash((subset_name, model))) % 1_000_000,
            )
            frame.insert(0, "model", model)
            frame.insert(0, "subset", subset_name)
            ci_frames.append(frame)

    model_metrics = pd.DataFrame(metric_rows)
    model_metrics.to_csv(args.output_dir / "02_model_metrics.csv", index=False)
    pd.concat(ci_frames, ignore_index=True).to_csv(
        args.output_dir / "03_bootstrap_confidence_intervals.csv", index=False
    )

    pair_rows = []
    for subset_name, mask in masks.items():
        subset = merged.loc[mask].copy()
        model_frames = {
            str(model): group.sort_values(args.id_col).copy()
            for model, group in subset.groupby("model", sort=True)
        }
        names = sorted(model_frames)
        if args.primary_model:
            if args.primary_model not in names:
                raise ValueError(
                    f"Primary model {args.primary_model!r} not found. Available: {names}"
                )
            pairs = [(args.primary_model, name) for name in names if name != args.primary_model]
        else:
            pairs = list(itertools.combinations(names, 2))

        for model_a, model_b in pairs:
            a = model_frames[model_a][
                [args.id_col, args.label_col, "pred_np", "score_np"]
            ]
            b = model_frames[model_b][
                [args.id_col, args.label_col, "pred_np", "score_np"]
            ]
            paired = a.merge(
                b,
                on=[args.id_col, args.label_col],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            y = paired[args.label_col].to_numpy(int)
            pred_a = paired["pred_np_a"].to_numpy(int)
            score_a = paired["score_np_a"].to_numpy(float)
            pred_b = paired["pred_np_b"].to_numpy(int)
            score_b = paired["score_np_b"].to_numpy(float)
            mc = mcnemar(y, pred_a, pred_b)
            for metric_name in PAIR_METRICS:
                comp = paired_difference(
                    y,
                    pred_a,
                    score_a,
                    pred_b,
                    score_b,
                    metric_name,
                    args.n_bootstrap,
                    args.seed
                    + abs(hash((subset_name, model_a, model_b, metric_name))) % 1_000_000,
                )
                pair_rows.append(
                    {
                        "subset": subset_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric_name,
                        **comp,
                        **mc,
                    }
                )

    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise):
        pairwise["p_bootstrap_holm"] = np.nan
        for _, idx in pairwise.groupby(["subset", "metric"]).groups.items():
            pairwise.loc[idx, "p_bootstrap_holm"] = holm(
                pairwise.loc[idx, "p_bootstrap_two_sided"]
            )
        unique_mc = pairwise[
            ["subset", "model_a", "model_b", "mcnemar_exact_p"]
        ].drop_duplicates().copy()
        unique_mc["mcnemar_exact_p_holm"] = np.nan
        for _, idx in unique_mc.groupby("subset").groups.items():
            unique_mc.loc[idx, "mcnemar_exact_p_holm"] = holm(
                unique_mc.loc[idx, "mcnemar_exact_p"]
            )
        pairwise = pairwise.merge(
            unique_mc[
                ["subset", "model_a", "model_b", "mcnemar_exact_p_holm"]
            ],
            on=["subset", "model_a", "model_b"],
            how="left",
            validate="many_to_one",
        )
    pairwise.to_csv(
        args.output_dir / "04_pairwise_model_comparisons.csv", index=False
    )

    config = {
        "external": str(args.external),
        "prediction_files": [str(p) for p in args.predictions],
        "n_external": int(len(external)),
        "n_penetrant": int((external[args.label_col] == 0).sum()),
        "n_nonpenetrant": int((external[args.label_col] == 1).sum()),
        "models": sorted(predictions["model"].unique().tolist()),
        "primary_model": args.primary_model,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "score_direction": "larger score_np = more likely non-penetrant",
    }
    (args.output_dir / "05_evaluation_config.json").write_text(
        json.dumps(config, indent=2)
    )

    cols = [
        "subset",
        "model",
        "n",
        "balanced_accuracy",
        "mcc",
        "np_recall",
        "np_precision",
        "np_f1",
        "penetrant_specificity",
        "roc_auc",
        "pr_auc",
        "wrong_side_count",
    ]
    print(
        model_metrics[cols]
        .sort_values(["subset", "balanced_accuracy"], ascending=[True, False])
        .to_string(index=False)
    )
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
