#!/usr/bin/env python3
"""Paired statistics for script 13's saved Kp,uu ensemble predictions.

No model loading, training, inference, chemistry, or network access.
Requires Python >=3.10, numpy, pandas, scipy.

python 19_kpuu_statistics.py \
  --predictions-csv ./kpuu_transfer/transfer_predictions.csv \
  --output-dir ./kpuu_transfer/kpuu_statistics \
  --bootstrap 2000 \
  --permutations 10000

"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


METRICS = (
    'spearman_r', 'pearson_r', 'kendall_tau_b', 'roc_auc_low_Kpuu',
    'average_precision_low_Kpuu', 'pr_auc_trapezoidal_low_Kpuu',
    'accuracy_BBB_rule', 'balanced_accuracy_BBB_rule',
    'precision_low_Kpuu_BBB_rule', 'recall_low_Kpuu_BBB_rule',
    'f1_low_Kpuu_BBB_rule', 'specificity_high_Kpuu_BBB_rule',
    'mcc_BBB_rule', 'cohens_kappa_BBB_rule',
)
KNOWN_MODELS = (
    'lightgbm_mse_regression',
    'lightgbm_mse_regression_threshold',
    'custom_mtl_bab',
)


def divide(numerator, denominator, default=np.nan):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, float), np.asarray(denominator, float))
    result = np.full(numerator.shape, default, dtype=float)
    return np.divide(numerator, denominator, out=result, where=denominator != 0)


def correlation(a, b):
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    return divide((ac * bc).sum(axis=1),
                  np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1)))


def metric_batch(y, pred, calls, cutoff):
    """Rows are replicates. Calls are supplied separately for paired swaps.

    A call always travels with its originating model's score. Reapplying a
    receiving model's threshold after swapping would invalidate comparisons
    between MSE and threshold-tuned MSE.
    """
    pred = np.atleast_2d(np.asarray(pred, float))
    y = np.broadcast_to(np.asarray(y, float), pred.shape)
    calls = np.broadcast_to(np.asarray(calls, bool), pred.shape)
    n = pred.shape[1]
    low = y < np.log10(cutoff)
    positives = low.sum(axis=1)
    negatives = n - positives
    both = (positives > 0) & (negatives > 0)
    variable = (n > 2) & (np.ptp(y, axis=1) > 0) & (np.ptp(pred, axis=1) > 0)

    pred_ranks = rankdata(pred, axis=1, method='average')
    spearman = correlation(rankdata(y, axis=1, method='average'), pred_ranks)
    pearson = correlation(y, pred)
    ii, jj = np.triu_indices(n, 1)
    ys = np.sign(y[:, ii] - y[:, jj])
    ps = np.sign(pred[:, ii] - pred[:, jj])
    kendall = divide((ys * ps).sum(axis=1),
                     np.sqrt((ys != 0).sum(axis=1).astype(float) * (ps != 0).sum(axis=1)))
    spearman[~variable] = pearson[~variable] = kendall[~variable] = np.nan

    # Lower predicted logBB is a higher score for the low-Kp,uu positive class.
    roc = divide(((n + 1 - pred_ranks) * low).sum(axis=1)
                 - positives * (positives + 1) / 2, positives * negatives)
    order = np.argsort(pred, axis=1, kind='stable')
    sorted_scores = np.take_along_axis(pred, order, axis=1)
    sorted_low = np.take_along_axis(low, order, axis=1)
    cumulative_tp = np.cumsum(sorted_low, axis=1)
    positions = np.arange(n)[None, :]
    end = np.concatenate((sorted_scores[:, :-1] != sorted_scores[:, 1:],
                          np.ones((len(pred), 1), bool)), axis=1)
    end_index = np.minimum.accumulate(np.where(end, positions, n)[:, ::-1], axis=1)[:, ::-1]
    precision = cumulative_tp / (positions + 1)
    group_precision = np.take_along_axis(precision, end_index, axis=1)
    ap = divide((sorted_low * group_precision).sum(axis=1), positives)
    previous_end = np.maximum.accumulate(np.where(end, positions, -1), axis=1)
    previous_end = np.concatenate((np.full((len(pred), 1), -1), previous_end[:, :-1]), axis=1)
    prev_tp = np.take_along_axis(np.column_stack((np.zeros(len(pred)), cumulative_tp)), previous_end + 1, axis=1)
    prev_precision = np.take_along_axis(np.column_stack((np.ones(len(pred)), precision)), previous_end + 1, axis=1)
    pr_area = divide(((cumulative_tp - prev_tp) * (precision + prev_precision) / 2 * end).sum(axis=1), positives)
    roc[~both] = ap[~both] = pr_area[~both] = np.nan

    tp = (low & calls).sum(axis=1).astype(float)
    fp = (~low & calls).sum(axis=1).astype(float)
    fn = (low & ~calls).sum(axis=1).astype(float)
    tn = (~low & ~calls).sum(axis=1).astype(float)
    recall = divide(tp, positives)
    specificity = divide(tn, negatives)
    accuracy = (tp + tn) / n
    balanced = (recall + specificity) / 2
    ppv = divide(tp, tp + fp, 0.)
    f1 = divide(2 * tp, 2 * tp + fp + fn, 0.)
    mcc = divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 0.)
    mcc[~both] = np.nan
    expected = ((tp + fp) * positives + (tn + fn) * negatives) / n**2
    kappa = divide(accuracy - expected, 1 - expected)
    kappa[~both] = np.nan
    return np.column_stack((spearman, pearson, kendall, roc, ap, pr_area,
                            accuracy, balanced, ppv, recall, f1, specificity, mcc, kappa))


def load_predictions(path, cutoff):
    frame = pd.read_csv(path, low_memory=False, dtype={'key': str, 'sample_id': str, 'model': str})
    id_col = 'key' if 'key' in frame else 'sample_id'
    required = {id_col, 'model', 'prediction', 'decision_threshold', 'log10_Kpuu'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. Use script 13\'s transfer_predictions.csv, not fold_predictions.csv or transfer_metrics.csv.')
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError('Empty table or missing required values; no rows are silently removed.')
    if frame.duplicated(['model', id_col]).any():
        raise ValueError('Multiple rows per model/compound. Supply ensemble transfer_predictions.csv, not fold_predictions.csv.')
    for column in ['prediction', 'decision_threshold', 'log10_Kpuu'] + (['Kpuu'] if 'Kpuu' in frame else []):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all():
            raise ValueError(f'{column} has nonfinite values.')
    if 'Kpuu' in frame:
        if (frame.Kpuu <= 0).any() or not np.allclose(np.log10(frame.Kpuu), frame.log10_Kpuu, atol=1e-8, rtol=1e-8):
            raise ValueError('Kpuu must be positive and consistent with log10_Kpuu.')
    found = set(frame.model)
    models = [m for m in KNOWN_MODELS if m in found] + sorted(found - set(KNOWN_MODELS))
    if len(models) < 2:
        raise ValueError('At least two models are needed for paired comparisons.')
    ids = sorted(frame[id_col].unique())
    if len(ids) < 3:
        raise ValueError('At least three compounds are needed.')
    if any(set(frame.loc[frame.model.eq(m), id_col]) != set(ids) for m in models):
        raise ValueError('Models have different compound sets; complete paired predictions are required.')
    aligned = [frame.loc[frame.model.eq(m)].set_index(id_col).loc[ids] for m in models]
    truth = aligned[0].log10_Kpuu.to_numpy(float)
    for m, group in zip(models[1:], aligned[1:]):
        if not np.allclose(group.log10_Kpuu, truth, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{m}: inconsistent Kp,uu targets across models.')
        for col in ['sample_id', 'smiles', 'species']:
            if col in group and not group[col].fillna('').equals(aligned[0][col].fillna('')):
                raise ValueError(f'{m}: inconsistent {col} values across models.')
    thresholds = []
    for m, group in zip(models, aligned):
        if group.decision_threshold.nunique() != 1:
            raise ValueError(f'{m}: expected one saved ensemble decision threshold.')
        thresholds.append(float(group.decision_threshold.iloc[0]))
    thresholds = np.asarray(thresholds)
    scores = np.stack([g.prediction.to_numpy(float) for g in aligned])
    calls = scores < thresholds[:, None]
    low = truth < np.log10(cutoff)
    if low.all() or not low.any():
        raise ValueError('Both low- and high-Kp,uu compounds are needed for classification comparisons.')
    for m, group, predicted in zip(models, aligned, calls):
        for col, expected in [('true_low_Kpuu', low), ('predicted_low_Kpuu_BBB_rule', predicted)]:
            if col in group:
                saved = pd.to_numeric(group[col], errors='raise').to_numpy()
                if not np.array_equal(saved, expected.astype(int)):
                    raise ValueError(f'{m}: saved {col} disagrees with the requested cutoff or saved decision threshold.')
    checks = []
    manifest_path = path.parent / 'run_manifest.json'
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text())
        if 'n_eligible' in original and int(original['n_eligible']) != len(ids):
            raise ValueError('Compound count disagrees with neighboring run_manifest.json.')
        original_cutoff = original.get('settings', {}).get('kpuu_cutoff')
        if original_cutoff is not None and not np.isclose(float(original_cutoff), cutoff, rtol=0, atol=1e-12):
            raise ValueError('Requested Kp,uu cutoff differs from the original run_manifest.json.')
        checks.append(str(manifest_path))
    eligible_path = path.parent / 'kpuu_eligible.csv'
    if eligible_path.exists():
        eligible = pd.read_csv(eligible_path, dtype={id_col: str})
        if id_col in eligible:
            if eligible[id_col].duplicated().any() or set(eligible[id_col]) != set(ids):
                raise ValueError('Prediction compound set disagrees with neighboring kpuu_eligible.csv.')
            checks.append(str(eligible_path))
    return models, ids, truth, scores, calls, thresholds, checks


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [.025, .975])
    return float(lo), float(hi), len(finite)


def holm(p_values):
    p = np.asarray(p_values, float)
    result = np.full(len(p), np.nan)
    indices = np.flatnonzero(np.isfinite(p))
    order = indices[np.argsort(p[indices], kind='stable')]
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return result


def analyze(models, truth, scores, calls, thresholds, args):
    n = len(truth)
    points = metric_batch(truth, scores, calls, args.kpuu_cutoff)
    bootstrap = np.empty((len(models), args.bootstrap, len(METRICS)))
    # Sorted compound keys + the same RNG draw order reproduce script 13 CIs.
    rng = np.random.default_rng(args.seed)
    for start in range(0, args.bootstrap, args.batch_size):
        stop = min(start + args.batch_size, args.bootstrap)
        ix = rng.integers(0, n, size=(stop - start, n))
        for i in range(len(models)):
            bootstrap[i, start:stop] = metric_batch(truth[ix], scores[i, ix], calls[i, ix], args.kpuu_cutoff)
    print(f'Paired bootstrap complete ({args.bootstrap:,} resamples).', flush=True)
    estimates = []
    for i, model in enumerate(models):
        for j, metric in enumerate(METRICS):
            lo, hi, count = interval(bootstrap[i, :, j])
            estimates.append(dict(model=model, metric=metric, estimate=points[i, j],
                                  ci_low=lo, ci_high=hi, n_compounds=n, n_valid_bootstraps=count,
                                  kpuu_cutoff=args.kpuu_cutoff, predicted_logBB_cutoff=thresholds[i]))

    comparisons = []
    for a, b in combinations(range(len(models)), 2):
        observed = points[b] - points[a]
        extreme = np.zeros(len(METRICS), int)
        valid = np.zeros(len(METRICS), int)
        # Reset per pair: identical score comparisons get identical Monte Carlo p-values.
        permutation_rng = np.random.default_rng(args.seed + 1)
        for start in range(0, args.permutations, args.batch_size):
            size = min(args.batch_size, args.permutations - start)
            swap = permutation_rng.integers(0, 2, size=(size, n)).astype(bool)
            a_score, b_score = np.where(swap, scores[b], scores[a]), np.where(swap, scores[a], scores[b])
            a_call, b_call = np.where(swap, calls[b], calls[a]), np.where(swap, calls[a], calls[b])
            perm_delta = (metric_batch(truth, b_score, b_call, args.kpuu_cutoff)
                          - metric_batch(truth, a_score, a_call, args.kpuu_cutoff))
            finite = np.isfinite(perm_delta)
            valid += finite.sum(axis=0)
            tolerance = 1e-12 * np.maximum(1., np.abs(observed))
            extreme += (finite & (np.abs(perm_delta) >= np.abs(observed) - tolerance)).sum(axis=0)
        for j, metric in enumerate(METRICS):
            lo, hi, count = interval(bootstrap[b, :, j] - bootstrap[a, :, j])
            # Do not silently change the null by discarding undefined permutation draws.
            if not np.isfinite(observed[j]):
                p_value, status = np.nan, 'observed_statistic_undefined'
            elif abs(observed[j]) <= 1e-12:
                p_value, status = 1., 'zero_observed_difference'
            elif valid[j] != args.permutations:
                p_value, status = np.nan, 'undefined_permutation_statistics'
            else:
                p_value, status = (int(extreme[j]) + 1) / (args.permutations + 1), 'ok'
            comparisons.append(dict(model=models[b], reference=models[a], metric=metric,
                                    delta=observed[j], ci_low=lo, ci_high=hi,
                                    p_value=p_value, p_status=status,
                                    n_compounds=n, n_valid_bootstraps=count,
                                    n_permutations=args.permutations,
                                    n_valid_permutations=int(valid[j]), n_extreme_permutations=int(extreme[j]),
                                    multiplicity_family='correlation' if j < 3 else 'classification'))
        print(f'Paired permutation complete: {models[b]} vs {models[a]}.', flush=True)
    comparison = pd.DataFrame(comparisons)
    comparison['p_holm'] = holm(comparison.p_value)
    comparison['p_holm_family'] = np.nan
    for _, group in comparison.groupby('multiplicity_family', sort=False):
        comparison.loc[group.index, 'p_holm_family'] = holm(group.p_value)
    return pd.DataFrame(estimates), comparison


def methods_text(n, args, comparisons):
    return (
        f'Statistical comparisons used the saved ensemble predictions for {n} compounds from '
        f'the Kp,uu transfer evaluation, without retraining, new inference, or threshold selection. '
        f'Predicted logBB was compared with log10(Kp,uu) using Spearman correlation, Pearson '
        f'correlation, and Kendall tau-b. Low unbound exposure was defined as Kp,uu < '
        f'{args.kpuu_cutoff:g}. ROC AUC, average precision, and trapezoidal precision-recall AUC '
        f'used negative predicted logBB as the positive-class score. Classification metrics '
        f'used each model\'s previously saved logBB decision threshold. Percentile 95% confidence '
        f'intervals for metrics and paired between-model differences were obtained using '
        f'{args.bootstrap:,} paired compound bootstrap resamples. Two-sided paired permutation '
        f'tests used {args.permutations:,} random within-compound swaps of the two models\' '
        f'prediction outputs, preserving the score and its corresponding binary classification '
        f'together. P-values were computed as (1 + the number of permuted absolute differences '
        f'at least as large as the observed absolute difference) / (1 + the number of permutations). '
        f'These tests assume within-compound exchangeability of the model outputs under the null. '
        f'Raw p-values and Holm-adjusted p-values across all {len(comparisons)} pairwise metric '
        f'tests were reported; additional family-specific Holm adjustments were supplied separately '
        f'for correlation and classification comparisons. Uncertainty was conditional on the '
        f'fitted models and fixed thresholds. Undefined precision and F1 were set to zero; '
        f'constant-prediction MCC was zero when both target classes were present. Undefined '
        f'bootstrap statistics were excluded with valid replicate counts reported. Tests with '
        f'undefined permutation statistics were flagged rather than assigned a numerical p-value, '
        f'except that a zero observed difference was assigned p = 1.\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions-csv', required=True, type=Path,
                        help='Script 13 output: transfer_predictions.csv (one row/model/compound).')
    parser.add_argument('--output-dir', type=Path, default=Path('kpuu_statistics'))
    parser.add_argument('--kpuu-cutoff', type=float, default=.3)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if not args.predictions_csv.is_file():
        parser.error(f'File not found: {args.predictions_csv}. Point --predictions-csv to the transfer_predictions.csv already saved by script 13.')
    if not np.isfinite(args.kpuu_cutoff) or args.kpuu_cutoff <= 0:
        parser.error('--kpuu-cutoff must be finite and positive.')
    if args.bootstrap < 1 or args.permutations < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error('--bootstrap, --permutations, and --batch-size must be positive; --seed must be nonnegative.')
    try:
        models, ids, truth, scores, calls, thresholds, checks = load_predictions(args.predictions_csv, args.kpuu_cutoff)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    out = args.output_dir
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        out = out.with_name(out.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
        print(f'Preserving existing output; writing to {out}.', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Using {len(ids)} paired compounds and {len(models)} models; no retraining or inference.', flush=True)
    estimates, comparisons = analyze(models, truth, scores, calls, thresholds, args)
    estimates.to_csv(out / 'transfer_metrics.csv', index=False)
    comparisons.to_csv(out / 'paired_differences_with_pvalues.csv', index=False)
    counts = []
    low = truth < np.log10(args.kpuu_cutoff)
    for m, predicted, threshold in zip(models, calls, thresholds):
        counts.append(dict(model=m, positive_class='low_Kpuu', kpuu_cutoff=args.kpuu_cutoff,
                           predicted_logBB_cutoff=threshold,
                           tn=int((~low & ~predicted).sum()), fp=int((~low & predicted).sum()),
                           fn=int((low & ~predicted).sum()), tp=int((low & predicted).sum()),
                           n_low_Kpuu=int(low.sum()), n_high_Kpuu=int((~low).sum())))
    pd.DataFrame(counts).to_csv(out / 'classification_counts.csv', index=False)
    (out / 'methods.txt').write_text(methods_text(len(ids), args, comparisons), encoding='utf-8')
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    actual_output_dir=str(out.resolve()), models=models, n_compounds=len(ids),
                    input_sha256=hashlib.sha256(args.predictions_csv.read_bytes()).hexdigest(),
                    decision_thresholds=dict(zip(models, thresholds.tolist())),
                    completeness_checks=checks, model_refitting=False, new_inference=False,
                    threshold_fitting=False, bootstrap='ordinary paired compound percentile',
                    permutation='paired output swap, two-sided absolute delta, plus-one correction',
                    p_holm_family_sizes=comparisons.groupby('multiplicity_family').size().to_dict(),
                    p_holm_tests=len(comparisons),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, scipy=scipy.__version__))
    (out / 'statistics_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    display_columns = ['model', 'reference', 'metric', 'delta', 'ci_low', 'ci_high', 'p_value', 'p_holm', 'p_holm_family']
    (out / 'results.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Kp,uu statistics</title>'
        '<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse;margin-bottom:32px}'
        'th,td{padding:7px;border:1px solid #ddd;text-align:right}th{background:#eef2f6}</style></head><body>'
        '<h1>Kp,uu transfer statistics</h1><p>Delta = model minus reference. Positive values favor the model. '
        'p_holm adjusts all comparisons; p_holm_family adjusts correlation and classification families separately. '
        'Confidence intervals are unadjusted 95% paired bootstrap intervals.</p><h2>Model metrics</h2>'
        + estimates.to_html(index=False, float_format=lambda x: f'{x:.6g}')
        + '<h2>Paired comparisons</h2>' + comparisons[display_columns].to_html(index=False, float_format=lambda x: f'{x:.6g}')
        + '</body></html>', encoding='utf-8')
    print('\n' + comparisons[display_columns].to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print(f'\nSaved results: {out.resolve()}\nMain table: paired_differences_with_pvalues.csv', flush=True)


if __name__ == '__main__':
    main()
