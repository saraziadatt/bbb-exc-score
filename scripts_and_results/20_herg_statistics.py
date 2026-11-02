#!/usr/bin/env python3
"""Complete hERG metrics and paired inference from script 12's saved predictions; no training.

python 21_herg_statistics.py --predictions-csv herg_mtl_vs_mse/predictions.csv 
Only numpy and pandas are required. The default output is a new statistics directory
beside the prediction file. Run --help for resampling settings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


CLASSIFICATION = {
    'accuracy': ('Accuracy', 'higher'),
    'balanced_accuracy': ('Balanced accuracy', 'higher'),
    'blocker_precision': ('Blocker precision', 'higher'),
    'blocker_recall': ('Blocker recall (sensitivity)', 'higher'),
    'blocker_f1': ('Blocker F1', 'higher'),
    'nonblocker_specificity': ('Nonblocker recall (specificity)', 'higher'),
    'nonblocker_precision': ('Nonblocker precision (NPV)', 'higher'),
    'nonblocker_f1': ('Nonblocker F1', 'higher'),
    'roc_auc': ('ROC AUC', 'higher'),
    'average_precision': ('Average precision', 'higher'),
    'pr_auc_trapezoidal': ('PR AUC (trapezoidal)', 'higher'),
    'mcc': ('Matthews correlation coefficient', 'higher'),
    'cohens_kappa': ("Cohen's kappa", 'higher'),
    'wrong_side_rate': ('Misclassification rate', 'lower'),
    'wrong_side_distance': ('Mean boundary distance among misclassified compounds (pIC50)', 'lower'),
    'mean_distance_from_boundary': ('Mean absolute boundary distance, all predictions (pIC50)', 'descriptive'),
}
REGRESSION = {
    'mae': ('MAE (pIC50)', 'lower'),
    'mse': ('MSE (squared pIC50)', 'lower'),
    'rmse': ('RMSE (pIC50)', 'lower'),
    'median_absolute_error': ('Median absolute error (pIC50)', 'lower'),
    'r2': ('R-squared', 'higher'),
    'pearson_r': ('Pearson r', 'higher'),
    'spearman_r': ('Spearman rho', 'higher'),
    'blocker_mae': ('Blocker MAE (pIC50)', 'lower'),
    'nonblocker_mae': ('Nonblocker MAE (pIC50)', 'lower'),
    'blocker_rmse': ('Blocker RMSE (pIC50)', 'lower'),
    'nonblocker_rmse': ('Nonblocker RMSE (pIC50)', 'lower'),
}
SUBSET_LABELS = {'all': 'All compounds', 'measured': 'Measured IC50', 'screen_only': 'Screen-only'}


def ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def weighted_correlation(x, y, w):
    total = w.sum()
    keep = w > 0
    if total < 2 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return np.nan
    x0, y0 = x - (w * x).sum() / total, y - (w * y).sum() / total
    xx, yy = (w * x0 * x0).sum(), (w * y0 * y0).sum()
    return float(np.clip((w * x0 * y0).sum() / np.sqrt(xx * yy), -1, 1)) if xx > 0 and yy > 0 else np.nan


class SortedValues:
    """Sort once; integer-weight calculations exactly match explicit replicated rows."""
    def __init__(self, values, descending=False):
        self.order = np.argsort(-values if descending else values, kind='stable')
        ordered = values[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        group = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(values)]))
        self.group = np.empty(len(values), dtype=int)
        self.group[self.order] = group

    def sums(self, weights):
        return np.add.reduceat(weights[self.order], self.starts)

    def ranks(self, weights):
        sums = self.sums(weights)
        return (np.cumsum(sums) - (sums - 1) / 2)[self.group]

    def median(self, values, weights):
        total = int(weights.sum())
        if not total:
            return np.nan
        cumulative = np.cumsum(weights[self.order])
        positions = np.searchsorted(cumulative, [(total - 1) // 2, total // 2], side='right')
        return float(values[self.order[positions]].mean())


class Classification:
    def __init__(self, labels, reference, comparison, threshold):
        self.n = len(labels)
        self.labels = np.tile(np.asarray(labels, dtype=int), 2)
        self.scores = np.r_[reference, comparison]
        self.calls = self.scores >= threshold
        self.wrong = self.calls != self.labels
        self.distance = np.abs(self.scores - threshold)
        self.categories = 2 * self.labels + self.calls
        self.sorted = SortedValues(self.scores, descending=True)

    def counts(self, w):
        tn, fp, fn, tp = np.bincount(self.categories, weights=w, minlength=4)
        return dict(n=int(w.sum()), n_blocker=int(tp + fn), n_nonblocker=int(tn + fp),
                    predicted_blocker_n=int(tp + fp), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                    wrong_side_n=int(fp + fn))

    def evaluate(self, w):
        tn, fp, fn, tp = np.bincount(self.categories, weights=w, minlength=4)
        n, pos, neg = w.sum(), tp + fn, tn + fp
        sensitivity, specificity = ratio(tp, pos), ratio(tn, neg)
        accuracy = ratio(tp + tn, n)
        expected = ((tp + fp) * pos + (tn + fn) * neg) / (n * n)
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        auc = ap = pr_auc = np.nan
        if pos > 0 and neg > 0:
            mass = self.sorted.sums(w)
            positives = self.sorted.sums(w * self.labels)
            active = mass > 0
            mass, positives = mass[active], positives[active]
            total_seen, true_seen = np.cumsum(mass), np.cumsum(positives)
            false_seen = total_seen - true_seen
            recall, precision = true_seen / pos, true_seen / total_seen
            tpr, fpr = np.r_[0., recall], np.r_[0., false_seen / neg]
            auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
            ap = float(np.sum(positives * precision) / pos)
            rec, prec = np.r_[0., recall], np.r_[1., precision]
            pr_auc = float(np.sum(np.diff(rec) * (prec[1:] + prec[:-1]) / 2))
        return dict(accuracy=accuracy, balanced_accuracy=(sensitivity + specificity) / 2,
            blocker_precision=ratio(tp, tp + fp), blocker_recall=sensitivity,
            blocker_f1=ratio(2 * tp, 2 * tp + fp + fn), nonblocker_specificity=specificity,
            nonblocker_precision=ratio(tn, tn + fn), nonblocker_f1=ratio(2 * tn, 2 * tn + fp + fn),
            roc_auc=auc, average_precision=ap, pr_auc_trapezoidal=pr_auc,
            mcc=float((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator > 0 else 0.,
            cohens_kappa=ratio(accuracy - expected, 1 - expected), wrong_side_rate=ratio(fp + fn, n),
            wrong_side_distance=ratio((w * self.distance * self.wrong).sum(), fp + fn),
            mean_distance_from_boundary=float((w * self.distance).sum() / n))


class Regression:
    def __init__(self, targets, labels, reference, comparison):
        self.n = len(targets)
        self.y, self.labels = np.tile(targets, 2), np.tile(labels, 2)
        self.scores = np.r_[reference, comparison]
        self.error = np.abs(self.scores - self.y)
        self.squared_error = self.error ** 2
        self.target_order = SortedValues(self.y)
        self.score_order = SortedValues(self.scores)
        self.error_order = SortedValues(self.error)

    def evaluate(self, w):
        n = w.sum()
        mse = float((w * self.squared_error).sum() / n)
        centered = self.y - (w * self.y).sum() / n
        sst = (w * centered ** 2).sum()
        result = dict(mae=float((w * self.error).sum() / n), mse=mse, rmse=np.sqrt(mse),
            median_absolute_error=self.error_order.median(self.error, w),
            r2=float(1 - (w * self.squared_error).sum() / sst) if n > 1 and sst > 0 else np.nan,
            pearson_r=weighted_correlation(self.y, self.scores, w),
            spearman_r=weighted_correlation(self.target_order.ranks(w), self.score_order.ranks(w), w))
        for label, name in [(1, 'blocker'), (0, 'nonblocker')]:
            weights = w * (self.labels == label)
            result[name + '_mae'] = ratio((weights * self.error).sum(), weights.sum())
            result[name + '_rmse'] = np.sqrt(ratio((weights * self.squared_error).sum(), weights.sum()))
        return {metric: result[metric] for metric in REGRESSION}


def pair_weights(weights, swap=None):
    if swap is None:
        zeros = np.zeros_like(weights)
        return np.r_[weights, zeros], np.r_[zeros, weights]
    return np.r_[weights * (~swap), weights * swap], np.r_[weights * swap, weights * (~swap)]


def paired_metrics(engine, weights, swap=None):
    a, b = pair_weights(weights, swap)
    return engine.evaluate(a), engine.evaluate(b)


def load_predictions(path, reference, comparison, threshold):
    required = ['sample_id', 'outer_fold', 'source', 'true_value', 'true_class', 'model', 'prediction']
    raw = pd.read_csv(path, low_memory=False, dtype={'sample_id': str, 'outer_fold': str, 'model': str})
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f'Prediction CSV is missing columns: {sorted(missing)}')
    raw = raw[raw.model.isin([reference, comparison])].copy()
    if raw.empty or raw[required[:-4] + ['model']].isna().any().any():
        raise ValueError('Missing sample IDs, folds, sources, or model names.')
    if raw.duplicated(['sample_id', 'model']).any():
        raise ValueError('Duplicate sample_id/model pairs. Supply one out-of-fold prediction per compound per model.')
    for col in ['true_value', 'true_class', 'prediction']:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    if not raw.true_class.isin([0, 1]).all() or not np.isfinite(raw.prediction).all():
        raise ValueError('Require binary 0/1 labels and finite pIC50 predictions.')
    if np.isinf(raw.true_value).any():
        raise ValueError('Infinite quantitative targets are invalid; missing values must be blank/NaN.')
    a = raw[raw.model.eq(reference)].set_index('sample_id').sort_index()
    b = raw[raw.model.eq(comparison)].set_index('sample_id').sort_index()
    if a.empty or set(a.index) != set(b.index):
        raise ValueError('Models do not have exactly the same compound IDs. No unmatched rows will be dropped.')
    for col in ['outer_fold', 'source', 'true_class'] + (['smiles'] if 'smiles' in raw else []):
        if not a[col].equals(b[col]):
            raise ValueError(f'Models disagree on {col} for paired compounds.')
    if not np.allclose(a.true_value, b.true_value, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError('Models disagree on the measured pIC50 targets.')
    measured = a.true_value.notna()
    if not np.array_equal(a.true_class[measured], a.true_value[measured].ge(threshold).astype(int)):
        raise ValueError('Measured classes disagree with the specified pIC50 threshold.')
    if not (a.source.eq('IC50') == measured).all() or not a.source.isin(['IC50', 'screen_only']).all():
        raise ValueError('Expected source=IC50 for measured targets and source=screen_only for missing targets.')
    data = a[['outer_fold', 'source', 'true_value', 'true_class']].copy()
    data['reference_prediction'], data['comparison_prediction'] = a.prediction, b.prediction
    data = data.reset_index()
    manifest = path.parent / 'run_manifest.json'
    curated = path.parent / 'curated_herg.csv'
    completeness = []
    if manifest.exists():
        run = json.loads(manifest.read_text())
        if 'n' in run and int(run['n']) != len(data):
            raise ValueError('Prediction table is incomplete: compound count differs from run_manifest.json.')
        folds = run.get('settings', {}).get('outer_folds')
        if folds is not None and int(folds) != data.outer_fold.nunique():
            raise ValueError('Not all declared outer folds are present in predictions.csv.')
        completeness.append('run_manifest.json')
    if curated.exists():
        ids = pd.read_csv(curated, usecols=['sample_id'], dtype=str).sample_id
        if ids.duplicated().any() or set(ids) != set(data.sample_id):
            raise ValueError('Prediction IDs do not match curated_herg.csv; finish the original run first.')
        completeness.append('curated_herg.csv')
    if not completeness:
        print('Note: no neighboring manifest/curated table; pairing checked, but full-run completeness cannot be verified.', flush=True)
    return data, completeness


def build_engines(data, threshold):
    measured = data.true_value.notna().to_numpy()
    y, labels = data.true_value.to_numpy(float), data.true_class.to_numpy(int)
    a, b = data.reference_prediction.to_numpy(float), data.comparison_prediction.to_numpy(float)
    engines = []
    for subset, mask in [('all', np.ones(len(data), bool)), ('measured', measured), ('screen_only', ~measured)]:
        idx = np.flatnonzero(mask)
        if len(idx):
            engines.append(('classification', subset, idx, Classification(labels[idx], a[idx], b[idx], threshold), CLASSIFICATION))
    idx = np.flatnonzero(measured)
    if len(idx):
        engines.append(('regression', 'measured', idx, Regression(y[idx], labels[idx], a[idx], b[idx]), REGRESSION))
    return engines


def fold_tables(data, engines, reference, comparison):
    rows, counts = [], []
    for family, subset, idx, engine, definitions in engines:
        folds = data.outer_fold.to_numpy()[idx]
        for fold in ['pooled'] + sorted(set(folds)):
            w = np.ones(len(idx), dtype=int) if fold == 'pooled' else (folds == fold).astype(int)
            if not w.sum():
                continue
            wa, wb = pair_weights(w)
            for model, weights in [(reference, wa), (comparison, wb)]:
                values = engine.evaluate(weights)
                for metric, value in values.items():
                    rows.append(dict(family=family, subset=subset, outer_fold=fold, model=model,
                                     metric=metric, estimate=value, n=int(w.sum())))
                if family == 'classification':
                    counts.append(dict(subset=subset, outer_fold=fold, model=model, **engine.counts(weights)))
    return pd.DataFrame(rows), pd.DataFrame(counts)


def report_progress(label, i, total, started, last):
    now = time.monotonic()
    if i == 1 or i == total or i % 100 == 0 or now - last >= 15:
        elapsed = now - started
        remaining = elapsed / i * (total - i) / 60
        print(f'{label}: {i}/{total}; elapsed {elapsed / 60:.1f} min; estimated remaining {remaining:.1f} min.', flush=True)
        return now
    return last


def holm(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return result


def inference(data, engines, args):
    columns = [(family, subset, metric) for family, subset, _, _, definitions in engines for metric in definitions]
    point = np.empty((2, len(columns)))
    for model in range(2):
        point[model] = np.concatenate([list(paired_metrics(engine, np.ones(len(idx), int))[model].values())
                                      for _, _, idx, engine, _ in engines])
    # Independent deterministic streams allow more permutations without changing bootstrap results.
    bootstrap_seed, permutation_seed = np.random.SeedSequence(args.seed).spawn(2)
    rng = np.random.default_rng(bootstrap_seed)
    strata = list(data.groupby(['outer_fold', 'source', 'true_class'], sort=True).indices.values())
    bootstrap = np.empty((args.bootstrap, 2, len(columns)), dtype=float)
    started = last = time.monotonic()
    for i in range(args.bootstrap):
        weights = np.zeros(len(data), dtype=int)
        for idx in strata:
            weights[idx] = rng.multinomial(len(idx), np.full(len(idx), 1. / len(idx)))
        results = [paired_metrics(engine, weights[idx]) for _, _, idx, engine, _ in engines]
        for model in range(2):
            bootstrap[i, model] = np.concatenate([list(values[model].values()) for values in results])
        last = report_progress('Paired bootstrap', i + 1, args.bootstrap, started, last)
    rng = np.random.default_rng(permutation_seed)
    null = np.empty((args.permutations, len(columns)), dtype=float)
    started = last = time.monotonic()
    ones = np.ones(len(data), dtype=int)
    for i in range(args.permutations):
        swap = rng.integers(0, 2, len(data), dtype=np.int8).astype(bool)
        results = [paired_metrics(engine, ones[idx], swap[idx]) for _, _, idx, engine, _ in engines]
        null[i] = np.concatenate([np.array(list(b.values())) - np.array(list(a.values())) for a, b in results])
        last = report_progress('Paired permutations', i + 1, args.permutations, started, last)
    metric_rows, comparisons = [], []
    for j, (family, subset, metric) in enumerate(columns):
        definitions = CLASSIFICATION if family == 'classification' else REGRESSION
        label, direction = definitions[metric]
        idx = next(idx for f, s, idx, _, _ in engines if (f, s) == (family, subset))
        n_metric = len(idx)
        if metric.startswith('blocker_') and family == 'regression':
            n_metric = int(data.iloc[idx].true_class.eq(1).sum())
        elif metric.startswith('nonblocker_') and family == 'regression':
            n_metric = int(data.iloc[idx].true_class.eq(0).sum())
        meta = dict(family=family, subset=subset, metric=metric, label=label, better=direction,
                    n_subset=len(idx), n_metric=n_metric)
        for k, model in enumerate([args.reference, args.comparison]):
            valid = bootstrap[:, k, j][np.isfinite(bootstrap[:, k, j])]
            ci = np.quantile(valid, [0.025, 0.975]) if len(valid) else [np.nan, np.nan]
            metric_rows.append(dict(**meta, model=model, estimate=point[k, j], ci_low=ci[0], ci_high=ci[1],
                n_valid_bootstraps=len(valid), ci_degenerate=bool(len(valid) and ci[0] == ci[1])))
        delta = point[1, j] - point[0, j]
        differences = bootstrap[:, 1, j] - bootstrap[:, 0, j]
        valid = differences[np.isfinite(differences)]
        ci = np.quantile(valid, [0.025, 0.975]) if len(valid) else [np.nan, np.nan]
        perm = null[:, j][np.isfinite(null[:, j])]
        tolerance = 100 * np.finfo(float).eps * max(1., abs(delta)) if np.isfinite(delta) else 0.
        extreme = int(np.count_nonzero(np.abs(perm) >= abs(delta) - tolerance)) if np.isfinite(delta) else 0
        p = (extreme + 1) / (len(perm) + 1) if np.isfinite(delta) and len(perm) else np.nan
        # Dropping undefined randomized statistics changes the null distribution; withhold that p-value.
        if len(perm) != args.permutations:
            p = np.nan
        comparisons.append(dict(**meta, reference=args.reference, comparison=args.comparison,
            reference_estimate=point[0, j], comparison_estimate=point[1, j], delta=delta,
            delta_ci_low=ci[0], delta_ci_high=ci[1], n_valid_bootstraps=len(valid),
            p_raw=p, n_valid_permutations=len(perm), n_extreme=extreme,
            p_at_resolution_limit=bool(np.isfinite(p) and extreme == 0),
            mc_standard_error=np.sqrt(p * (1 - p) / (len(perm) + 1)) if np.isfinite(p) else np.nan))
    comparisons = pd.DataFrame(comparisons)
    comparisons['p_holm'] = np.nan
    for _, indices in comparisons.groupby('family').groups.items():
        comparisons.loc[indices, 'p_holm'] = holm(comparisons.loc[indices, 'p_raw'])
    return pd.DataFrame(metric_rows), comparisons, bootstrap, null, columns


def formatted_tables(metrics, comparisons, out, args):
    pages = []
    def number(x):
        return f'{x:.3f}' if np.isfinite(x) else 'NA'
    def interval(point, low, high):
        return f'{number(point)} [{number(low)}, {number(high)}]'
    def pnumber(x):
        return f'{x:.4g}' if np.isfinite(x) else 'NA'
    for (family, subset), group in comparisons.groupby(['family', 'subset'], sort=False):
        rows = []
        for row in group.itertuples():
            values = metrics[(metrics.family == family) & (metrics.subset == subset) & (metrics.metric == row.metric)].set_index('model')
            a, b = values.loc[args.reference], values.loc[args.comparison]
            rows.append({'Metric': row.label, 'N': row.n_metric,
                args.reference + ' [95% CI]': interval(a.estimate, a.ci_low, a.ci_high),
                args.comparison + ' [95% CI]': interval(b.estimate, b.ci_low, b.ci_high),
                'Difference [95% CI]': interval(row.delta, row.delta_ci_low, row.delta_ci_high),
                'p (raw)': pnumber(row.p_raw), 'p (Holm)': pnumber(row.p_holm)})
        table = pd.DataFrame(rows)
        table.to_csv(out / f'table_{family}_{subset}.csv', index=False)
        pages.append(f'<h2>{family.title()}: {SUBSET_LABELS[subset]}</h2>' + table.to_html(index=False, escape=True))
    note = (f'Positive class: hERG blocker. Predictions >= {args.threshold:g} pIC50 are classified as blockers. '
        f'Difference = {args.comparison} minus {args.reference}. CI: paired percentile bootstrap, '
        f'{args.bootstrap:,} resamples stratified by fold, label source and class. '
        f'P-values: two-sided paired prediction-swap permutation tests, {args.permutations:,} randomizations. '
        'Holm correction is applied across all classification metrics and subsets together, and separately '
        'across measured-subset regression metrics. NA denotes an undefined estimate/test. '
        'N is the eligible subset size; wrong-side distance uses each model\'s own misclassified compounds '
        '(counts are in counts.csv). CIs are conditional on these fitted out-of-fold models; training-set '
        'and chemical-series dependence are not captured. Degenerate bootstrap intervals can understate uncertainty.')
    document = ('<!doctype html><meta charset="utf-8"><title>hERG model comparison</title>'
        '<style>body{font:15px Arial,sans-serif;margin:32px;color:#17202a;max-width:1500px} '
        'table{border-collapse:collapse;width:100%;margin:18px 0 36px} th,td{padding:9px 12px;'
        'border-bottom:1px solid #d7dde3;text-align:right;font-variant-numeric:tabular-nums}'
        'th{background:#e8eef4} th:first-child,td:first-child{text-align:left} '
        'h2{font-size:20px;margin-top:36px} p{line-height:1.5} '
        '@media print{body{font-size:10px;margin:0} h2{break-before:page} tr{break-inside:avoid}}</style>'
        '<h1>hERG model comparison</h1><p>' + html.escape(note) + '</p>' + ''.join(pages))
    (out / 'tables.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reference', default='mse')
    parser.add_argument('--comparison', default='mtl')
    parser.add_argument('--threshold', type=float, default=5.)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--permutations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.predictions_csv.is_file():
        parser.error(f'Prediction file not found: {args.predictions_csv}')
    if min(args.bootstrap, args.permutations) < 1 or args.reference == args.comparison or not np.isfinite(args.threshold):
        parser.error('Use positive resample counts, different model names, and a finite threshold.')
    if min(args.bootstrap, args.permutations) < 1000:
        print('Exploratory setting: fewer than 1,000 resamples; p-value resolution and CI precision are limited.', flush=True)
    data, completeness = load_predictions(args.predictions_csv, args.reference, args.comparison, args.threshold)
    out = args.output_dir or args.predictions_csv.parent / 'statistics'
    if out.exists() and any(out.iterdir()):
        out = out.parent / (out.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=True)
    print(f'Paired compounds: {len(data):,}; measured IC50: {data.true_value.notna().sum():,}; '
          f'screen-only: {data.true_value.isna().sum():,}.\nResults: {out.resolve()}', flush=True)
    engines = build_engines(data, args.threshold)
    per_fold, counts = fold_tables(data, engines, args.reference, args.comparison)
    per_fold.to_csv(out / 'metrics_by_fold.csv', index=False)
    counts.to_csv(out / 'counts.csv', index=False)
    fold = per_fold[per_fold.outer_fold != 'pooled']
    summary = fold.groupby(['family', 'subset', 'model', 'metric'], sort=False).estimate.agg(['mean', 'std', 'count']).reset_index()
    summary.rename(columns={'std': 'sd', 'count': 'n_valid_folds'}).to_csv(out / 'fold_mean_sd.csv', index=False)
    metrics, comparisons, bootstrap, null, columns = inference(data, engines, args)
    metrics.to_csv(out / 'metrics_with_ci.csv', index=False)
    comparisons.to_csv(out / 'paired_comparisons.csv', index=False)
    np.savez_compressed(out / 'resampling_distributions.npz', bootstrap=bootstrap, permutation_deltas=null,
                        columns=np.asarray(['|'.join(c) for c in columns]))
    formatted_tables(metrics, comparisons, out, args)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        input_sha256=hashlib.sha256(args.predictions_csv.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, n=len(data), n_measured=int(data.true_value.notna().sum()),
        n_outer_folds=int(data.outer_fold.nunique()), completeness_checked_against=completeness,
        p_value_definition='(1 + count(abs(permuted_delta) >= abs(observed_delta))) / (1 + n_permutations)',
        p_resolution=1 / (args.permutations + 1), correction='Holm across classification metrics/subsets; separately across regression metrics',
        retraining=False, inference_scope='Conditional on the saved fitted models and evaluation compounds; assumes independent compound pairs.')
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    methods = (
        f'Saved out-of-fold predictions from the MSE and custom MTL hERG models were paired by compound '
        f'identifier, with matching fold assignments, labels, and measured targets verified before analysis. '
        f'No models were retrained. Classification used a fixed pIC50 threshold of {args.threshold:g} '
        f'(blocker: prediction >= threshold) and was evaluated on all {len(data):,} compounds and separately '
        f'on the measured-IC50 and screening-only subsets. Regression metrics used only the '
        f'{data.true_value.notna().sum():,} measured pIC50 targets. Point estimates were calculated from '
        f'pooled out-of-fold predictions; per-fold estimates and their mean and sample standard deviation '
        f'were also reported. Model-specific estimates and paired differences ({args.comparison} minus '
        f'{args.reference}) received 95% percentile confidence intervals from {args.bootstrap:,} paired '
        f'compound bootstrap samples stratified jointly by outer fold, label source, and observed class. '
        f'Two-sided permutation tests independently exchanged the two predictions within each compound '
        f'pair over {args.permutations:,} randomizations under a null of within-pair prediction '
        f'exchangeability. The p-value was (b+1)/(B+1), where b counted '
        f'absolute randomized differences at least as large as the observed absolute difference, with '
        f'floating-point tolerance. Raw p-values and Holm-adjusted values were reported, with all '
        f'classification metric/subset comparisons forming one family and all measured-subset regression '
        f'comparisons forming a second family. Average precision and trapezoidal PR AUC were reported '
        f'separately. Precision is undefined when there are no predicted positives; undefined metrics '
        f'are stored as NA, and valid bootstrap counts are reported. A permutation p-value is withheld '
        f'if any randomized value is undefined. Wrong-side distance is the mean absolute distance '
        f'from the threshold among each model\'s own misclassified compounds; it is not a measurement '
        f'of error against the observed pIC50. Mean distance across all predictions is descriptive and '
        f'has no universal preferred direction. Resampling inference is conditional on the fitted '
        f'models and observed strata sizes and does not account for variation from retraining, overlapping '
        f'cross-validation training sets, or dependence within chemical series. Degenerate bootstrap '
        f'intervals, particularly for boundary-valued rates, can understate uncertainty.\n')
    (out / 'methods.txt').write_text(methods, encoding='utf-8')
    selected = comparisons[(comparisons.subset == 'all') & comparisons.metric.isin(['balanced_accuracy', 'roc_auc', 'average_precision'])
                           | ((comparisons.family == 'regression') & comparisons.metric.eq('mae'))]
    print(selected[['subset', 'metric', 'reference_estimate', 'comparison_estimate', 'delta', 'p_raw', 'p_holm']].to_string(index=False))
    print(f'\nSaved complete metric tables, confidence intervals, paired p-values and methods to {out.resolve()}.', flush=True)


if __name__ == '__main__':
    main()
