#!/usr/bin/env python3
"""hERG: nested-CV custom MTL versus MSE, with genuinely additional screen-only labels.
Default run downloads public hERGCentral and ChEMBL data, curates, featurizes,
trains two LightGBM models, and evaluates matched outer-fold predictions.
  python 20_herg_mtl_vs_mse.py --output-dir herg_mtl_vs_mse
Downloads are cached separately; result directories must be empty/new.
"""
import argparse
import re
from datetime import datetime, timezone
# Shared implementation embedded in the two standalone training scripts.
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, r2_score)


def dump_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, Path): return str(x)
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def versions():
    out = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'optuna', 'rdkit']:
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return out


def output_dir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'{p} is not empty. Choose a new --output-dir; existing results are protected.')
    p.mkdir(parents=True, exist_ok=True)
    (p / 'fitted_models').mkdir()
    return p


def structure_key(smiles):
    # Match salts, protonation forms, tautomers, and stereoisomers conservatively.
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as std
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return None
        mol = std.Cleanup(mol)
        mol = std.FragmentParent(mol)
        mol = std.Uncharger().uncharge(mol)
        mol = std.TautomerEnumerator().Canonicalize(mol)
        if mol.GetNumHeavyAtoms() < 2 or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def repair_invalid_standardization(smiles):
    """Retain the parent charge only when uncharging creates an unreadable molecule."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as std
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, 'original_SMILES_rejected_by_RDKit'
        mol = std.FragmentParent(std.Cleanup(mol))
        candidate = std.Uncharger().uncharge(Chem.Mol(mol))
        candidate_smiles = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=False)
        if Chem.MolFromSmiles(candidate_smiles) is not None:
            # The failure was not caused by uncharging. Do not invent a chemical repair.
            return None, 'unreadable_standardized_SMILES_not_resolved_by_charge_retention'
        mol = std.TautomerEnumerator().Canonicalize(mol)
        if mol.GetNumHeavyAtoms() < 2 or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            return None, 'parent_fails_existing_structure_eligibility_rules'
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if Chem.MolFromSmiles(key) is None:
            return None, 'charge_retaining_parent_SMILES_rejected_by_RDKit'
        return key, 'parent_charge_retained_after_invalid_uncharging'
    except Exception as exc:
        return None, f'charge_retention_failed: {type(exc).__name__}: {exc}'


def standardize_series(series, cache_dir, out, label):
    """Cache canonical-tautomer SMILES and validate serialization before curation."""
    import logging
    import sqlite3
    import time
    from rdkit import Chem, rdBase
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Increment the algorithm version if structure_key's chemical operations change.
    cache_path = cache_dir / f'canonical_smiles_v1_rdkit_{rdBase.rdkitVersion}.sqlite'
    connection = sqlite3.connect(cache_path, timeout=30)
    connection.execute('CREATE TABLE IF NOT EXISTS structures (input_smiles TEXT PRIMARY KEY, canonical_smiles TEXT)')
    # A separate validation table upgrades old caches without repeating tautomer enumeration.
    # Keep the original standardized strings, including failures, for auditability.
    connection.execute('CREATE TABLE IF NOT EXISTS smiles_validation_v1 '
                       '(canonical_smiles TEXT PRIMARY KEY, is_valid INTEGER NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS standardization_repairs_v1 '
                       '(input_smiles TEXT, original_key TEXT, repaired_key TEXT, reason TEXT, '
                       'PRIMARY KEY (input_smiles, original_key))')
    raw = series.map(str)
    unique = list(pd.unique(raw))
    wanted = set(unique)
    lookup = {smiles: key for smiles, key in connection.execute('SELECT input_smiles, canonical_smiles FROM structures')
              if smiles in wanted}
    cached = len(lookup)
    pending = [smiles for smiles in unique if smiles not in lookup]
    print(f'{label}: {len(unique):,} unique SMILES; {cached:,} cached; {len(pending):,} to standardize.', flush=True)
    log_path = out / 'standardization_rdkit.log'
    print(f'RDKit diagnostics: {log_path.resolve()}', flush=True)
    logger = logging.getLogger('rdkit')
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    current_smiles = ['']
    class MoleculeContext(logging.Filter):
        def filter(self, record):
            record.input_smiles = current_smiles[0]
            return True
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s | input_SMILES=%(input_smiles)s'))
    handler.addFilter(MoleculeContext())
    for old in handlers: logger.removeHandler(old)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    rdBase.LogToPythonLogger()
    batch = []
    validation_batch = []
    repair_batch = []
    start = last_report = last_commit = time.monotonic()
    def commit():
        if batch:
            connection.executemany('INSERT OR REPLACE INTO structures VALUES (?, ?)', batch)
        if validation_batch:
            connection.executemany('INSERT OR REPLACE INTO smiles_validation_v1 VALUES (?, ?)', validation_batch)
        if repair_batch:
            connection.executemany('INSERT OR REPLACE INTO standardization_repairs_v1 VALUES (?, ?, ?, ?)', repair_batch)
        if batch or validation_batch or repair_batch:
            connection.commit()
            batch.clear()
            validation_batch.clear()
            repair_batch.clear()
    try:
        for i, smiles in enumerate(pending, 1):
            current_smiles[0] = smiles
            key = structure_key(smiles)
            lookup[smiles] = key
            batch.append((smiles, key))
            now = time.monotonic()
            if len(batch) >= 250 or now - last_commit >= 5:
                commit(); last_commit = now
            if i == len(pending) or i % 1000 == 0 or now - last_report >= 10:
                elapsed = max(now - start, 1e-9)
                eta = (len(pending) - i) / (i / elapsed) / 60
                print(f'{label}: {cached + i:,}/{len(unique):,} unique structures complete '
                      f'({i / elapsed:.1f} new/s; estimated remaining {eta:.1f} min).', flush=True)
                last_report = now
        # MolToSmiles can serialize a transformed molecule that default MolFromSmiles
        # subsequently rejects. Check NEW AND CACHED outputs before grouping or splitting.
        standardized = list(dict.fromkeys(k for k in lookup.values() if k is not None))
        needed = set(standardized)
        checked = {key: bool(valid) for key, valid in connection.execute(
            'SELECT canonical_smiles, is_valid FROM smiles_validation_v1') if key in needed}
        unchecked = [key for key in standardized if key not in checked]
        print(f'{label}: validating standardized SMILES; {len(checked):,} cached checks, '
              f'{len(unchecked):,} to check (no tautomer enumeration).', flush=True)
        last_report = time.monotonic()
        for i, key in enumerate(unchecked, 1):
            current_smiles[0] = '[standardized] ' + key
            try:
                valid = Chem.MolFromSmiles(key) is not None
            except Exception:
                valid = False
            checked[key] = valid
            validation_batch.append((key, int(valid)))
            now = time.monotonic()
            if len(validation_batch) >= 250 or now - last_commit >= 5:
                commit(); last_commit = now
            if i == len(unchecked) or i % 10000 == 0 or now - last_report >= 10:
                print(f'{label}: standardized SMILES checks {i:,}/{len(unchecked):,}.', flush=True)
                last_report = now
        repairs = {(sm, key): (repaired, reason) for sm, key, repaired, reason in connection.execute(
            'SELECT input_smiles, original_key, repaired_key, reason FROM standardization_repairs_v1')
            if sm in wanted}
        audited = []
        for smiles in unique:
            key = lookup[smiles]
            if key is not None and not checked[key]:
                current_smiles[0] = smiles
                if (smiles, key) not in repairs:
                    repaired, reason = repair_invalid_standardization(smiles)
                    repairs[(smiles, key)] = (repaired, reason)
                    repair_batch.append((smiles, key, repaired, reason))
                    if len(repair_batch) >= 250:
                        commit()
                repaired, reason = repairs[(smiles, key)]
                audited.append(dict(source=label, input_smiles=smiles, standardized_smiles=key,
                    retained_smiles=repaired, action='retained_charge' if repaired is not None else 'excluded',
                    reason=reason))
                lookup[smiles] = repaired
        audit = out / 'standardization_validation_audit.csv'
        pd.DataFrame(audited, columns=['source','input_smiles','standardized_smiles',
            'retained_smiles','action','reason']).to_csv(
            audit, mode='a', header=not audit.exists(), index=False)
        if audited:
            retained = sum(row['action'] == 'retained_charge' for row in audited)
            print(f'{label}: {len(audited):,} unique input SMILES had unreadable standardized '
                  f'representations; {retained:,} retained with validated parent charge, '
                  f'{len(audited) - retained:,} excluded. See {audit}.', flush=True)
    except KeyboardInterrupt:
        print(f'\nInterrupted. Completed standardizations are cached in {cache_path}. '
              'Rerun with a new output directory to reuse them.', flush=True)
        raise
    finally:
        try:
            commit()
        finally:
            connection.close()
            rdBase.LogToCppStreams()
            logger.removeHandler(handler)
            handler.close()
            for old in handlers: logger.addHandler(old)
            logger.setLevel(level)
            logger.propagate = propagate
    keys = raw.map(lookup)
    print(f'{label}: finished; {keys.isna().sum():,}/{len(keys):,} input records have no usable standardized structure.', flush=True)
    return keys


def custom_objective(y, labels, beta=1.0, buffer=0.25, margin=-1.0, direction=-1):
    """Exact saved BBB objective, with an explicit endpoint direction.

    direction=-1: class 1 should lie BELOW the margin (BBB nonpenetrant).
    direction=+1: class 1 should lie ABOVE the margin (hERG blocker, pIC50).
    L = .5 * sum_measured (f-y)^2
        + beta * sum_classes mean_class max(0, buffer - s*(f-margin))^2.
    The regression SUM is NOT divided by the number of measured compounds.
    Class weights use all training labels, including measured compounds.
    """
    y = np.asarray(y, float)
    labels = np.asarray(labels, int)
    measured = np.isfinite(y)
    signed = direction * (2 * labels - 1)
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any(): raise ValueError('Custom MTL needs both training classes.')
    weights = 1.0 / counts[labels]
    def objective(pred, dataset):
        grad = np.zeros(len(pred), dtype=float)
        hess = np.zeros(len(pred), dtype=float)
        grad[measured] = pred[measured] - y[measured]
        hess[measured] = 1.0
        violation = buffer - signed * (pred - margin)
        active = violation > 0
        grad[active] -= 2 * beta * weights[active] * signed[active] * violation[active]
        hess[active] += 2 * beta * weights[active]
        return grad, hess
    return objective


def fit_model(X, y, labels, kind, params, rounds, margin, direction, seed, threads):
    if kind == 'mse' and not np.isfinite(y).all():
        raise ValueError('MSE training must contain only genuinely measured targets.')
    p = dict(params)
    beta, buffer = p.pop('beta', 1.0), p.pop('margin_buffer', 0.25)
    objective = 'regression' if kind == 'mse' else custom_objective(y, labels, beta, buffer, margin, direction)
    p.update(objective=objective, metric='None', verbosity=-1, num_threads=threads,
             seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
             data_random_seed=seed, deterministic=True, force_col_wise=True,
             bagging_freq=1, feature_pre_filter=False)
    # Placeholders are inert: custom_objective closes over the original NaN mask.
    d = lgb.Dataset(X, label=np.where(np.isfinite(y), y, 0.0), free_raw_data=True)
    return lgb.train(p, d, num_boost_round=rounds)


def classify(pred, margin, direction):
    # Match the existing BBB strict '<' convention; hERG includes IC50=10 uM.
    return (pred < margin).astype(int) if direction < 0 else (pred >= margin).astype(int)


def metrics(y, labels, pred, margin, direction):
    y, labels, pred = np.asarray(y, float), np.asarray(labels, int), np.asarray(pred, float)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite model predictions.')
    binary = classify(pred, margin, direction)
    both = len(np.unique(labels)) == 2
    measured = np.isfinite(y)
    wrong = binary != labels
    result = dict(n=len(y), n_measured=int(measured.sum()), n_positive=int(labels.sum()),
        balanced_accuracy=balanced_accuracy_score(labels, binary) if both else np.nan,
        positive_precision=precision_score(labels, binary, zero_division=0),
        positive_recall=recall_score(labels, binary, zero_division=0),
        positive_f1=f1_score(labels, binary, zero_division=0),
        negative_recall=float(np.mean(binary[labels == 0] == 0)) if np.any(labels == 0) else np.nan,
        roc_auc=roc_auc_score(labels, direction * pred) if both else np.nan,
        average_precision=average_precision_score(labels, direction * pred) if both else np.nan,
        mcc=matthews_corrcoef(labels, binary) if both else np.nan,
        wrong_side_n=int(wrong.sum()),
        wrong_side_distance=float(np.mean(np.abs(pred[wrong] - margin))) if wrong.any() else np.nan,
        mae=np.nan, rmse=np.nan, r2=np.nan, spearman_r=np.nan)
    if measured.any():
        a, b = y[measured], pred[measured]
        result.update(mae=mean_absolute_error(a, b), rmse=np.sqrt(mean_squared_error(a, b)))
        if len(a) > 1 and np.ptp(a) > 0:
            result['r2'] = r2_score(a, b)
            if np.ptp(b) > 0: result['spearman_r'] = spearmanr(a, b).statistic
    return result


def metric_tables(predictions, out, margin, direction):
    rows = []
    for model, g in predictions.groupby('model', sort=True):
        for fold in ['pooled'] + sorted(g.outer_fold.unique().tolist(), key=str):
            f = g if fold == 'pooled' else g[g.outer_fold == fold]
            subsets = {'all': f, 'measured': f[f.true_value.notna()],
                       'classification_only': f[f.true_value.isna()]}
            for subset, v in subsets.items():
                if len(v): rows.append(dict(model=model, outer_fold=fold, subset=subset,
                    **metrics(v.true_value, v.true_class, v.prediction, margin, direction)))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'metrics.csv', index=False)
    fold = result[result.outer_fold != 'pooled']
    names = [c for c in result if c not in ['model', 'outer_fold', 'subset']]
    summary = fold.groupby(['model', 'subset'])[names].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / 'fold_mean_sd.csv', index=False)
    print(result[(result.outer_fold == 'pooled') & (result.subset == 'all')][
        ['model', 'n', 'balanced_accuracy', 'roc_auc', 'average_precision', 'mae']].to_string(index=False))
    return result

SCREEN_URLS = [
    'https://dataverse.harvard.edu/api/access/datafile/5740618',
    'https://huggingface.co/datasets/scikit-fingerprints/TDC_herg_central_at_10um/resolve/main/tdc_herg_central_at_10um.csv',
]
EPHYS = r'patch.?clamp|electrophysiolog|tail.current|whole.cell.current|voltage.clamp|ionworks|qpatch|syncropatch|patchxpress|potassium.current|K\+.current'
BINDING = r'displacement|radioligand|radiolabel|\[3H\]|\[35S\]|fluorescen|thallium|rubidium|Rb\+|mutant|mutation|selectivity'


def session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


def download(url, path):
    with session().get(url, timeout=(15, 90), stream=True) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + '.partial')
        with tmp.open('wb') as f:
            for block in r.iter_content(1024 * 1024): f.write(block)
        tmp.replace(path)


def chembl_table(endpoint, params, cache):
    from urllib.parse import urljoin
    path = cache / f'chembl_{endpoint}.csv'
    if path.exists(): return pd.read_csv(path, low_memory=False)
    url = f'https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json'
    rows, pages = [], []
    s = session()
    while url:
        print(f'Downloading ChEMBL {endpoint}: {len(rows)} records', flush=True)
        r = s.get(url, params=params, timeout=(15, 90)); r.raise_for_status()
        payload = r.json(); pages.append(payload['page_meta'])
        rows.extend(payload[{'activity': 'activities', 'assay': 'assays'}[endpoint]])
        next_page = payload['page_meta'].get('next')
        url = urljoin('https://www.ebi.ac.uk', next_page) if next_page else None
        params = None
    data = pd.DataFrame(rows)
    if data.empty: raise ValueError(f'ChEMBL returned no {endpoint} data.')
    data.to_csv(path, index=False)
    dump_json(cache / f'chembl_{endpoint}_pages.json', pages)
    return data


def load_sources(args, out):
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    if args.screen_csv:
        screen_path = args.screen_csv
    else:
        screen_path = cache / 'herg_central_10uM.csv'
        if not screen_path.exists():
            errors = []
            for url in SCREEN_URLS:
                try:
                    print('Downloading hERGCentral 10-uM screen...', flush=True)
                    download(url, screen_path)
                    dump_json(cache / 'screen_source.json', {'url': url, 'sha256': sha256(screen_path)})
                    break
                except Exception as e: errors.append(f'{url}: {e}')
            else: raise RuntimeError('Screen download failed. Supply --screen-csv with the TDC raw 10-uM table.\n' + '\n'.join(errors))
    first = screen_path.open().readline()
    screen = pd.read_csv(screen_path, sep='\t' if '\t' in first else ',', low_memory=False)
    if args.ic50_csv:
        quant = pd.read_csv(args.ic50_csv, low_memory=False)
        if not args.assay_csv: raise ValueError('--ic50-csv requires --assay-csv with ChEMBL confidence and relationship metadata.')
        assays = pd.read_csv(args.assay_csv, low_memory=False)
    else:
        quant = chembl_table('activity', dict(target_chembl_id='CHEMBL240', standard_type='IC50',
            standard_relation='=', standard_units='nM', assay_type__in='B,F', limit=1000), cache)
        assays = chembl_table('assay', dict(target_chembl_id='CHEMBL240', confidence_score=9,
            relationship_type='D', limit=1000), cache)
    sources = {'screen': {'path': str(screen_path), 'sha256': sha256(screen_path)},
        'ic50': {'path': str(args.ic50_csv or cache / 'chembl_activity.csv'),
                 'sha256': sha256(args.ic50_csv or cache / 'chembl_activity.csv')},
        'assays': {'path': str(args.assay_csv or cache / 'chembl_assay.csv'),
                   'sha256': sha256(args.assay_csv or cache / 'chembl_assay.csv')},
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'references': ['https://doi.org/10.1371/journal.pone.0199348',
                       'https://tdcommons.ai/single_pred_tasks/tox/#herg-central',
                       'https://www.ebi.ac.uk/chembl/api/data/docs']}
    if (cache / 'screen_source.json').exists() and not args.screen_csv:
        sources['screen']['download'] = json.loads((cache / 'screen_source.json').read_text())
    dump_json(out / 'data_sources.json', sources)
    return screen, quant, assays


def require_columns(df, names, label):
    missing = set(names) - set(df.columns)
    if missing: raise ValueError(f'{label} lacks required columns: {sorted(missing)}')


def curate(screen, quant, assays, args, out):
    # ChEMBL B/F categories do NOT reliably distinguish current measurements from ligand binding.
    require_columns(quant, ['canonical_smiles','standard_type','standard_relation','standard_units',
        'standard_value','assay_chembl_id','assay_description','target_chembl_id','assay_type',
        'data_validity_comment','potential_duplicate','standard_flag'], 'IC50 table')
    require_columns(assays, ['assay_chembl_id','confidence_score','relationship_type'], 'Assay table')
    valid_assays = set(assays.loc[(pd.to_numeric(assays.confidence_score, errors='coerce') == 9)
        & assays.relationship_type.eq('D'), 'assay_chembl_id'])
    description = quant.assay_description.fillna('')
    electrophys = description.str.contains(EPHYS, flags=re.I, regex=True)
    binding = description.str.contains(BINDING, flags=re.I, regex=True)
    if args.assay_allowlist:
        allowed = set(args.assay_allowlist.read_text().split())
        electrophys = quant.assay_chembl_id.isin(allowed)
    assay_audit = quant[['assay_chembl_id','assay_description','assay_type']].copy()
    assay_audit['explicit_electrophysiology'] = electrophys
    assay_audit['excluded_description'] = binding
    assay_audit['direct_single_target_confidence9'] = quant.assay_chembl_id.isin(valid_assays)
    assay_audit.drop_duplicates().to_csv(out / 'assay_audit.csv', index=False)
    values = pd.to_numeric(quant.standard_value, errors='coerce')
    valid = (quant.target_chembl_id.eq('CHEMBL240') & quant.standard_type.eq('IC50')
        & quant.standard_relation.eq('=') & quant.standard_units.eq('nM')
        & quant.assay_type.isin(['B','F']) & quant.assay_chembl_id.isin(valid_assays)
        & electrophys & ~binding & np.isfinite(values) & values.gt(0)
        & quant.data_validity_comment.fillna('').isin(['','Manually validated'])
        & pd.to_numeric(quant.potential_duplicate, errors='coerce').eq(0)
        & pd.to_numeric(quant.standard_flag, errors='coerce').eq(1))
    if 'assay_variant_mutation' in quant:
        valid &= quant.assay_variant_mutation.fillna('').eq('')
    quant = quant.copy()
    quant['included_record'] = valid
    quant.to_csv(out / 'ic50_record_audit.csv', index=False)
    q = quant.loc[valid].copy()
    q['pIC50'] = 9 - np.log10(values[valid])
    print(f'Standardizing {len(q)} selected exact IC50 records...', flush=True)
    q['key'] = standardize_series(q.canonical_smiles, args.cache_dir, out, 'IC50')
    q[q.key.isna()].to_csv(out / 'invalid_ic50_structures.csv', index=False)
    q = q[q.key.notna()]
    measured, q_audit = [], []
    for key, g in q.groupby('key', sort=True):
        ys = g.pIC50.to_numpy(float)
        reason = ''
        if np.ptp(ys) > args.max_ic50_range: reason = 'replicate_range_exceeds_limit'
        elif len(np.unique(ys >= 5.0)) > 1: reason = 'replicates_disagree_at_10uM'
        q_audit.append(dict(key=key, n=len(g), pIC50_min=ys.min(), pIC50_max=ys.max(), exclusion=reason))
        if not reason:
            y = float(np.median(ys))
            measured.append(dict(smiles=key, key=key, true_value=y, true_class=int(y >= 5.0),
                source='IC50', n_records=len(g),
                assay_ids=';'.join(sorted(g.assay_chembl_id.astype(str).unique()))))
    pd.DataFrame(q_audit).to_csv(out / 'ic50_compound_audit.csv', index=False)
    if not measured: raise ValueError('No usable exact electrophysiological IC50 compounds after curation.')
    m = pd.DataFrame(measured)
    # Recognize only the 10-uM readout; never silently substitute the binary TDC label.
    smiles_col = args.screen_smiles_col or next((c for c in ['smiles','SMILES','Drug'] if c in screen), None)
    value_col = args.screen_value_col or next((c for c in ['hERG_at_10uM','Y'] if c in screen), None)
    if not smiles_col or not value_col: raise ValueError('Specify --screen-smiles-col and --screen-value-col for this screen table.')
    v = pd.to_numeric(screen[value_col], errors='coerce')
    if set(v.dropna().unique()).issubset({0, 1}): raise ValueError('Binary input found; supply the raw 10-uM screening readout.')
    s = pd.DataFrame({'original_smiles': screen[smiles_col], 'inhibition_pct': -v if args.screen_sign == 'tdc-negative' else v})
    s = s[np.isfinite(s.inhibition_pct)]
    print(f'Standardizing {len(s)} single-concentration screen records...', flush=True)
    # Keep identical chemistry and eligibility rules; reuse completed standardizations.
    s['key'] = standardize_series(s.original_smiles, args.cache_dir, out, 'hERGCentral')
    s[s.key.isna()].to_csv(out / 'invalid_screen_structures.csv', index=False)
    s = s[s.key.notna()]
    aggregate = s.groupby('key').inhibition_pct.agg(['median','min','max','size']).reset_index()
    aggregate['true_class'] = np.where(aggregate['median'] > 70, 1,
        np.where(aggregate['median'] < 30, 0, np.nan))
    aggregate['conflicting_extremes'] = (aggregate['min'] < 30) & (aggregate['max'] > 70)
    aggregate['has_exact_IC50'] = aggregate.key.isin(m.key)
    aggregate.to_csv(out / 'screen_compound_audit.csv', index=False)
    overlap = aggregate.merge(m[['key','true_class']], on='key', suffixes=('_screen','_IC50'))
    overlap.to_csv(out / 'cross_source_overlap.csv', index=False)
    keep = aggregate.true_class.notna() & ~aggregate.conflicting_extremes & ~aggregate.has_exact_IC50
    s = aggregate.loc[keep].copy()
    s = pd.DataFrame(dict(key=s.key, smiles=s.key, true_value=np.nan,
        true_class=s.true_class.astype(int), source='screen_only', n_records=s['size'], assay_ids='hERGCentral_10uM'))
    if args.max_screen and len(s) > args.max_screen:
        from sklearn.model_selection import train_test_split
        s, _ = train_test_split(s, train_size=args.max_screen, stratify=s.true_class, random_state=args.seed)
    data = pd.concat([m, s], ignore_index=True).sort_values('key').reset_index(drop=True)
    data['sample_id'] = ['herg_' + hashlib.sha256(k.encode()).hexdigest()[:16] for k in data.key]
    if data.key.duplicated().any(): raise AssertionError('Duplicate structures remain.')
    if not len(s): raise ValueError('No additional screen-only compounds: this would not test the requested mixed-label use case.')
    data.groupby(['source','true_class']).size().rename('n').reset_index().to_csv(out / 'dataset_counts.csv', index=False)
    return data


def fingerprints(data, bits, out):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors, Crippen, Lipinski, rdMolDescriptors
    from scipy import sparse
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits, includeChirality=False)
    rows = []
    for i, sm in enumerate(data.smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            raise ValueError(f'Fingerprint input row {i} has unreadable SMILES: {sm!r}. '
                'Standardized SMILES validation must run before curation and fold assignment. '
                'No feature rows were skipped or replaced with zeros.')
        a = np.zeros(bits, dtype=np.uint8); b = np.zeros(167, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), a)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), b)
        small = [Descriptors.ExactMolWt(mol), Crippen.MolLogP(mol), Lipinski.NumHDonors(mol),
                 Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcTPSA(mol), Lipinski.NumRotatableBonds(mol)]
        rows.append(sparse.csr_matrix(np.concatenate([a, b, small]).astype(np.float32)[None, :]))
        if (i + 1) % 10000 == 0 or i + 1 == len(data):
            print(f'Fingerprints: {i + 1}/{len(data)}', flush=True)
    X = sparse.vstack(rows, format='csr')
    names = [f'ECFP4_{i}' for i in range(bits)] + [f'MACCS_{i}' for i in range(167)] + ['ExactMolWt','MolLogP','HBD','HBA','TPSA','RotBonds']
    sparse.save_npz(out / 'features.npz', X)
    dump_json(out / 'feature_columns.json', names)
    return X


def make_splits(data, n, seed, split):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    strata = data.source.astype(str) + '_' + data.true_class.astype(str)
    counts = strata.value_counts()
    if counts.min() < n: raise ValueError(f'Need at least {n} molecules in every source/class stratum: {counts.to_dict()}')
    if split == 'scaffold':
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        groups = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False) or s for s in data.smiles]
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        result = list(cv.split(np.zeros(len(data)), strata, groups))
    else:
        cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
        result = list(cv.split(np.zeros(len(data)), strata))
    for tr, va in result:
        for ix in [tr, va]:
            d = data.iloc[ix]
            if len(d[d.true_value.notna()].true_class.unique()) != 2 or len(d.true_class.unique()) != 2:
                raise ValueError('A split lacks measured examples of both classes; reduce folds or choose another split seed.')
    return result


def tune(X, data, splits, kind, args, seed, out, fold):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y, c = data.true_value.to_numpy(float), data.true_class.to_numpy(int)
    def objective(trial):
        params = dict(learning_rate=trial.suggest_float('learning_rate', .01, .3),
            num_leaves=trial.suggest_int('num_leaves', 20, 150),
            max_depth=trial.suggest_int('max_depth', 3, 12),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            bagging_fraction=trial.suggest_float('bagging_fraction', .5, 1),
            feature_fraction=trial.suggest_float('feature_fraction', .5, 1))
        if kind == 'mtl':
            params.update(beta=trial.suggest_float('beta', args.beta_min, args.beta_max, log=True),
                          margin_buffer=trial.suggest_float('margin_buffer', .15, .35, step=.02))
        mse_values, ba_values = [], []
        for i, (tr, va) in enumerate(splits):
            fit = tr[np.isfinite(y[tr])] if kind == 'mse' else tr
            model = fit_model(X[fit], y[fit], c[fit], kind, params, args.rounds, 5., 1,
                              seed+i, args.threads)
            pred = model.predict(X[va], num_threads=args.threads)
            m = np.isfinite(y[va])
            mse_values.append(mean_squared_error(y[va][m], pred[m]))
            ba_values.append(balanced_accuracy_score(c[va], pred >= 5.))
        trial.set_user_attr('inner_mse', float(np.mean(mse_values)))
        trial.set_user_attr('inner_ba', float(np.mean(ba_values)))
        return (np.mean(mse_values), np.mean(ba_values)) if kind == 'mtl' else np.mean(mse_values)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=['minimize','maximize'], sampler=sampler) if kind == 'mtl' else optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=args.trials)
    study.trials_dataframe().to_csv(out / f'trials_{kind}_outer_{fold}.csv', index=False)
    if kind == 'mtl':
        front = sorted(study.best_trials, key=lambda t: t.number)
        v = np.array([t.values for t in front], dtype=float)
        span = np.ptp(v, axis=0)
        distance = np.column_stack([v[:,0]-v[:,0].min(), v[:,1].max()-v[:,1]]) / np.where(span > 0, span, 1)
        selected = front[int(np.argmin(np.linalg.norm(distance, axis=1)))]
    else: selected = study.best_trial
    return dict(selected.params), dict(selected_trial=selected.number, objectives=selected.values,
                                      inner_mse=selected.user_attrs['inner_mse'], inner_ba=selected.user_attrs['inner_ba'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('herg_mtl_vs_mse'))
    p.add_argument('--cache-dir', type=Path, default=Path('herg_mtl_vs_mse/herg_downloads'))
    p.add_argument('--screen-csv', type=Path)
    p.add_argument('--ic50-csv', type=Path, help='Optional cached ChEMBL activity export, in the API schema.')
    p.add_argument('--assay-csv', type=Path, help='Matching ChEMBL assay export, required with --ic50-csv.')
    p.add_argument('--screen-smiles-col')
    p.add_argument('--screen-value-col')
    p.add_argument('--screen-sign', choices=['tdc-negative','percent-inhibition'], default='tdc-negative')
    p.add_argument('--assay-allowlist', type=Path, help='Optional manually reviewed electrophysiology assay IDs, one per line; other validity filters still apply.')
    p.add_argument('--max-ic50-range', type=float, default=1.0, help='Largest permitted within-structure pIC50 range.')
    p.add_argument('--max-screen', type=int, default=0, help='0 uses every eligible screen-only compound; positive values give an explicitly subsampled exploratory run.')
    p.add_argument('--prepare-only', action='store_true', help='Download/curate/audit the data and stop before model training.')
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--split', choices=['random','scaffold'], default='random')
    p.add_argument('--trials', type=int, default=30, help='Same trial budget per model per outer fold.')
    p.add_argument('--beta-min', type=float, default=0.5)
    p.add_argument('--beta-max', type=float, default=5000.0,
        help='Log-uniform MTL weight range; wider than BBB tuning because regression is a SUM and class penalties are means.')
    p.add_argument('--rounds', type=int, default=500)
    p.add_argument('--bits', type=int, default=2048)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.trials, args.rounds, args.bits) < 1 or min(args.outer_folds, args.inner_folds) < 2:
        p.error('Use positive trials/rounds/bits and at least two CV folds.')
    if args.max_screen < 0 or args.max_ic50_range < 0: p.error('Invalid curation limits.')
    if not 0 < args.beta_min <= args.beta_max: p.error('Require 0 < beta-min <= beta-max.')
    out = output_dir(args.output_dir)
    screen, quant, assays = load_sources(args, out)
    data = curate(screen, quant, assays, args, out)
    data.to_csv(out / 'curated_herg.csv', index=False)
    dump_json(out / 'run_manifest.json', dict(settings=vars(args), versions=versions(), n=len(data),
        n_measured=int(data.true_value.notna().sum()),
        positive_class='hERG blocker: measured pIC50 >= 5; screen inhibition >70% at 10 uM',
        negative_screen='inhibition <30% at 10 uM; intermediate screen compounds excluded',
        target='pIC50=-log10(IC50 in mol/L)', classification_margin=5.,
        loss='0.5*sum_measured_squared_error + beta*sum_class_mean_squared_hinge',
        structure_validation='Default RDKit sanitization before deduplication/splitting; invalid uncharging falls back to a validated charge-retaining parent tautomer; unresolved failures excluded; all affected inputs audited.',
        curation_note='Description-filtered electrophysiology; assay_audit.csv enables review. Assay heterogeneity remains.'))
    if args.prepare_only:
        print(f'Curated {len(data)} unique compounds. Inspect audit tables in {out}.'); return
    outer = make_splits(data, args.outer_folds, args.seed, args.split)
    data['outer_fold'] = -1
    for fold, (_, test) in enumerate(outer): data.loc[test, 'outer_fold'] = fold
    data.to_csv(out / 'curated_herg.csv', index=False)
    X = fingerprints(data, args.bits, out)
    predictions, parameters, training_counts = [], [], []
    for fold, (tr, te) in enumerate(outer):
        train = data.iloc[tr].reset_index(drop=True)
        inner = make_splits(train, args.inner_folds, args.seed+1000*fold, args.split)
        inner_assignments = pd.DataFrame({'sample_id': train.sample_id, 'inner_fold': -1})
        for j, (_, va) in enumerate(inner): inner_assignments.loc[va, 'inner_fold'] = j
        inner_assignments.to_csv(out / f'inner_folds_outer_{fold}.csv', index=False)
        yt, ct = train.true_value.to_numpy(float), train.true_class.to_numpy(int)
        for kind in ['mse','mtl']:
            print(f'Outer fold {fold}: tuning {kind.upper()}', flush=True)
            params, selected = tune(X[tr], train, inner, kind, args, args.seed+1000*fold, out, fold)
            fit = np.flatnonzero(np.isfinite(yt)) if kind == 'mse' else np.arange(len(train))
            model = fit_model(X[tr][fit], yt[fit], ct[fit], kind, params, args.rounds, 5., 1,
                              args.seed+1000*fold, args.threads)
            pred = model.predict(X[te], num_threads=args.threads)
            g = data.iloc[te][['sample_id','smiles','outer_fold','true_value','true_class','source']].copy()
            g['model'], g['prediction'] = kind, pred
            g['predicted_class'] = (pred >= 5.).astype(int)
            g['absolute_error'] = np.abs(g.true_value - pred)
            predictions.append(g)
            model.save_model(str(out / 'fitted_models' / f'{kind}_outer_{fold}.txt'))
            parameters.append(dict(model=kind, outer_fold=fold, parameters=params, **selected))
            training_counts.append(dict(model=kind, outer_fold=fold, n_train=len(fit),
                n_train_measured=int(np.isfinite(yt[fit]).sum()), n_test=len(te)))
            pd.concat(predictions).to_csv(out / 'predictions.csv', index=False)
            pd.DataFrame(training_counts).to_csv(out / 'training_counts.csv', index=False)
            dump_json(out / 'selected_parameters.json', parameters)
    predictions = pd.concat(predictions, ignore_index=True)
    if predictions.duplicated(['model','sample_id']).any() or len(predictions) != 2*len(data):
        raise AssertionError('Expected exactly one OOF prediction per compound per model.')
    metric_tables(predictions, out, 5., 1)
    (out / 'methods.txt').write_text(
        f'An hERG mixed-label benchmark combined exact ChEMBL IC50 measurements with single-concentration '
        f'hERGCentral screening data at 10 micromolar. ChEMBL records were restricted to direct single-target '
        f'assignments (confidence 9), exact positive IC50 values in nM, and descriptions explicitly indicating '
        f'electrophysiological measurements; flagged, potentially duplicated, mutant, binding-displacement, '
        f'and indirect-readout records were excluded. Structures were standardized to nonstereochemical '
        f'parent tautomer representations. Standardized SMILES were reparsed with default RDKit '
        f'sanitization before deduplication and splitting. When uncharging produced unreadable SMILES, '
        f'a validated charge-retaining parent tautomer was used; unresolved failures were excluded. '
        f'All affected inputs were audited. '
        f'Quantitative replicates were summarized by median pIC50, excluding '
        f'groups spanning more than {args.max_ic50_range} log units or disagreeing across pIC50=5. Screen '
        f'labels were assigned from median inhibition (>70% blocker; <30% nonblocker), excluding intermediate '
        f'or conflicting groups. Exact quantitative data took precedence for overlapping structures; '
        f'only additional compounds supplied screen-only labels. Both LightGBM models used radius-2 '
        f'{args.bits}-bit Morgan fingerprints, 167 MACCS keys, and six RDKit physicochemical descriptors. '
        f'The MSE comparator used measured compounds only; custom MTL additionally used screening-only '
        f'compounds through its class-count-weighted squared hinge penalty. Nested {args.outer_folds}-fold '
        f'outer/{args.inner_folds}-fold inner {args.split} cross-validation used identical splits and '
        f'{args.trials} optimization trials per model, with {args.rounds} boosting rounds. MSE tuning '
        f'minimized measured-target MSE; MTL tuning minimized MSE and maximized balanced accuracy, '
        f'selecting the normalized Pareto point nearest the ideal. The classification boundary was fixed '
        f'at pIC50=5; alpha was fixed at 1, beta was sampled log-uniformly from {args.beta_min} '
        f'to {args.beta_max}, and the hinge buffer was sampled from 0.15 to 0.35 in 0.02 increments. '
        f'Classification was evaluated on all held-out compounds and separately by label '
        f'source; regression was evaluated on measured compounds only. Average precision summarized '
        f'precision-recall performance. Results were reported per fold and from pooled out-of-fold '
        f'predictions. This benchmark retains cross-assay heterogeneity and does not equate screening '
        f'readouts with measured IC50 values.\n'
        + (f'The screen-only pool was capped at {args.max_screen} compounds by stratified random sampling.\n' if args.max_screen else ''))
    print(f'\nSaved data provenance, curation audits, models, predictions, metrics, and methods.txt to {out}')


if __name__ == '__main__':
    main()
