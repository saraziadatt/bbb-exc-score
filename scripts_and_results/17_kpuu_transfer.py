#!/usr/bin/env python3
"""All-in-one Kp,uu transfer evaluation using existing saved BBB regressors.
Downloads/curates rat Kp,uu data, excludes development overlap, generates matching
Mordred (including 3D), ECFP and MACCS features, predicts, and evaluates.
No separate descriptor or prediction script is needed. No training on Kp,uu.
3D preparation matches notebook 01: seeded OpenChemLib from the same DataWarrior
JAR, MMFF94s+ minimization, and Mordred -3 on its SDF. Requires Java with jdk.compiler.
"""
import argparse
import base64
import os
import shutil
import subprocess
import sys
import zipfile
import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.base import BaseEstimator, TransformerMixin



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
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'rdkit', 'mordredcommunity', 'mordred', 'joblib']:
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return out


def output_dir(path):
    p = Path(path)
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(str(p))
    if p.exists() and any(p.iterdir()):
        # Keep prior results; reruns need no manual cleanup.
        p = p / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    p.mkdir(parents=True, exist_ok=True)
    return p



@lru_cache(maxsize=100000)
def structure_key(smiles):
    """Fast, non-enumerative parent/tautomer/stereoisomer matching key, NOT SMILES."""
    from rdkit import Chem
    from rdkit.Chem import rdMolHash
    from rdkit.Chem.MolStandardize import rdMolStandardize as std
    hash_type = rdMolHash.HashFunction.HetAtomTautomerv2
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return None
        mol = std.FragmentParent(std.Cleanup(mol), skipStandardize=True)
        mol = std.Uncharger().uncharge(mol)
        if mol.GetNumHeavyAtoms() < 2 or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            return None
        Chem.RemoveStereochemistry(mol)
        return rdMolHash.MolHash(mol, hash_type)
    except Exception:
        return None



LAWRENZ_URL = 'https://ndownloader.figshare.com/files/41040571'
LAWRENZ_DOI = 'https://doi.org/10.1021/acs.jcim.3c00150'
TUNED_THRESHOLD_MODELS = {'lightgbm_mse_regression_threshold'}


def get_kpuu(args, out):
    if args.input:
        raw = pd.read_csv(args.input, low_memory=False)
        require = [args.smiles_col, args.kpuu_col]
        if any(c not in raw for c in require): raise ValueError(f'Input requires {require}.')
        raw = raw.rename(columns={args.smiles_col: 'smiles', args.kpuu_col: 'Kpuu'})
        raw['source_id'] = raw.index.astype(str)
        source = {'file': str(args.input), 'sha256': sha256(args.input)}
        if args.species_col in raw: raw['species'] = raw[args.species_col].astype(str).str.lower().str.strip()
        elif args.species != 'any':
            raise ValueError('Species column missing. Supply --species-col, or explicitly --species any for an input already curated by species.')
        else: raw['species'] = 'not_reported'
    else:
        import requests
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        path = args.cache_dir / 'Lawrenz_2023_SI_002.xlsx'
        if not path.exists():
            print('Downloading published Kp,uu supplement...', flush=True)
            r = requests.get(LAWRENZ_URL, timeout=(15, 90)); r.raise_for_status()
            path.write_bytes(r.content)
        raw = pd.read_excel(path).rename(columns={'Compound_Structure': 'smiles',
            'Kpuu species': 'species', 'ID': 'source_id'})
        required = {'smiles','species','Kpuu','source_id'}
        if not required.issubset(raw): raise ValueError('Unexpected Lawrenz supplement schema.')
        # Keep only measured endpoint and identifiers, not published model/QM predictions.
        raw = raw[['source_id','smiles','Kpuu','species']].copy()
        raw['species'] = raw.species.astype(str).str.lower().str.strip()
        source = {'url': LAWRENZ_URL, 'doi': LAWRENZ_DOI, 'sha256': sha256(path)}
    raw['reason'] = ''
    if args.species != 'any': raw.loc[raw.species != args.species.lower(), 'reason'] = 'other_species'
    elif raw.species.nunique() > 1:
        raise ValueError('Multiple species present. Select one --species rather than pooling species.')
    raw['Kpuu_numeric'] = pd.to_numeric(raw.Kpuu, errors='coerce')
    invalid = ~np.isfinite(raw.Kpuu_numeric) | (raw.Kpuu_numeric <= 0)
    raw.loc[invalid, 'reason'] = 'missing_nonpositive_or_censored_Kpuu'
    # An optional separate qualifier is honored; inequalities are never treated as exact values.
    if args.relation_col:
        if args.relation_col not in raw: raise ValueError('Requested relation column is absent.')
        raw.loc[~raw[args.relation_col].fillna('').isin(['=', '']), 'reason'] = 'censored_Kpuu'
    raw['key'] = raw.smiles.map(structure_key)
    raw.loc[raw.key.isna(), 'reason'] = 'invalid_structure'
    development = pd.read_csv(args.development_csv, low_memory=False)
    if args.development_smiles_col not in development: raise ValueError('Development SMILES column absent.')
    print('Checking overlap with the full BBB development set...', flush=True)
    dev_keys = development[args.development_smiles_col].map(structure_key)
    if dev_keys.isna().any():
        raise ValueError('Some development structures cannot be standardized; overlap exclusion would be incomplete.')
    raw.loc[raw.key.isin(set(dev_keys)), 'reason'] = 'BBB_development_overlap'
    raw.to_csv(out / 'kpuu_curation_audit.csv', index=False)
    keep = raw[raw.reason == ''].copy()
    rows = []
    for key, g in keep.groupby('key', sort=True):
        row = g.iloc[0].to_dict()
        row['Kpuu'] = float(10 ** np.median(np.log10(g.Kpuu_numeric.to_numpy(float))))
        row['log10_Kpuu'] = float(np.log10(row['Kpuu']))
        row['n_measurements'] = len(g)
        row['source_id'] = ';'.join(sorted(g.source_id.astype(str).unique()))
        row['sample_id'] = 'kpuu_' + hashlib.sha256(key.encode()).hexdigest()[:16]
        rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty: raise ValueError('No exact, positive Kp,uu values remain after species filtering and development-overlap exclusion.')
    data = data.drop(columns=['reason','Kpuu_numeric'])
    if data.sample_id.duplicated().any(): raise AssertionError('Duplicate generated sample IDs.')
    data.to_csv(out / 'kpuu_eligible.csv', index=False)
    dump_json(out / 'data_source.json', source)
    print(f'{len(data)} independent structures remain for Kp,uu transfer testing.', flush=True)
    return data


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

ShakerRegressionPreprocessor = ShakerFeaturePreprocessor



def saved_input_names(saved):
    """Collect raw columns needed by saved preprocessing, including intermediate filters."""
    names, seen = [], set()
    def visit(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        fields = ('feature_names', 'feature_columns', 'selected_features',
                  'feature_names_in_', 'input_columns_', 'kept_columns_',
                  'columns_', 'selected_columns')
        for field in fields:
            value = obj.get(field) if isinstance(obj, dict) else getattr(obj, field, None)
            if value is not None and not isinstance(value, str):
                names.extend(normalize_names(value))
        for field in ('preprocessor', 'feature_selector', 'selector', 'complete_filter',
                      'pearson_filter', 'pcc_filter', 'imputer', 'scaler'):
            visit(obj.get(field) if isinstance(obj, dict) else getattr(obj, field, None))
        if hasattr(obj, 'steps') and obj.steps:
            visit(obj.steps[0][1])
    visit(saved)
    return list(dict.fromkeys(names))


def load_models(args, models):
    """Load existing fold artifacts without depending on another project script."""
    import joblib
    import lightgbm as lgb
    import sys
    # Old training scripts pickled their locally defined classes as __main__.*.
    for name in ('TrainingColumnFilter', 'DataFrameMedianImputer', 'CorrelationFilter',
                 'CompleteMeasuredTrainingFilter', 'TargetAwarePearsonFilter',
                 'Model4ShakerPreprocessor', 'CompleteTrainingDescriptorFilter',
                 'ExactShakerFeatureSelector', 'ShakerFeaturePreprocessor',
                 'ShakerRegressionPreprocessor'):
        setattr(sys.modules['__main__'], name, globals()[name])
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f'Saved BBB model directory does not exist: {args.model_dir.resolve()}')
    dev = pd.read_csv(args.development_csv, usecols=[args.fold_col])
    folds = pd.to_numeric(dev[args.fold_col], errors='raise').to_numpy(float)
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError('Development outer-fold IDs must be finite integers.')
    required_folds = sorted(set(folds.astype(int)))
    records = []
    for name in models:
        native = sorted(args.model_dir.glob('custom_mtl_bab_outer_*.txt')) if name == 'custom_mtl_bab' else []
        paths = native
        if not paths:
            families = [name]
            if name == 'lightgbm_mse_regression':
                families.append('lightgbm_mse_regression_threshold')
            elif name == 'lightgbm_mse_regression_threshold':
                families.append('lightgbm_mse_regression')
            if name == 'logbb_pred_shaker_retrained':
                families.append('logbb_pred_matched')
            for family in families:
                for stem in ('outer', 'fold'):
                    for suffix in ('joblib', 'pkl', 'pickle'):
                        candidates = sorted(args.model_dir.glob(f'{family}_{stem}_*.{suffix}'))
                        paths = [p for p in candidates if not any(s in p.stem for s in ('preprocessor', 'selector', 'thresholds'))]
                        if paths: break
                    if paths: break
                if paths: break
        if not paths:
            raise FileNotFoundError(f'No saved fold models for {name!r} in {args.model_dir.resolve()}')
        actual_folds = [fold_number(path) for path in paths]
        if sorted(actual_folds) != required_folds:
            raise ValueError(f'{name}: expected exactly one saved model for each development fold '
                             f'{required_folds}; found {actual_folds}.')
        for path in sorted(paths, key=fold_number):
            record = dict(model=name, fold=fold_number(path), path=path, native=bool(native))
            record['decision_threshold'] = -1.
            record['threshold_variant'] = 'fixed'
            if name in TUNED_THRESHOLD_MODELS:
                threshold, threshold_path = load_tuned_threshold_for_fold(
                    args.model_dir, name, record['fold'], args.threshold_pattern)
                record.update(decision_threshold=threshold, threshold_variant='inner_tuned',
                              threshold_source=threshold_path, threshold_sha256=sha256(threshold_path))
            if native:
                fold = record['fold']
                preferred = args.model_dir / f'custom_mtl_bab_outer_{fold}_preprocessor.joblib'
                if preferred.exists():
                    prep_path = preferred
                else:
                    candidates = sorted(args.model_dir.glob(f'custom_mtl_bab_outer_{fold}_*preprocessor.joblib'))
                    if not candidates:
                        candidates = sorted(args.model_dir.glob(f'custom_mtl_bab_outer_{fold}_*selector.joblib'))
                    if len(candidates) != 1:
                        raise FileNotFoundError(f'{path.name}: expected one matching saved preprocessor; found {candidates}.')
                    prep_path = candidates[0]
                processor = joblib.load(prep_path)
                record.update(estimator=lgb.Booster(model_file=str(path)), processor=processor,
                              preprocessor_path=prep_path, required_names=saved_input_names(processor))
            else:
                estimator, metadata = unwrap_loaded(joblib.load(path))
                if getattr(estimator, '_estimator_type', None) == 'classifier':
                    raise ValueError(f'{path}: transfer requires a regressor producing logBB, not a classifier.')
                names = feature_names_from_estimator(estimator)
                if names is None or all(re.fullmatch(r'(?:Column|feature)_?\d+', n) for n in names):
                    for key in ('feature_names', 'feature_columns', 'selected_features'):
                        if metadata.get(key) is not None:
                            names = normalize_names(metadata[key]); break
                record.update(estimator=estimator, feature_names=names, required_names=names)
            if not record['required_names']:
                raise ValueError(f'{path}: cannot recover saved descriptor names/order. Save the training feature metadata alongside this model.')
            record['sha256'] = sha256(path)
            if native: record['preprocessor_sha256'] = sha256(record['preprocessor_path'])
            records.append(record)
        print(f'Loaded {len(paths)} saved fold models for {name}.', flush=True)
    return records


def load_tuned_threshold_for_fold(model_dir, model_name, fold, threshold_pattern=None):
    """Use the same per-fold inner-CV JSON metadata as external script 10."""
    pattern = (threshold_pattern.format(fold=fold, model=model_name) if threshold_pattern
               else f'{model_name}_outer_{fold}.json')
    matches = []
    for path in sorted(model_dir.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
            saved_fold, threshold = float(payload['outer_fold']), float(payload['tuned_threshold'])
        except (OSError, ValueError, TypeError, KeyError):
            continue
        if saved_fold == fold and np.isfinite(threshold):
            matches.append((threshold, path))
    if len(matches) != 1:
        raise ValueError(f'{model_name}, fold {fold}: expected exactly one valid inner-tuned '
                         f'threshold JSON matching {pattern!r} in {model_dir}; found {len(matches)}. '
                         'The script will not substitute -1 or tune a threshold on Kp,uu.')
    return matches[0]


def descriptor_setup(required_names):
    """Resolve exact saved names; never fabricate missing descriptor columns."""
    try:
        from mordred import Calculator, descriptors
    except ImportError as exc:
        raise ImportError('Automatic descriptor generation needs Mordred. In this environment run '
                          '`python -m pip install mordredcommunity`, then rerun this same command.') from exc
    registry = {str(d): d for d in Calculator(descriptors, ignore_3D=False).descriptors}
    chosen, ecfp, maccs, unknown = [], [], [], []
    for name in required_names:
        match = re.fullmatch(r'ECFP_(\d+)', name)
        if match and 0 <= int(match[1]) < 2048:
            ecfp.append((name, int(match[1]))); continue
        match = re.fullmatch(r'MACCS_(\d+)', name)
        if match and 0 <= int(match[1]) < 167:
            maccs.append((name, int(match[1]))); continue
        if name in registry:
            chosen.append(registry[name])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError('Saved features are not recognized as Mordred, ECFP_0..2047, or MACCS_0..166: '
                         f'{unknown[:30]}. Check the saved feature definitions; missing columns will not be fabricated.')
    return Calculator(chosen, ignore_3D=False), ecfp, maccs



# Coordinate settings and Java helper copied from notebook 01.
BASE_SEED = 20260915
DIELECTRIC_CONSTANT = 1

# Use the jar from the SAME DataWarrior version used for the manuscript.
# Leave as None to search the macOS app and launcher-managed update folder.
# Typical macOS path: /Applications/DataWarrior.app/Contents/Java/update/datawarrior_all.jar


def find_openchemlib_jar(explicit_path=None):
    # DataWarrior may package OpenChemLib in openchemlib.jar OR in a larger
    # DataWarrior jar. Test jar contents instead of relying on the filename.
    required_classes = {
        'org/openmolecules/chem/conf/gen/ConformerGenerator.class',
        'com/actelion/research/chem/forcefield/mmff/ForceFieldMMFF94.class',
    }

    def contains_required_classes(path):
        try:
            with zipfile.ZipFile(path) as jar:
                names = set(jar.namelist())
            return required_classes.issubset(names)
        except (OSError, zipfile.BadZipFile):
            return False

    candidates = []
    if explicit_path:
        explicit_path = Path(explicit_path).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(f'OPEN_CHEM_LIB_JAR does not exist: {explicit_path}')
        if not contains_required_classes(explicit_path):
            raise ValueError(f'The selected jar does not contain the required OpenChemLib classes: {explicit_path}')
        return explicit_path.resolve()
    if os.environ.get('OPENCHEMLIB_JAR'):
        candidates.append(Path(os.environ['OPENCHEMLIB_JAR']).expanduser())


    # Spotlight fallback catches launcher-managed jars stored in an unexpected
    # macOS user directory without recursively scanning the entire home folder.
    mdfind = shutil.which('mdfind')
    if mdfind:
        query = "kMDItemFSName == '*openchemlib*.jar'cd || kMDItemFSName == '*datawarrior*.jar'cd"
        result = subprocess.run([mdfind, query], capture_output=True, text=True, check=False)
        candidates.extend(Path(line) for line in result.stdout.splitlines() if line.strip())

    # Prefer a jar explicitly named for OpenChemLib if more than one valid jar exists.
    candidates = sorted(set(candidates), key=lambda p: ('openchemlib' not in p.name.lower(), str(p)))
    for candidate in candidates:
        if candidate.is_file() and contains_required_classes(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        'No installed jar containing the required OpenChemLib conformer classes was found. '
        'The usual macOS target is /Applications/DataWarrior.app/Contents/Java/update/datawarrior_all.jar. '
        'In Terminal, run: find /Applications/DataWarrior.app ~/Library -type f -name \"*.jar\" 2>/dev/null '
        'Then set OPEN_CHEM_LIB_JAR to the matching DataWarrior/OpenChemLib jar path.'
    )

def conformer_seed(smiles, base_seed=BASE_SEED):
    digest = hashlib.sha256(f'{base_seed}\0{smiles}'.encode()).digest()[:8]
    seed = int.from_bytes(digest, 'big', signed=True)
    return seed if seed != 0 else 1

def b64(value):
    return base64.b64encode(str(value).encode('utf-8')).decode('ascii')

def find_java_runtime(ocl_jar, explicit_path=None):
    # Prefer the Java runtime bundled with the selected DataWarrior app.
    # /usr/bin/java on macOS may be only a placeholder that prints
    # 'Unable to locate a Java Runtime' and exits.
    ocl_jar = Path(ocl_jar).resolve()
    candidates = []
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file(): raise FileNotFoundError(f'--java not found: {candidate}')
        candidates.append(candidate)
    else:
        # The executed notebook uses Java under the active Python environment.
        environment_java = Path(sys.prefix) / 'bin' / 'java'
        if environment_java.is_file(): candidates.append(environment_java)
        else:
            candidates.extend(p for p in Path(sys.prefix).rglob('java')
                              if p.is_file() and p.parent.name == 'bin')
    app_bundle = next((parent for parent in ocl_jar.parents if parent.suffix == '.app'), None)
    if app_bundle is not None:
        contents = app_bundle / 'Contents'
        candidates.extend(
            path for path in contents.rglob('java')
            if path.is_file() and path.parent.name == 'bin'
        )

    system_java = shutil.which('java')
    if system_java:
        candidates.append(Path(system_java))

    diagnostics = []
    for candidate in dict.fromkeys(candidates):
        if not os.access(candidate, os.X_OK):
            continue
        version = subprocess.run(
            [str(candidate), '-version'], capture_output=True, text=True, check=False
        )
        if version.returncode != 0:
            diagnostics.append(f'{candidate}: {version.stderr.strip()}')
            continue
        modules = subprocess.run(
            [str(candidate), '--list-modules'], capture_output=True, text=True, check=False
        )
        if modules.returncode == 0 and 'jdk.compiler@' in modules.stdout:
            return candidate.resolve()
        diagnostics.append(f'{candidate}: Java found, but the jdk.compiler module is absent')

    detail = '\n'.join(diagnostics) if diagnostics else 'No Java executable was found.'
    raise RuntimeError(
        'A working Java runtime with the jdk.compiler module is required. '
        'Use --java to select the environment or DataWarrior Java executable.\n' + detail
    )

def pack_sdf_properties(row):
    """Encode every original CSV column/value for the Java helper."""
    return "\u001e".join(
        f"{column}\u001f{row[column]}"
        for column in row.index
    )

JAVA_SOURCE = r'''
import com.actelion.research.chem.MolfileV3Creator;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.SmilesParser;
import com.actelion.research.chem.forcefield.mmff.ForceFieldMMFF94;
import org.openmolecules.chem.conf.gen.ConformerGenerator;
import org.openmolecules.chem.conf.gen.RigidFragmentCache;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class SeededDataWarriorConformers {
  static final int MAX_TORSION_SETS = 1000;
  static final int MAX_ITERATIONS = 10000;
  static final double GRADIENT_TOLERANCE = 0.0001;
  static final double FUNCTION_TOLERANCE = 1.0e-6;

  public static void main(String[] args) throws Exception {
    Path manifest = Path.of(args[0]);
    Path output = Path.of(args[1]);
    Path failures = Path.of(args[2]);
    double dielectric = Double.parseDouble(args[3]);
    RigidFragmentCache.getDefaultInstance().loadDefaultCache();
    ForceFieldMMFF94.initialize(ForceFieldMMFF94.MMFF94SPLUS);
    Map<String,Object> options = new HashMap<>();
    options.put("dielectric constant", dielectric);
    int inputCount = 0, writtenCount = 0, failureCount = 0;

    try (BufferedReader in = Files.newBufferedReader(manifest, StandardCharsets.UTF_8);
         BufferedWriter sdf = Files.newBufferedWriter(output, StandardCharsets.UTF_8);
         BufferedWriter failed = Files.newBufferedWriter(failures, StandardCharsets.UTF_8)) {
      failed.write("dataframe_row\tinput_id\tsmiles\terror\n");
      String line;
      while ((line = in.readLine()) != null) {
        inputCount++;
        String[] f = line.split("\\t", -1);
        String row = decode(f[0]), id = decode(f[1]), smiles = decode(f[2]);
        long seed = Long.parseLong(f[3]);
        String originalProperties = decode(f[4]);
        try {
          StereoMolecule mol = new StereoMolecule();
          new SmilesParser().parse(mol, smiles);
          ConformerGenerator cg = new ConformerGenerator(seed, true);
          if (!cg.initializeConformers(mol, ConformerGenerator.STRATEGY_LIKELY_SYSTEMATIC,
                  MAX_TORSION_SETS, false))
            throw new Exception("Conformer generator could not initialize");
          mol = cg.getNextConformerAsMolecule(mol);
          if (mol == null) throw new Exception("No conformer generated");
          double energy = Double.NaN;
          String minimizationError = "";

          try {
            energy = minimize(mol, options);
          } catch (Exception e) {
            // Match DataWarrior: retain the generated conformer when
            // MMFF94 typing or minimization fails.
            minimizationError = e.getClass().getSimpleName() + ": " + e.getMessage();
          }

          center(mol);
          mol.setName(id.isBlank() ? "row " + row : id);
          sdf.write(new MolfileV3Creator(mol).getMolfile());
          // Restore every original CSV column using its exact original name.
          for (String entry : originalProperties.split("\u001e", -1)) {
            if (entry.isEmpty()) continue;

            int separator = entry.indexOf('\u001f');
            if (separator < 0) continue;

            String columnName = entry.substring(0, separator);
            String columnValue = entry.substring(separator + 1);

            property(sdf, columnName, columnValue);
          }
          property(sdf, "DATAFRAME_ROW", row);
          property(sdf, "INPUT_ID", id);
          property(sdf, "INPUT_SMILES", smiles);
          property(sdf, "CONFORMER_SEED", Long.toString(seed));
          property(sdf, "CONFORMER_ALGORITHM", "systematic, low-energy biased");
          property(sdf, "TORSION_SOURCE", "crystallographic");
          property(sdf, "FORCE_FIELD", "MMFF94s+");
          property(sdf, "DIELECTRIC_CONSTANT", Double.toString(dielectric));
          property(
              sdf,
              "MMFF94_ENERGY",
              Double.isNaN(energy) ? "" : String.format(Locale.ROOT, "%.9g", energy)
          );
          property(sdf, "MINIMIZATION_ERROR", minimizationError);
          sdf.write("$$$$\n");
          writtenCount++;
        } catch (Throwable e) {
          failureCount++;
          failed.write(clean(row)+"\t"+clean(id)+"\t"+clean(smiles)+"\t"
              +clean(e.getClass().getSimpleName()+": "+e.getMessage())+"\n");
        }
      }
    }
    System.out.printf(Locale.ROOT, "Input: %d  Written: %d  Failed: %d%n",
        inputCount, writtenCount, failureCount);
    if (failureCount != 0) System.exit(1);
  }

  static double minimize(StereoMolecule mol, Map<String,Object> options) throws Exception {
    int[] fragmentNo = new int[mol.getAllAtoms()];
    int fragmentCount = mol.getFragmentNumbers(fragmentNo, false, true);
    double energy = Double.NaN;
    if (fragmentCount == 1) {
      ForceFieldMMFF94 ff = new ForceFieldMMFF94(mol, ForceFieldMMFF94.MMFF94SPLUS, options);
      int error = ff.minimise(MAX_ITERATIONS, GRADIENT_TOLERANCE, FUNCTION_TOLERANCE);
      if (error != 0) throw new Exception("MMFF94 error code " + error);
      energy = ff.getTotalEnergy();
    } else {
      StereoMolecule[] fragments = mol.getFragments(fragmentNo, fragmentCount);
      int maxAtoms = 0;
      for (StereoMolecule fragment : fragments) {
        if (fragment.getAllAtoms() > 2) {
          ForceFieldMMFF94 ff = new ForceFieldMMFF94(fragment, ForceFieldMMFF94.MMFF94SPLUS, options);
          int error = ff.minimise(MAX_ITERATIONS, GRADIENT_TOLERANCE, FUNCTION_TOLERANCE);
          if (error != 0) throw new Exception("MMFF94 error code " + error);
          if (maxAtoms < fragment.getAllAtoms()) {
            maxAtoms = fragment.getAllAtoms(); energy = ff.getTotalEnergy();
          }
        }
      }
      int[] atomInFragment = new int[fragmentCount];
      for (int atom=0; atom<fragmentNo.length; atom++) {
        int fragment = fragmentNo[atom], sourceAtom = atomInFragment[fragment]++;
        mol.setAtomX(atom, fragments[fragment].getAtomX(sourceAtom));
        mol.setAtomY(atom, fragments[fragment].getAtomY(sourceAtom));
        mol.setAtomZ(atom, fragments[fragment].getAtomZ(sourceAtom));
      }
    }
    return energy;
  }

  static void center(StereoMolecule mol) {
    double x=0, y=0, z=0;
    for (int a=0; a<mol.getAllAtoms(); a++) { x+=mol.getAtomX(a); y+=mol.getAtomY(a); z+=mol.getAtomZ(a); }
    x/=mol.getAllAtoms(); y/=mol.getAllAtoms(); z/=mol.getAllAtoms();
    for (int a=0; a<mol.getAllAtoms(); a++) {
      mol.setAtomX(a,mol.getAtomX(a)-x); mol.setAtomY(a,mol.getAtomY(a)-y); mol.setAtomZ(a,mol.getAtomZ(a)-z);
    }
  }

  static void property(BufferedWriter w, String name, String value) throws Exception {
    w.write(">  <"+name+">\n"+clean(value)+"\n\n");
  }
  static String decode(String s) { return new String(Base64.getDecoder().decode(s), StandardCharsets.UTF_8); }
  static String clean(String s) { return s == null ? "" : s.replace('\t',' ').replace('\r',' ').replace('\n',' '); }
}
'''

def generate_openchemlib_conformers(data, args, out):
    """Notebook 01: one seeded OpenChemLib conformer, MMFF94s+, V3000 SDF."""
    from rdkit.Chem.PandasTools import LoadSDF
    jar = find_openchemlib_jar(args.openchemlib_jar)
    java = find_java_runtime(jar, args.java)
    sdf = (out / 'kpuu_conformers.sdf').resolve()
    failures = out / 'conformer_failures.tsv'
    java_file = (out / 'SeededDataWarriorConformers.java').resolve()
    manifest = (out / 'conformer_manifest.tsv').resolve()
    java_file.write_text(JAVA_SOURCE, encoding='utf-8')
    # Match notebook 01's CSV read as strings, original SMILES, and seed derivation.
    data.to_csv(out / 'conformer_input.csv', index=False)
    source = pd.read_csv(out / 'conformer_input.csv', dtype=str, keep_default_na=False)
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        for dataframe_row, row in source.iterrows():
            smiles = row['smiles'].strip()
            seed = conformer_seed(smiles, args.conformer_seed)
            handle.write(f'{b64(dataframe_row)}\t{b64(row.sample_id)}\t{b64(smiles)}\t'
                         f'{seed}\t{b64(pack_sdf_properties(row))}\n')
    command = [str(java), '--class-path', str(jar), str(java_file), str(manifest),
               str(sdf), str(failures.resolve()), str(args.dielectric_constant)]
    print(f'Generating notebook-01 OpenChemLib conformers with {jar}', flush=True)
    print(f'Conformer base seed: {args.conformer_seed}; MMFF94s+ dielectric: {args.dielectric_constant}', flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (out / 'conformer_generation.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.stdout.strip(): print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f'OpenChemLib conformer generation failed: {result.stderr.strip()}. '
                           f'See {failures} and {out / "conformer_generation.log"}. No RDKit fallback was used.')
    with sdf.open(encoding='utf-8') as handle:
        count = sum(line.rstrip() == '$$$$' for line in handle)
    if count != len(data):
        raise ValueError(f'Expected {len(data)} SDF records but found {count}; no molecules may be silently dropped.')
    loaded = LoadSDF(str(sdf), smilesName='SMILES')
    if len(loaded) != len(data):
        raise ValueError('RDKit could not load every generated SDF record; inspect conformer_generation.log.')
    for column, expected in [('INPUT_ID', source.sample_id), ('INPUT_SMILES', source.smiles.str.strip())]:
        if column not in loaded or loaded[column].astype(str).tolist() != expected.tolist():
            raise ValueError(f'SDF {column} does not match the input rows in order.')
    audit_columns = ['INPUT_ID', 'INPUT_SMILES', 'CONFORMER_SEED', 'CONFORMER_ALGORITHM',
                     'TORSION_SOURCE', 'FORCE_FIELD', 'DIELECTRIC_CONSTANT',
                     'MMFF94_ENERGY', 'MINIMIZATION_ERROR']
    audit = loaded.reindex(columns=audit_columns).rename(columns={'INPUT_ID': 'sample_id', 'INPUT_SMILES': 'smiles'})
    audit.to_csv(out / 'conformer_audit.csv', index=False)
    failed_minimization = audit.MINIMIZATION_ERROR.fillna('').astype(str).str.strip().ne('').sum()
    if failed_minimization:
        print(f'{failed_minimization} conformers retained after minimization errors, matching notebook 01; '
              'see conformer_audit.csv.', flush=True)
    version = subprocess.run([str(java), '-version'], capture_output=True, text=True, check=True)
    metadata = dict(conformer_method='OpenChemLib / notebook 01', openchemlib_jar=str(jar),
                    openchemlib_jar_sha256=sha256(jar), java=str(java),
                    java_version=(version.stderr or version.stdout).strip(),
                    java_source_sha256=sha256(java_file), conformer_seed=args.conformer_seed,
                    seed_derivation='signed 64-bit SHA256 of base_seed + NUL + stripped original SMILES; zero replaced by one',
                    conformer_selection='first conformer from STRATEGY_LIKELY_SYSTEMATIC',
                    torsion_source='crystallographic', max_torsion_sets=1000,
                    forcefield='MMFF94s+', dielectric_constant=args.dielectric_constant,
                    max_optimization_iterations=10000, gradient_tolerance=0.0001,
                    function_tolerance=1e-6,
                    minimization_failure_policy='retain generated conformer, as in notebook 01',
                    n_minimization_errors=int(failed_minimization), sdf_sha256=sha256(sdf))
    dump_json(out / 'conformer_generation.json', metadata)
    return sdf, metadata


def generate_features(data, required_names, args, out, setup):
    """Use the notebook's SDF -> Mordred CLI and original-SMILES fingerprints."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdMolDescriptors
    calc, ecfp, maccs = setup
    # Always use the notebook's complete SDF/Mordred route, even if a saved fold
    # selected only 2D features. Select saved model columns only after calculation.
    sdf, metadata = generate_openchemlib_conformers(data, args, out)
    mordred_file = out / 'kpuu_3d_mordred_labels.csv'
    command = [sys.executable, '-m', 'mordred', '-3', str(sdf),
               '-o', str(mordred_file), '-p', str(args.threads)]
    print('Calculating Mordred descriptors from the OpenChemLib SDF (notebook 01 procedure)...', flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (out / 'mordred.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f'Mordred failed: {result.stderr.strip()}. See {out / "mordred.log"}.')
    mordred = pd.read_csv(mordred_file, low_memory=False)
    if len(mordred) != len(data) or 'name' not in mordred or mordred['name'].astype(str).tolist() != data.sample_id.astype(str).tolist():
        raise ValueError('Mordred output names/count/order do not match the eligible compounds.')
    descriptor_names = [str(d) for d in calc.descriptors]
    frame = numeric_frame(mordred, descriptor_names, context='Notebook-01 Mordred output')
    issues = []
    for name in descriptor_names:
        for i in np.flatnonzero(frame[name].isna().to_numpy()):
            issues.append(dict(sample_id=data.iloc[i].sample_id, descriptor=name, reason=str(mordred.iloc[i][name])))
    fingerprints = []
    for record in data.itertuples(index=False):
        # Notebook 01 calculates both fingerprints from INPUT_SMILES, not the
        # SDF molecule, so hydrogen handling matches the training features.
        mol = Chem.MolFromSmiles(str(record.smiles).strip())
        if mol is None: raise ValueError(f'Invalid fingerprint input: {record.sample_id}.')
        values = {}
        if ecfp:
            bits = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
            values.update({name: int(bits[index]) for name, index in ecfp})
        if maccs:
            bits = MACCSkeys.GenMACCSKeys(mol)
            values.update({name: int(bits[index]) for name, index in maccs})
        fingerprints.append(values)
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(fingerprints)], axis=1).loc[:, required_names]
    pd.DataFrame(issues, columns=['sample_id', 'descriptor', 'reason']).to_csv(out / 'descriptor_errors.csv', index=False)
    pd.DataFrame({'descriptor': required_names, 'missing_count': frame.isna().sum().to_numpy()}).to_csv(
        out / 'descriptor_missingness.csv', index=False)
    entirely_missing = frame.columns[frame.isna().all()].tolist()
    if entirely_missing:
        print(f'{len(entirely_missing)} descriptors are undefined for every eligible compound; '
              'retaining NaN for the saved preprocessing. See descriptor_missingness.csv.', flush=True)
    external = pd.concat([data.reset_index(drop=True).drop(columns=required_names, errors='ignore'), frame], axis=1)
    external.to_csv(out / 'kpuu_with_descriptors.csv', index=False)
    metadata.update(method='automatic', feature_names=required_names, versions=versions(),
                    reference_workflow='01_clean_label_generate_b3db_outerfolds: seeded OpenChemLib + Mordred -3',
                    mordred_command=command, fingerprint_input='original SMILES (INPUT_SMILES in notebook 01)',
                    ecfp=dict(radius=2, n_bits=2048, include_chirality=False), maccs_n_bits=167,
                    undefined_values='NaN; handled by original saved preprocessing or native LightGBM',
                    entirely_undefined_descriptors=entirely_missing)
    dump_json(out / 'feature_generation.json', metadata)
    return external, metadata


def inference(data, args, out, models, records, setup):
    feature_metadata = {'method': 'precomputed_predictions'}
    if args.predictions_csv:
        raw = pd.read_csv(args.predictions_csv, low_memory=False)
        required = {'model', args.prediction_col, args.prediction_smiles_col}
        if not required.issubset(raw): raise ValueError(f'Prediction CSV requires {sorted(required)}.')
        raw = raw[raw.model.isin(models)].copy()
        raw['key'] = raw[args.prediction_smiles_col].map(structure_key)
        raw = raw[raw.key.isin(data.key)]
        raw['prediction'] = pd.to_numeric(raw[args.prediction_col], errors='raise')
        if raw.duplicated(['model', 'key']).any():
            raise ValueError('Prediction CSV must contain one ENSEMBLE prediction per model and structure, not fold or replicate rows.')
        raw['decision_threshold'] = raw.get('decision_threshold', raw.get('threshold', np.nan))
        for model in models:
            mask = raw.model.eq(model)
            if model in TUNED_THRESHOLD_MODELS:
                thresholds = pd.to_numeric(raw.loc[mask, 'decision_threshold'], errors='coerce')
                if len(thresholds) and (not np.isfinite(thresholds).all() or thresholds.nunique() != 1):
                    raise ValueError(f'{model}: precomputed ensemble predictions require one finite '
                                     'decision_threshold (or threshold) per model. Rerun inference with '
                                     '--features-csv and --model-dir to load the saved fold thresholds.')
                raw.loc[mask, 'decision_threshold'] = thresholds
            else:
                raw.loc[mask, 'decision_threshold'] = -1.
        raw['threshold_variant'] = np.where(raw.model.isin(TUNED_THRESHOLD_MODELS), 'inner_tuned', 'fixed')
        predictions = raw[['model', 'key', 'prediction', 'decision_threshold', 'threshold_variant']].copy()
    else:
        required_names = list(dict.fromkeys(n for r in records for n in r['required_names']))
        if args.features_csv:
            feat = pd.read_csv(args.features_csv, low_memory=False)
            if args.features_smiles_col not in feat: raise ValueError('Feature CSV SMILES column absent.')
            feat['key'] = feat[args.features_smiles_col].map(structure_key)
            feat = feat[feat.key.isin(data.key)]
            if feat.key.duplicated().any(): raise ValueError('Feature table must have one row per eligible standardized structure.')
            if set(data.key) - set(feat.key): raise ValueError('Some eligible compounds are missing from the feature table.')
            numeric = numeric_frame(feat, required_names, context='Supplied feature table')
            numeric['key'] = feat.key.to_numpy()
            external = data.drop(columns=required_names, errors='ignore').merge(numeric, on='key', validate='one_to_one', sort=False)
            feature_metadata = dict(method='supplied_features', file=str(args.features_csv), sha256=sha256(args.features_csv))
            dump_json(out / 'feature_generation.json', feature_metadata)
        else:
            external, feature_metadata = generate_features(data, required_names, args, out, setup)
        per_fold = []
        for record in records:
            context = f'{record["model"]}, fold {record["fold"]}'
            estimator = record['estimator']
            if record['native']:
                X, _ = transform_custom_external(record['processor'], external, context=context)
                if X.shape[1] != estimator.num_feature(): raise ValueError(f'{context}: saved preprocessing/model feature-count mismatch.')
                pred = np.asarray(estimator.predict(X, num_threads=args.threads), float).reshape(-1)
            else:
                X = numeric_frame(external, record['feature_names'], context=context)
                pred = np.asarray(estimator.predict(X), float).reshape(-1)
            if len(pred) != len(external) or not np.isfinite(pred).all(): raise ValueError(f'{context}: invalid predictions.')
            per_fold.append(pd.DataFrame(dict(key=external.key, model=record['model'],
                                             outer_fold_model=record['fold'], prediction=pred,
                                             decision_threshold=record['decision_threshold'],
                                             threshold_variant=record['threshold_variant'],
                                             threshold_source=str(record.get('threshold_source', 'fixed -1')),
                                             model_source=str(record['path']))))
        folds = pd.concat(per_fold, ignore_index=True)
        folds.to_csv(out / 'fold_predictions.csv', index=False)
        # Match script 10: mean fold decision margin equals mean prediction
        # minus mean threshold. This is not majority voting over fold labels.
        predictions = folds.groupby(['model', 'key'], as_index=False, sort=False).agg(
            prediction=('prediction', 'mean'), decision_threshold=('decision_threshold', 'mean'),
            threshold_variant=('threshold_variant', 'first'))
    if not np.isfinite(predictions.prediction).all(): raise ValueError('Nonfinite model predictions.')
    for model in models:
        keys = set(predictions.loc[predictions.model == model, 'key'])
        if keys != set(data.key): raise ValueError(f'{model}: missing predictions; all models must cover the same eligible compounds.')
    result = predictions.merge(data[['key', 'sample_id', 'smiles', 'Kpuu', 'log10_Kpuu', 'species']],
                               on='key', validate='many_to_one')
    result.to_csv(out / 'transfer_predictions.csv', index=False)
    return result, feature_metadata



def transfer_metrics(y, pred, cutoff, logbb_threshold=-1.):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
        precision_recall_curve, auc, cohen_kappa_score)
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    variable = len(y) > 2 and np.ptp(y) > 0 and np.ptp(pred) > 0
    values = {'spearman_r': float(spearmanr(y, pred).statistic) if variable else np.nan,
              'pearson_r': float(pearsonr(y, pred).statistic) if variable else np.nan,
              'kendall_tau_b': float(kendalltau(y, pred, variant='b').statistic) if variable else np.nan}
    if cutoff is not None:
        low, predicted_low = y < np.log10(cutoff), pred < logbb_threshold
        tn, fp, fn, tp = classification_counts(low, predicted_low)
        both = (tp + fn > 0) and (tn + fp > 0)
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        # Match sklearn zero_division=0 for undefined precision and F1.
        precision = tp / (tp + fp) if tp + fp else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
        mcc = (tp * tn - fp * fn) / denominator if denominator else (0. if both else np.nan)
        if both:
            pr_precision, pr_recall, _ = precision_recall_curve(low, -pred)
            pr_auc = float(auc(pr_recall, pr_precision))
        else:
            pr_auc = np.nan
        values.update(roc_auc_low_Kpuu=float(roc_auc_score(low, -pred)) if both else np.nan,
            average_precision_low_Kpuu=float(average_precision_score(low, -pred)) if both else np.nan,
            pr_auc_trapezoidal_low_Kpuu=pr_auc,
            accuracy_BBB_rule=(tp + tn) / len(low),
            balanced_accuracy_BBB_rule=(sensitivity + specificity) / 2 if both else np.nan,
            precision_low_Kpuu_BBB_rule=precision,
            recall_low_Kpuu_BBB_rule=sensitivity,
            f1_low_Kpuu_BBB_rule=f1,
            specificity_high_Kpuu_BBB_rule=specificity,
            mcc_BBB_rule=mcc,
            cohens_kappa_BBB_rule=float(cohen_kappa_score(low, predicted_low)) if both else np.nan)
    return values


def classification_counts(low, predicted_low):
    low, predicted_low = np.asarray(low, bool), np.asarray(predicted_low, bool)
    return tuple(int(np.sum(mask)) for mask in
                 (~low & ~predicted_low, ~low & predicted_low,
                  low & ~predicted_low, low & predicted_low))


def evaluate(predictions, models, args, out):
    wide = predictions.pivot(index='key', columns='model', values='prediction').sort_index()
    truth = predictions.drop_duplicates('key').set_index('key').loc[wide.index, 'log10_Kpuu'].to_numpy(float)
    thresholds = {}
    for model in models:
        values = pd.to_numeric(predictions.loc[predictions.model.eq(model), 'decision_threshold'], errors='raise')
        if not np.isfinite(values).all() or values.nunique() != 1:
            raise ValueError(f'{model}: expected a single finite ensemble decision threshold.')
        thresholds[model] = float(values.iloc[0])
    if args.kpuu_cutoff is not None:
        counts = []
        for model in models:
            tn, fp, fn, tp = classification_counts(truth < np.log10(args.kpuu_cutoff), wide[model].to_numpy(float) < thresholds[model])
            counts.append(dict(model=model, positive_class='low_Kpuu', kpuu_cutoff=args.kpuu_cutoff,
                predicted_logBB_cutoff=thresholds[model], tn=tn, fp=fp, fn=fn, tp=tp,
                n_low_Kpuu=tp + fn, n_high_Kpuu=tn + fp, n_predicted_low_Kpuu=tp + fp))
        pd.DataFrame(counts).to_csv(out / 'classification_counts.csv', index=False)
        predictions = predictions.assign(true_low_Kpuu=predictions.log10_Kpuu.lt(np.log10(args.kpuu_cutoff)).astype(int),
                                         predicted_low_Kpuu_BBB_rule=predictions.prediction.lt(predictions.decision_threshold).astype(int))
        predictions.to_csv(out / 'transfer_predictions.csv', index=False)
    else:
        print('Reporting correlations. Add --kpuu-cutoff VALUE to also evaluate classification.', flush=True)
    point = {m: transfer_metrics(truth, wide[m].to_numpy(float), args.kpuu_cutoff, thresholds[m]) for m in models}
    rng = np.random.default_rng(args.seed)
    samples = {m: {k: [] for k in point[m]} for m in models}
    for _ in range(args.bootstrap):
        ix = rng.integers(0, len(truth), len(truth))
        for model in models:
            v = transfer_metrics(truth[ix], wide[model].to_numpy(float)[ix], args.kpuu_cutoff, thresholds[model])
            for metric, score in v.items(): samples[model][metric].append(score)
    rows, comparisons = [], []
    def interval(values):
        values = np.asarray(values, float); values = values[np.isfinite(values)]
        return (*np.quantile(values, [.025,.975]), len(values)) if len(values) else (np.nan,np.nan,0)
    for model in models:
        for metric, value in point[model].items():
            lo, hi, n = interval(samples[model][metric])
            rows.append(dict(model=model, metric=metric, estimate=value, ci_low=lo, ci_high=hi,
                             n_compounds=len(truth), n_valid_bootstraps=n,
                             kpuu_cutoff=args.kpuu_cutoff, predicted_logBB_cutoff=thresholds[model] if args.kpuu_cutoff is not None else None))
    for i, reference in enumerate(models):
        for model in models[i + 1:]:
            for metric in point[model]:
                delta = np.asarray(samples[model][metric])-np.asarray(samples[reference][metric])
                lo, hi, n = interval(delta)
                comparisons.append(dict(model=model, reference=reference, metric=metric,
                    delta=point[model][metric]-point[reference][metric], ci_low=lo, ci_high=hi, n_valid_bootstraps=n))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'transfer_metrics.csv', index=False)
    if comparisons: pd.DataFrame(comparisons).to_csv(out / 'paired_differences.csv', index=False)
    print(result.to_string(index=False))


def write_methods(args, out, data, feature_metadata):
    uses_tuned = bool({m.strip() for m in args.models.split(',')} & TUNED_THRESHOLD_MODELS)
    classification_rule = (
        'Native models used the fixed predicted logBB < -1 rule. For the threshold-tuned MSE '
        'comparator, the classification thresholds previously selected on BBB inner folds were '
        'retained. Ensemble classification compared mean fold prediction with mean fold threshold, '
        'equivalent to averaging fold decision margins and matching the external BBB evaluation. '
        if uses_tuned else 'Classification used the fixed predicted logBB < -1 rule. ')
    if feature_metadata['method'] == 'automatic':
        representation = ('Required descriptors were computed by name using Mordred, radius-2 2048-bit '
                          'Morgan fingerprints without chirality, and 167-bit MACCS keys, retaining only '
                          'features needed by the saved models and their preprocessing. ')
        if feature_metadata.get('conformer_method'):
            representation += (f'Three-dimensional coordinates were generated with the same seeded '
                f'OpenChemLib procedure as notebook 01, using the DataWarrior/OpenChemLib JAR recorded '
                f'in the run metadata. The first conformer from low-energy-biased systematic generation '
                f'with crystallographic torsions (maximum 1,000 torsion sets) was minimized with '
                f'MMFF94s+ (dielectric constant {args.dielectric_constant}; maximum 10,000 iterations; '
                f'gradient tolerance 0.0001; function tolerance 1e-6). Seeds were derived from '
                f'base seed {args.conformer_seed} and the original SMILES using the notebook SHA256 scheme. '
                f'As in notebook 01, conformers were retained if minimization failed, with failures '
                f'recorded, centered, and exported as V3000 SDF. Mordred descriptors were calculated '
                f'using the same python -m mordred -3 command; ECFP4 and MACCS fingerprints were '
                f'calculated from the original input SMILES. ')
    elif feature_metadata['method'] == 'supplied_features':
        representation = ('A supplied descriptor table was aligned to saved training feature names; '
                          'its coordinate preparation should be documented separately. ')
    else:
        representation = 'Precomputed ensemble logBB predictions were supplied. '
    text = (f'Cross-endpoint transfer was evaluated in {len(data)} independent structures with exact '
        f'positive {args.species} Kp,uu measurements, with repeated measurements summarized by the median '
        f'on the log10 scale. Compounds matching any member of the full supplied BBB development dataset '
        f'were excluded using RDKit parent-fragment standardization, uncharging, removal of stereochemistry, '
        f'and the HetAtomTautomerv2 hash for tautomer-insensitive matching. ' + representation +
        f'Existing BBB regressors were applied with their saved preprocessing, without retraining, '
        f'recalibration, or Kp,uu-based model or threshold selection. Saved outer-fold predictions were '
        f'averaged to give one prediction per model per compound. Transfer was assessed primarily by '
        f'Spearman correlation between predicted logBB and log10(Kp,uu), with Pearson correlation '
        f'and Kendall tau-b reported as additional association measures. Paired resampling of compounds '
        f'({args.bootstrap} bootstrap samples) provided percentile 95% confidence intervals for metrics '
        f'and all pairwise between-model differences, conditional on the fitted models and saved thresholds. Numerical logBB errors were '
        f'not calculated against Kp,uu because these represent different endpoints. ')
    if args.kpuu_cutoff is not None:
        text += (f'Secondary classification defined low unbound exposure as Kp,uu < {args.kpuu_cutoff}; '
                 f'ROC AUC, average precision, and trapezoidal precision-recall AUC used negative '
                 f'predicted logBB as the score. Accuracy, balanced accuracy, low-exposure precision, '
                 f'recall and F1, high-exposure specificity, MCC, Cohen kappa, and confusion counts '
                 f'assessed transfer of the saved BBB classification rules. ' + classification_rule +
                 f'The logBB and Kp,uu cutoffs were not treated as equivalent, and no thresholds were '
                 f'selected or adjusted using Kp,uu measurements. Undefined precision and F1 were set to zero; '
                 f'constant-prediction MCC was zero when both observed classes were present. '
                 f'Metrics requiring an absent observed class were undefined, and confidence intervals '
                 f'used finite bootstrap estimates only. ')
    (out / 'methods.txt').write_text(text.strip() + '\n', encoding='utf-8')


def main():

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--development-csv', required=True, type=Path,
        help='FULL BBB model-development dataset, including all outer folds and class-only compounds.')
    p.add_argument('--development-smiles-col', default='smiles')
    p.add_argument('--fold-col', default='outer_fold')
    p.add_argument('--input', type=Path, help='Optional local Kp,uu CSV; omit to download the Lawrenz supplement.')
    p.add_argument('--smiles-col', default='smiles')
    p.add_argument('--kpuu-col', default='Kpuu', help='Linear unbound brain/plasma ratio, NOT logarithm or total Kp.')
    p.add_argument('--species-col', default='species')
    p.add_argument('--relation-col', help='Optional column recording =, <, >, etc.')
    p.add_argument('--species', default='rat', help='Analyze one species; any requires a single-species input.')
    p.add_argument('--cache-dir', type=Path, default=Path('kpuu_transfer/kpuu_downloads'))
    p.add_argument('--output-dir', type=Path, default=Path('kpuu_transfer'))
    p.add_argument('--prepare-only', action='store_true', help='Optional curation-only mode; not required before a full run.')
    p.add_argument('--model-dir', type=Path, default=Path('random_outerfolds/all_model_results_main/fitted_models'),
        help='Existing saved BBB outer-fold models; these are applied without refitting.')
    p.add_argument('--models', default='lightgbm_mse_regression,custom_mtl_bab,lightgbm_mse_regression_threshold')
    p.add_argument('--threshold-pattern', help='Optional saved threshold JSON glob; supports {fold} and {model}. Default: MODEL_outer_FOLD.json.')
    p.add_argument('--openchemlib-jar', type=Path, default=Path('../tools/datawarrior_all.jar'),
        help='The SAME DataWarrior/OpenChemLib JAR used by notebook 01.')
    p.add_argument('--java', type=Path, help='Optional Java executable; otherwise try the active environment, DataWarrior and PATH.')
    p.add_argument('--conformer-seed', type=int, default=BASE_SEED,
        help='Notebook 01 BASE_SEED; independent of --seed used for statistical resampling.')
    p.add_argument('--dielectric-constant', type=float, default=DIELECTRIC_CONSTANT,
        help='Notebook 01 DIELECTRIC_CONSTANT for OpenChemLib MMFF94s+.')
    p.add_argument('--threads', type=int, default=1, help='Mordred workers and native LightGBM inference threads.')
    p.add_argument('--features-csv', type=Path, help='Optional existing table from the notebook-01 pipeline; omit to regenerate features.')
    p.add_argument('--features-smiles-col', default='smiles')
    p.add_argument('--predictions-csv', type=Path, help='Optional existing ENSEMBLE predictions; omit to generate predictions automatically.')
    p.add_argument('--prediction-col', default='predicted_logbb')
    p.add_argument('--prediction-smiles-col', default='smiles')
    p.add_argument('--kpuu-cutoff', type=float, help='Optional PRESPECIFIED linear cutoff for low-exposure classification.')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.kpuu_cutoff is not None and (not np.isfinite(args.kpuu_cutoff) or args.kpuu_cutoff <= 0): p.error('Kp,uu cutoff must be positive.')
    if args.bootstrap < 0: p.error('Bootstrap count must be nonnegative.')
    if args.threads < 1: p.error('Thread count must be positive.')
    if not np.isfinite(args.dielectric_constant) or args.dielectric_constant <= 0: p.error('Dielectric constant must be positive.')
    if args.features_csv and args.predictions_csv: p.error('Choose at most one precomputed input; omit both for the complete automatic workflow.')
    for key in ('development_csv', 'input', 'features_csv', 'predictions_csv'):
        path = getattr(args, key)
        if path is not None and not path.is_file():
            tip = ' Omit --predictions-csv to generate predictions automatically.' if key == 'predictions_csv' else ''
            p.error(f'{key.replace("_", "-")}: file not found: {path.resolve()}.{tip}')
    models = [s.strip() for s in args.models.split(',') if s.strip()]
    if not models or len(models) != len(set(models)): p.error('Specify distinct model names.')
    from rdkit.Chem import rdMolHash
    if not hasattr(rdMolHash.HashFunction, 'HetAtomTautomerv2'):
        p.error('RDKit lacks HetAtomTautomerv2; update RDKit in this environment before overlap screening.')
    records, setup = [], None
    if not args.prepare_only and not args.predictions_csv:
        records = load_models(args, models)
        if not args.features_csv:
            required_names = list(dict.fromkeys(n for r in records for n in r['required_names']))
            setup = descriptor_setup(required_names)
            args.openchemlib_jar = find_openchemlib_jar(args.openchemlib_jar)
            args.java = find_java_runtime(args.openchemlib_jar, args.java)
    out = output_dir(args.output_dir)
    print(f'Results directory: {out.resolve()}', flush=True)
    data = get_kpuu(args, out)
    model_manifest = [{k: v for k, v in record.items() if k not in ('estimator', 'processor')}
                      for record in records]
    dump_json(out / 'run_manifest.json', dict(settings=vars(args), versions=versions(),
        development_sha256=sha256(args.development_csv), n_eligible=len(data), models=model_manifest,
        prediction_file_sha256=sha256(args.predictions_csv) if args.predictions_csv else None,
        overlap_policy='full development set; cleaned uncharged parent, no stereochemistry, HetAtomTautomerv2',
        model_refitting=False, threshold_fitting=False, primary_metric='Spearman correlation',
        uncertainty='paired molecule bootstrap, conditional on fitted models'))
    if args.prepare_only:
        print(f'Curation complete: {out / "kpuu_eligible.csv"}. Omit --prepare-only for the full automatic workflow.')
        return
    pred, feature_metadata = inference(data, args, out, models, records, setup)
    evaluate(pred, models, args, out)
    write_methods(args, out, data, feature_metadata)
    print(f'\nComplete. Saved descriptors/conformers when generated, overlap audit, predictions, '
          f'metrics, paired intervals, and methods.txt to {out.resolve()}')


if __name__ == '__main__':
    main()
