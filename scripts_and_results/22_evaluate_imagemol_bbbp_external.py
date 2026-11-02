#!/usr/bin/env python3
"""Evaluate the released ImageMol BBBP classifier on every row of an external CSV.

Dependencies: torch, torchvision, rdkit, pandas, numpy, scikit-learn, Pillow.
Checkpoint: https://drive.google.com/file/d/1eaWZ9sAuqfQiGPMsfIpRV5zk3udCcNk_/view
Official source: https://github.com/HongxinXiang/ImageMol
Source reviewed at commit debc1c595e9262d923bcd97e5dee1cfe1a5a119d.

Uses the existing trained classifier, RGB 224x224 molecular images, CenterCrop,
ToTensor, and the official ImageNet normalization; no fitting or threshold tuning.
Structure preparation follows the salt/stereo removal linked in the ImageMol
README for new datasets. Its physicochemical exclusion filters are NOT applied:
every supplied compound is evaluated, or the run stops on an invalid structure.
Original input SMILES and the actual rendered SMILES are both exported.

BBBP positive class = PENETRANT. Outputs and metrics use NONPENETRANT = 1.
The official strict penetrant probability > 0.5 rule assigns ties to nonpenetrant.
The input defaults to bbb_class: 0=penetrant, 1=nonpenetrant; configure label
arguments explicitly if your input uses a different encoding. This classifier
does not predict logBB. RDKit versions can affect molecular image rendering.

For native-library crashes, add --diagnostics and run with python -X faulthandler
-u. The command-line entry point defaults to one CPU thread before native imports.

python 23_evaluate_imagemol_bbbp_external.py --input ../datasets/inxight_without_bbbp.csv --output-dir ./inxight_results/random_outerfolds/ --checkpoint ../tools/bbbp.pth

"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

DIAGNOSTICS = __name__ == "__main__" and "--diagnostics" in sys.argv


def progress(message):
    if DIAGNOSTICS:
        print(f"[ImageMol] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    progress(f"Python {sys.version.split()[0]} | {platform.system()} {platform.machine()} | {sys.executable}")

# Initialize PyTorch before importing other packages that may use OpenMP.
# This is a compatibility measure, not a guarantee against incompatible binaries.
progress("Importing torch")
import torch
if __name__ == "__main__":
    progress("Setting PyTorch intra/inter-op threads to 1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
progress("Importing torchvision")
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
progress("Importing numpy and pandas")
import numpy as np
import pandas as pd
progress("Importing Pillow and RDKit")
from PIL import __version__ as pillow_version
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from rdkit.Chem.SaltRemover import SaltRemover
progress("Importing scikit-learn metrics")
import sklearn
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
progress(f"Imports completed: torch={torch.__version__}, torchvision={torchvision.__version__}, numpy={np.__version__}, rdkit={rdBase.rdkitVersion}, sklearn={sklearn.__version__}")

MODEL_NAME = "imagemol_bbbp_released"
RELEASE_SHA256 = "2be5f0cd9c68b112a4ca3ff004e123ed8dab4bfc98eceb6dac6d6c8e4d09badc"
REMOVER = SaltRemover()
progress("RDKit salt remover initialized")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_input(path, smiles_col, label_col, id_col, penetrant_label, nonpenetrant_label):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(set(header)) != len(header):
        raise ValueError("The input has empty or duplicated column headers.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("The remaining dataset is empty; there are no compounds to evaluate.")
    for col in (smiles_col, label_col):
        if col not in df:
            raise ValueError(f"Missing column {col!r}; set --smiles-col/--label-col to the actual column names.")
        if df[col].isna().any() or df[col].str.strip().eq("").any():
            raise ValueError(f"Missing values in {col!r}; no rows will be discarded.")
    try:
        p_label, np_label = float(penetrant_label), float(nonpenetrant_label)
        values = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite([p_label, np_label]).all():
            raise ValueError("Nonfinite label encoding")
    except (TypeError, ValueError):
        p_label, np_label = penetrant_label.strip().casefold(), nonpenetrant_label.strip().casefold()
        values = df[label_col].str.strip().str.casefold().to_numpy()
    if p_label == np_label:
        raise ValueError("Penetrant and nonpenetrant labels must differ.")
    valid = (values == p_label) | (values == np_label)
    if not valid.all():
        raise ValueError(f"Unrecognized labels: {df.loc[~valid, label_col].unique().tolist()}; configure --penetrant-label/--nonpenetrant-label.")
    if id_col is None:
        id_col = next((c for c in ("external_compound_id", "sample_id", "row_index") if c in df), smiles_col)
    if id_col not in df:
        raise ValueError(f"Missing ID column {id_col!r}.")
    if df[id_col].isna().any() or df[id_col].str.strip().eq("").any() or df[id_col].duplicated().any():
        raise ValueError(f"{id_col!r} must contain unique, nonmissing IDs; specify --id-col. No deduplication is performed.")
    return df, (values == np_label).astype(int), id_col


def prepare_smiles(smiles):
    """Same structure conversion as linked CDDD remove_salt_stereo, without filters."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError(f"Invalid or unspecified molecular structure: {smiles!r}")
    converted = Chem.MolToSmiles(REMOVER.StripMol(mol, dontRemoveEverything=True), isomericSmiles=False)
    if "." in converted:
        fragments = Chem.GetMolFrags(Chem.MolFromSmiles(converted), asMols=True)
        converted = Chem.MolToSmiles(max(fragments, key=lambda m: m.GetNumAtoms()))
    if not converted:
        raise ValueError(f"Empty structure after salt removal: {smiles!r}")
    return converted


class MoleculeImages(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles
        self.transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        progress(f"Rendering image for input row {index}")
        mol = Chem.MolFromSmiles(self.smiles[index])
        if mol is None:
            raise ValueError(f"Invalid prepared SMILES at row {index}.")
        image = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224)).convert("RGB")
        return self.transform(image)


def load_classifier(path):
    progress("Checking checkpoint file hash")
    digest = sha256(path)
    if digest != RELEASE_SHA256:
        raise ValueError("This file is not the verified released BBBP checkpoint. Download bbbp.pth from the link in this script's docstring; its trained classification head is required.")
    progress("Loading checkpoint tensors on CPU")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older torch; file hash was verified above.
        state = torch.load(path, map_location="cpu")
    progress("Constructing ResNet-18")
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("Expected a complete released ResNet-18 state dictionary, including fc.weight and fc.bias.")
    progress("Validating and loading all model tensors")
    for name, target in model.state_dict().items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval(), digest


def predict(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits = []
    progress(f"Moving model to {device}; evaluating {len(dataset)} rows in batches of {batch_size}")
    model = model.to(device).eval()
    with torch.no_grad():
        for batch_index, images in enumerate(loader):
            progress(f"Forward pass for batch {batch_index + 1}/{len(loader)}")
            output = model(images.to(device))
            if output.shape != (images.shape[0], 1) or not torch.isfinite(output).all():
                raise ValueError("Expected one finite penetrant logit per molecule.")
            logits.append(output[:, 0].detach().cpu())
    if not logits:
        raise ValueError("No images to predict.")
    z = torch.cat(logits).float()
    if len(z) != len(dataset):
        raise RuntimeError("Prediction count does not match the input.")
    p_penetrant = torch.sigmoid(z)
    p_nonpenetrant = 1.0 - p_penetrant.double()
    pred_nonpenetrant = (p_penetrant <= 0.5).to(torch.int64)
    return z.numpy(), p_penetrant.numpy(), p_nonpenetrant.numpy(), pred_nonpenetrant.numpy()


def classification_metrics(y, probability_np, predicted_np):
    y, p, pred = np.asarray(y), np.asarray(probability_np), np.asarray(predicted_np)
    if y.ndim != 1 or len(y) == 0 or p.shape != y.shape or pred.shape != y.shape:
        raise ValueError("Nonempty, aligned label/prediction vectors are required.")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels, class predictions, or probabilities.")
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    n = len(y); n_p, n_np = tn + fp, tp + fn
    both = n_p > 0 and n_np > 0
    recall = tp / n_np if n_np else None
    specificity = tn / n_p if n_p else None
    accuracy = (tn + tp) / n
    expected = ((tp + fp) * n_np + (tn + fn) * n_p) / n ** 2
    mcc_denominator = np.sqrt(float((tp + fp) * n_np * n_p * (tn + fn)))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(np.mean(p[bins == b]) - np.mean(y[bins == b])) for b in range(10) if (bins == b).any())
    precision_curve, recall_curve, _ = precision_recall_curve(y, p) if both else (None, None, None)
    return {"n": n, "n_penetrant": n_p, "n_nonpenetrant": n_np,
        "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2 if both else None,
        "nonpenetrant_recall": recall, "nonpenetrant_precision": tp / (tp + fp) if tp + fp else 0.0,
        "nonpenetrant_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "penetrant_specificity": specificity, "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc_nonpenetrant": float(auc(recall_curve, precision_curve)) if both else None,
        "average_precision_nonpenetrant": float(average_precision_score(y, p)) if both else None,
        "mcc": float((tp * tn - fp * fn) / mcc_denominator) if mcc_denominator else 0.0,
        "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else None,
        "brier_score": float(np.mean((p - y) ** 2)), "ece_10_bins": float(ece),
        "wrong_side_count": fp + fn, "wrong_side_rate": (fp + fn) / n,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--label-col", default="bbb_class")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--penetrant-label", default="0")
    parser.add_argument("--nonpenetrant-label", default="1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--diagnostics", action="store_true", help="Print native import, molecule-rendering, and inference stages for crash diagnosis")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    paths = [args.output_dir / name for name in ("bbbp_predictions.csv", "bbbp_metrics.csv", "bbbp_evaluation_metadata.json")]
    if any(p.exists() or p.resolve() in (args.input.resolve(), args.checkpoint.resolve()) for p in paths):
        parser.error("An output already exists or overlaps an input. Choose a new --output-dir.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if args.device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        parser.error("MPS is unavailable; use --device cpu")
    progress("Reading and validating external CSV")
    df, y, id_col = read_input(args.input, args.smiles_col, args.label_col, args.id_col, args.penetrant_label, args.nonpenetrant_label)
    progress(f"Preparing structures for all {len(df)} rows")
    prepared = []
    for i, smiles in enumerate(df[args.smiles_col]):
        progress(f"Preparing input row {i}")
        try:
            prepared.append(prepare_smiles(smiles))
        except Exception as exc:
            raise ValueError(f"Input row {i}, ID {df.iloc[i][id_col]!r}: {exc}") from exc
    model, digest = load_classifier(args.checkpoint)
    z, p_p, p_np, pred = predict(model, MoleculeImages(prepared), args.device, args.batch_size)
    progress("Calculating classification metrics")
    results = classification_metrics(y, p_np, pred)
    predictions = pd.DataFrame({"external_compound_id": df[id_col].to_numpy(), "input_row_position": np.arange(len(df)),
        "SMILES": df[args.smiles_col].to_numpy(), "rendered_smiles": prepared, "label_original": df[args.label_col].to_numpy(),
        "bbb_class": y, "model": MODEL_NAME, "model_task": "classification", "score_type": "probability_np",
        "score_np": p_np, "pred_np": pred, "probability_np": p_np, "probability_penetrant": p_p,
        "logit_penetrant": z, "threshold": 0.5})
    for col in ("external_compound_id", "sample_id", "row_index"):
        if col in df and col not in predictions:
            predictions[col] = df[col].to_numpy()
    metadata = {"model": MODEL_NAME, "input_path": str(args.input.resolve()), "input_sha256": sha256(args.input),
        "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": digest, "n_evaluated": len(df),
        "id_source_column": id_col, "label_source_column": args.label_col,
        "input_penetrant_label": args.penetrant_label, "input_nonpenetrant_label": args.nonpenetrant_label,
        "output_label_encoding": {"0": "penetrant", "1": "nonpenetrant"}, "threshold": 0.5,
        "decision_rule": "penetrant if sigmoid(logit) > 0.5; otherwise nonpenetrant",
        "structure_preparation": "ImageMol README linked salt/stereo conversion; no physicochemical exclusions",
        "n_smiles_changed_for_rendering": int(sum(a != b for a, b in zip(df[args.smiles_col], prepared))),
        "image_preprocessing": "RDKit MolsToGridImage RGB 224x224; CenterCrop(224); ToTensor; ImageNet Normalize",
        "evaluation_scope": "all input rows; no train/test split, fitting, calibration fitting, or threshold tuning",
        "roc_auc_positive_class": "nonpenetrant", "pr_auc_method": "trapezoidal; average precision also reported separately",
        "ece_definition": "10 equal-width bins of P(nonpenetrant), sample-weighted absolute probability/frequency gap",
        "zero_denominator_convention": "precision/F1/MCC=0; unavailable class-dependent metrics=null",
        "device": args.device, "batch_size": args.batch_size, "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "versions": {"torch": str(torch.__version__), "torchvision": torchvision.__version__, "rdkit": rdBase.rdkitVersion,
                     "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "Pillow": pillow_version}}
    progress("Writing predictions, metrics, and metadata")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths[0], index=False)
    pd.DataFrame([{"model": MODEL_NAME, **results}]).to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Evaluated all {len(df)} compounds: {results['n_penetrant']} penetrant, {results['n_nonpenetrant']} nonpenetrant.")
    print(json.dumps(results, indent=2, allow_nan=False))
    print(f"Saved predictions, metrics, and evaluation metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
