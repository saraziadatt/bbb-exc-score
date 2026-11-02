#!/usr/bin/env python3
"""Frozen ImageMol logBB regression versus custom mixed-supervision MTL.

This script uses ImageMol's official ResNet-18 checkpoint-loading convention.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, __version__ as torchvision_version

from pretrained_bbb_common import load_bbb_dataframe, run_nested_outer_folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Official pretrained ImageMol checkpoint.")
    p.add_argument("--resume-key", default="state_dict")
    p.add_argument("--embedding-cache", type=Path, default=None)
    p.add_argument("--models", nargs="+", choices=["regression", "mtl"], default=["regression", "mtl"])
    p.add_argument("--repr-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256, help="Prediction batch size; linear heads use full-batch updates.")
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--margin-buffer", type=float, default=0.25)
    p.add_argument("--tune-loss", action="store_true", help="Tune MTL beta (0.5:0.25:5) and buffer (0.15:0.02:0.35) within inner CV.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true", help="Replace this trainer's existing result files and checkpoints.")
    return p.parse_args()


class MoleculeImageDataset(Dataset):
    def __init__(self, smiles: list[str]):
        self.smiles = smiles
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> torch.Tensor:
        mol = Chem.MolFromSmiles(self.smiles[index])
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES at row {index}: {self.smiles[index]}")
        image = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224)).convert("RGB")
        return self.transform(image)


def torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_imagemol_encoder(checkpoint_path: Path, resume_key: str) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if resume_key not in checkpoint:
        raise KeyError(f"Checkpoint has no '{resume_key}' key; available keys: {list(checkpoint)}")

    # Load backbone tensors by name and shape, never by dictionary position.
    # Wrapper prefixes (e.g. module.) are accepted only when unambiguous.
    checkpoint_state = checkpoint[resume_key]
    current_state = model.state_dict()
    if not isinstance(checkpoint_state, dict):
        raise TypeError(f"Checkpoint {resume_key!r} is not a state dictionary.")
    backbone_keys = [key for key in current_state if not key.startswith("fc.")]
    named = {key: key for key in backbone_keys}
    # Official ImageMol uses nn.Sequential(*resnet.children()[:-1]), so its
    # backbone keys are embedding_layer.0, .1, .4, .5, .6, .7 rather than names.
    sequential_modules = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
    official = {key: "embedding_layer." + sequential_modules[key.split(".", 1)[0]] + "." + key.split(".", 1)[1]
                for key in backbone_keys}
    complete_mappings = []
    for mapping in (named, official):
        anchor = mapping["conv1.weight"]
        prefixes = {key[:-len(anchor)] for key in checkpoint_state if key == anchor or key.endswith("." + anchor)}
        for prefix in prefixes:
            if all(prefix + source in checkpoint_state for source in mapping.values()):
                # Deeper ResNets contain all ResNet-18 keys but have extra
                # residual blocks. Reject that superset instead of truncating it.
                roots = {source.split(".", 1)[0] for source in mapping.values()}
                backbone_entries = {key[len(prefix):] for key in checkpoint_state
                                    if key.startswith(prefix) and key[len(prefix):].split(".", 1)[0] in roots}
                if backbone_entries == set(mapping.values()):
                    complete_mappings.append({key: prefix + source for key, source in mapping.items()})
    if len(complete_mappings) != 1:
        raise RuntimeError(f"Expected one complete ResNet-18/ImageMol backbone; found {len(complete_mappings)}. Check --resume-key and checkpoint architecture.")
    source_keys = complete_mappings[0]
    for current_key in current_state:
        if current_key.startswith("fc."):
            continue
        checkpoint_key = source_keys[current_key]
        source = checkpoint_state[checkpoint_key]
        if not isinstance(source, torch.Tensor) or source.shape != current_state[current_key].shape:
            raise RuntimeError(
                f"Checkpoint mismatch: {checkpoint_key} -> "
                f"{current_key} {tuple(current_state[current_key].shape)}"
            )
        if source.is_floating_point() and not torch.isfinite(source).all():
            raise ValueError(f"Nonfinite pretrained tensor: {checkpoint_key}")
        current_state[current_key] = source
    model.load_state_dict(current_state)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    return model


def encoding_metadata(checkpoint_path: Path, resume_key: str = "state_dict") -> dict:
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"checkpoint_sha256": digest.hexdigest(), "resume_key": resume_key,
               "preprocessing": "rdkit_MolsToGridImage_RGB_224_center_crop_imagenet_norm_v1",
               "loading": "exact_resnet18_backbone_v2", "rdkit": rdBase.rdkitVersion,
               "torchvision": torchvision_version, "torch": torch.__version__}


def cache_digest(smiles: list[str], checkpoint_path: Path, resume_key: str = "state_dict") -> str:
    payload = {"smiles": smiles, **encoding_metadata(checkpoint_path, resume_key)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def get_embeddings(args: argparse.Namespace, smiles: list[str]) -> np.ndarray:
    cache = args.embedding_cache or (args.output_dir / "imagemol_resnet18_embeddings.npz")
    if not smiles or args.repr_batch_size < 1 or args.workers < 0:
        raise ValueError("Need nonempty SMILES, positive representation batch size, and nonnegative workers.")
    digest = cache_digest(smiles, args.checkpoint, args.resume_key)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["cache_sha256"].item()) != digest:
                raise ValueError(f"Embedding cache {cache} does not match the dataset/checkpoint/preprocessing. Use a new --embedding-cache path.")
            embeddings = saved["embeddings"].astype(np.float32)
        if embeddings.shape != (len(smiles), 512) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid cached ImageMol embedding shape or nonfinite values.")
        return embeddings

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_imagemol_encoder(args.checkpoint, args.resume_key).to(device)
    loader = DataLoader(
        MoleculeImageDataset(smiles),
        batch_size=args.repr_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    with torch.no_grad():
        for images in loader:
            chunks.append(model(images.to(device)).detach().cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    if embeddings.shape != (len(smiles), 512):
        raise RuntimeError(f"Expected ImageMol embeddings of shape ({len(smiles)}, 512), got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Nonfinite ImageMol embeddings.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings, cache_sha256=np.array(digest))
    return embeddings


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    df = load_bbb_dataframe(args.input)
    embeddings = get_embeddings(args, df["smiles"].astype(str).tolist())
    run_nested_outer_folds(
        df=df,
        embeddings=embeddings,
        encoder_name="imagemol_resnet18_frozen",
        output_dir=args.output_dir,
        model_kinds=args.models,
        beta=args.beta,
        margin_buffer=args.margin_buffer,
        trials=args.trials,
        inner_folds=args.inner_folds,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        tune_loss=args.tune_loss,
        overwrite=args.overwrite,
        encoder_metadata=encoding_metadata(args.checkpoint, args.resume_key),
    )


if __name__ == "__main__":
    main()
