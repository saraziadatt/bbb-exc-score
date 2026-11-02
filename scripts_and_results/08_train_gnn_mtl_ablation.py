#!/usr/bin/env python3
"""
Test whether the custom BBB multi-task loss transfers to a graph neural network.

"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from pretrained_bbb_common import squared_mtl_loss, validate_bbb_dataframe, dataframe_inner_splits, prepare_output_directory
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


MODEL_NAMES = {
    "mse": "gnn_mse",
    "mtl": "gnn_custom_mtl",
}

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRALITIES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot_with_other(value, choices) -> list[float]:
    return [float(value == choice) for choice in choices] + [
        float(value not in choices)
    ]


def atom_features(atom: Chem.Atom) -> list[float]:
    features = [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
        atom.GetMass() / 200.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]
    features += one_hot_with_other(
        atom.GetHybridization(),
        HYBRIDIZATIONS,
    )
    features += one_hot_with_other(
        atom.GetChiralTag(),
        CHIRALITIES,
    )
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    features = one_hot_with_other(
        bond.GetBondType(),
        BOND_TYPES,
    )
    features += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    features += one_hot_with_other(
        bond.GetStereo(),
        BOND_STEREOS,
    )
    return features


NODE_DIM = (
    7
    + len(HYBRIDIZATIONS)
    + 1
    + len(CHIRALITIES)
    + 1
)

EDGE_DIM = (
    len(BOND_TYPES)
    + 1
    + 2
    + len(BOND_STEREOS)
    + 1
)


def smiles_to_graph(
    smiles: str,
    y_reg: float,
    y_class: int,
    row_index: int,
) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            f"RDKit could not parse row {row_index}: {smiles}"
        )

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )

    edge_pairs = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = bond_features(bond)

        edge_pairs.extend([[i, j], [j, i]])
        edge_features.extend([features, features])

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).T.contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float32,
        )
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        edge_attr = torch.empty(
            (0, EDGE_DIM),
            dtype=torch.float32,
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_reg=torch.tensor(
            [y_reg],
            dtype=torch.float32,
        ),
        y_class=torch.tensor(
            [y_class],
            dtype=torch.long,
        ),
        row_index=torch.tensor(
            [row_index],
            dtype=torch.long,
        ),
    )


class BBBGINE(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(
            NODE_DIM,
            hidden_dim,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    train_eps=True,
                    edge_dim=EDGE_DIM,
                )
            )

            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

        self.dropout = dropout

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: Data,
    ) -> torch.Tensor:
        h = self.node_encoder(batch.x)

        for conv, norm in zip(self.convs, self.norms):
            update = conv(
                h,
                batch.edge_index,
                batch.edge_attr,
            )
            h = norm(h + update)
            h = F.relu(h)
            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

        graph_embedding = global_mean_pool(
            h,
            batch.batch,
        )

        return self.head(graph_embedding).view(-1)


def custom_mtl_loss(predictions, y_reg, y_class, *, alpha, beta, margin,
                    margin_buffer, class_counts=None):
    """Exact unscaled objective shared with the frozen encoders and LightGBM."""
    return squared_mtl_loss(predictions, y_class, y_reg, torch.isfinite(y_reg),
                            beta, margin_buffer, alpha=alpha, tau=margin, class_counts=class_counts)


def calculate_loss(predictions, batch, loss_mode, args, class_counts=None):
    y_reg, y_class = batch.y_reg.view(-1), batch.y_class.view(-1)
    if loss_mode == "mse":
        measured = torch.isfinite(y_reg)
        # Same summed regression scale as the MTL regression component.
        return 0.5 * args.alpha * (predictions[measured] - y_reg[measured]).square().sum()
    if loss_mode != "mtl":
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    return custom_mtl_loss(predictions, y_reg, y_class, alpha=args.alpha, beta=args.beta,
                           margin=args.threshold, margin_buffer=args.margin_buffer, class_counts=class_counts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(graphs, batch_size, shuffle, seed, num_workers):
    if not graphs or batch_size < 1:
        raise ValueError("Need graphs and a positive batch size.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=num_workers)


def train_epoch(model, loader, optimizer, device, loss_mode, args):
    model.train()
    classes = torch.cat([g.y_class.view(-1) for g in loader.dataset])
    counts = torch.bincount(classes.long(), minlength=2).to(device)
    # Sum additive batch contributions BEFORE clipping and stepping. This is
    # one full-partition update, with memory bounded by the graph batch size.
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        loss = calculate_loss(model(batch), batch, loss_mode, args, counts)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite GNN training loss.")
        loss.backward()
        total += float(loss.detach().cpu())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def validation_loss(model, loader, device, loss_mode, args):
    model.eval()
    predictions, targets, classes = [], [], []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).cpu())
        targets.append(batch.y_reg.view(-1).cpu())
        classes.append(batch.y_class.view(-1).cpu())
    temporary = Data(y_reg=torch.cat(targets), y_class=torch.cat(classes))
    loss = calculate_loss(torch.cat(predictions), temporary, loss_mode, args)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite GNN validation loss.")
    return float(loss)


def new_model_optimizer(args, device, seed):
    set_seed(seed)
    model = BBBGINE(args.hidden_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def select_training_epoch(fit_graphs, validation_graphs, loss_mode, args, device, seed):
    model, optimizer = new_model_optimizer(args, device, seed)
    fit_loader = make_loader(fit_graphs, args.batch_size, True, seed, args.num_workers)
    val_loader = make_loader(validation_graphs, args.batch_size, False, seed, args.num_workers)
    history, best_epoch, best_loss, best_state, stale = [], 0, np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, fit_loader, optimizer, device, loss_mode, args)
        validation = validation_loss(model, val_loader, device, loss_mode, args)
        history.append({"epoch": epoch, "training_loss": training, "validation_loss": validation})
        if validation < best_loss - args.min_delta:
            best_loss, best_epoch, stale = validation, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation loss was obtained.")
    model.load_state_dict(best_state)
    return best_epoch, history, model


def fit_final_model(graphs, loss_mode, epochs, args, device, seed):
    model, optimizer = new_model_optimizer(args, device, seed)
    loader = make_loader(graphs, args.batch_size, True, seed, args.num_workers)
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device, loss_mode, args)
    return model


@torch.no_grad()
def predict(model, graphs, args, device):
    model.eval()
    loader = make_loader(graphs, args.batch_size, False, args.seed, args.num_workers)
    predictions = np.concatenate([model(batch.to(device)).cpu().numpy() for batch in loader])
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite GNN predictions.")
    return predictions


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    measured = np.isfinite(y_true)

    y_true = y_true[measured]
    y_pred = y_pred[measured]

    if len(y_true) == 0:
        return {
            "n_measured": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
        }

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "n_measured": len(y_true),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": (
            r2_score(y_true, y_pred)
            if len(y_true) > 1
            else np.nan
        ),
        "pearson_r": (
            np.corrcoef(y_true, y_pred)[0, 1]
            if (
                len(y_true) > 1
                and np.std(y_true) > 0
                and np.std(y_pred) > 0
            )
            else np.nan
        ),
    }


def classification_metrics(
    y_true: np.ndarray,
    predicted_logbb: np.ndarray,
    threshold: float,
) -> dict:
    predicted_class = (
        predicted_logbb < threshold
    ).astype(int)

    score_nonpenetrant = -predicted_logbb
    both_classes = len(np.unique(y_true)) == 2
    penetrant = y_true == 0

    return {
        "n_classification": len(y_true),
        "accuracy": accuracy_score(
            y_true,
            predicted_class,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predicted_class,
        ) if both_classes else np.nan,
        "np_recall": recall_score(
            y_true,
            predicted_class,
            pos_label=1,
            zero_division=0,
        ),
        "np_precision": precision_score(
            y_true,
            predicted_class,
            pos_label=1,
            zero_division=0,
        ),
        "np_f1": f1_score(
            y_true,
            predicted_class,
            pos_label=1,
            zero_division=0,
        ),
        "penetrant_specificity": (
            np.mean(
                predicted_class[penetrant] == 0
            )
            if penetrant.any()
            else np.nan
        ),
        "roc_auc": (
            roc_auc_score(
                y_true,
                score_nonpenetrant,
            )
            if both_classes
            else np.nan
        ),
        "pr_auc_nonpenetrant": (
            average_precision_score(
                y_true,
                score_nonpenetrant,
            )
            if both_classes
            else np.nan
        ),
        "mcc": matthews_corrcoef(
            y_true,
            predicted_class,
        ),
        "wrong_side_count": int(
            np.sum(y_true != predicted_class)
        ),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:
    y_reg = predictions[
        "true_logBB"
    ].to_numpy(float)

    y_class = predictions[
        "bbb_class"
    ].to_numpy(int)

    predicted_logbb = predictions[
        "predicted_logBB"
    ].to_numpy(float)

    return {
        **regression_metrics(
            y_reg,
            predicted_logbb,
        ),
        **classification_metrics(
            y_class,
            predicted_logbb,
            threshold,
        ),
    }


def suggest_hyperparameters(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    trial_args = copy.copy(base_args)

    trial_args.hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [64, 128, 256],
    )

    trial_args.layers = trial.suggest_int(
        "layers",
        2,
        5,
    )

    trial_args.dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.40,
        step=0.05,
    )

    trial_args.learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True,
    )

    trial_args.weight_decay = trial.suggest_float(
        "weight_decay",
        1e-7,
        1e-3,
        log=True,
    )

    trial_args.epochs = (
        base_args.hpo_epochs
        if base_args.hpo_epochs is not None
        else base_args.epochs
    )

    trial_args.patience = base_args.hpo_patience

    return trial_args


def choose_pareto_trial(
    study: optuna.Study,
) -> optuna.trial.FrozenTrial:
    """
    Select a single Pareto-optimal trial using normalized distance to:

        minimum MSE
        maximum balanced accuracy

    Both objectives therefore contribute equally after min-max normalization.
    """
    candidates = [
        trial
        for trial in study.best_trials
        if (
            trial.values is not None
            and len(trial.values) == 2
            and np.all(np.isfinite(trial.values))
        )
    ]

    if not candidates:
        raise RuntimeError(
            "Optuna did not produce a finite Pareto-optimal trial."
        )

    mse_values = np.asarray(
        [trial.values[0] for trial in candidates],
        dtype=float,
    )

    ba_values = np.asarray(
        [trial.values[1] for trial in candidates],
        dtype=float,
    )

    mse_range = (
        mse_values.max()
        - mse_values.min()
    )

    ba_range = (
        ba_values.max()
        - ba_values.min()
    )

    if mse_range > 0:
        normalized_mse = (
            mse_values - mse_values.min()
        ) / mse_range
    else:
        normalized_mse = np.zeros_like(
            mse_values
        )

    if ba_range > 0:
        normalized_ba_gap = (
            ba_values.max() - ba_values
        ) / ba_range
    else:
        normalized_ba_gap = np.zeros_like(
            ba_values
        )

    distances = np.sqrt(
        normalized_mse**2
        + normalized_ba_gap**2
    )

    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            distances[index],
            candidates[index].number,
        ),
    )

    return candidates[selected_index]


def evaluate_inner_cv(graphs, outer_indices, inner_splits, df, loss_mode, args, device, seed):
    scores, best_epochs, histories = [], [], []
    for inner, (tr, va) in enumerate(inner_splits):
        fit_idx, val_idx = outer_indices[tr], outer_indices[va]
        if loss_mode == "mse":
            fit_idx = fit_idx[df.iloc[fit_idx]["_has_logbb"].to_numpy(bool)]
            stop_idx = val_idx[df.iloc[val_idx]["_has_logbb"].to_numpy(bool)]
        else:
            stop_idx = val_idx
        best, history, model = select_training_epoch(
            [graphs[i] for i in fit_idx], [graphs[i] for i in stop_idx],
            loss_mode, args, device, seed + inner)
        predictions = predict(model, [graphs[i] for i in val_idx], args, device)
        measured = df.iloc[val_idx]["_has_logbb"].to_numpy(bool)
        mse = mean_squared_error(df.iloc[val_idx]["logBB"].to_numpy(float)[measured], predictions[measured])
        ba = balanced_accuracy_score(df.iloc[val_idx]["bbb_class"].to_numpy(int), (predictions < args.threshold).astype(int))
        scores.append((mse, ba))
        best_epochs.append(best)
        histories.extend({**row, "inner_fold": inner, "inner_best_epoch": best} for row in history)
        del model
    return np.mean(scores, axis=0), best_epochs, histories


def tune_hyperparameters(*, graphs, outer_indices, inner_splits, df, loss_mode,
                         model_name, outer_fold, args, device, seed):
    def objective(trial):
        trial_args = suggest_hyperparameters(trial, args)
        if args.tune_loss and loss_mode == "mtl":
            trial_args.beta = trial.suggest_float("beta", 0.5, 5.0, step=0.25)
            trial_args.margin_buffer = trial.suggest_float("margin_buffer", 0.15, 0.35, step=0.02)
        scores, epochs, _ = evaluate_inner_cv(graphs, outer_indices, inner_splits, df,
                                             loss_mode, trial_args, device, seed)
        trial.set_user_attr("inner_best_epochs", epochs)
        trial.set_user_attr("selected_epoch", int(np.median(epochs)))
        return tuple(float(v) for v in scores)

    study = optuna.create_study(directions=["minimize", "maximize"],
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=args.hpo_trials, gc_after_trial=True,
                   show_progress_bar=args.hpo_progress_bar)
    chosen = choose_pareto_trial(study)
    tuned_args = copy.copy(args)
    for key, value in chosen.params.items():
        setattr(tuned_args, key, value)
    rows = [{"model": model_name, "outer_fold": int(outer_fold), "trial": t.number,
             "state": t.state.name, "selected": t.number == chosen.number,
             "validation_mse": t.values[0] if t.values is not None else np.nan,
             "validation_balanced_accuracy": t.values[1] if t.values is not None else np.nan,
             "selected_epoch": t.user_attrs.get("selected_epoch"), **t.params} for t in study.trials]
    selection = {"model": model_name, "outer_fold": int(outer_fold), "selected_trial": chosen.number,
                 "validation_mse": chosen.values[0], "validation_balanced_accuracy": chosen.values[1],
                 "parameters": dict(chosen.params), "inner_best_epochs": chosen.user_attrs["inner_best_epochs"]}
    return tuned_args, rows, selection


def build_comparison_table(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    preferred_direction = {
        "mae": "lower",
        "mse": "lower",
        "rmse": "lower",
        "r2": "higher",
        "pearson_r": "higher",
        "accuracy": "higher",
        "balanced_accuracy": "higher",
        "np_recall": "higher",
        "np_precision": "higher",
        "np_f1": "higher",
        "penetrant_specificity": "higher",
        "roc_auc": "higher",
        "pr_auc_nonpenetrant": "higher",
        "mcc": "higher",
        "wrong_side_count": "lower",
    }

    rows = []

    for metric, direction in preferred_direction.items():
        pivot = fold_metrics.pivot(
            index="outer_fold",
            columns="model",
            values=metric,
        ).dropna()

        if not set(
            MODEL_NAMES.values()
        ).issubset(pivot.columns):
            continue

        mse_values = pivot[
            MODEL_NAMES["mse"]
        ]

        mtl_values = pivot[
            MODEL_NAMES["mtl"]
        ]

        delta = mtl_values - mse_values

        rows.append(
            {
                "metric": metric,
                "preferred_direction": direction,
                "gnn_mse_mean": mse_values.mean(),
                "gnn_mse_sd": mse_values.std(ddof=1),
                "gnn_custom_mtl_mean": (
                    mtl_values.mean()
                ),
                "gnn_custom_mtl_sd": (
                    mtl_values.std(ddof=1)
                ),
                "delta_mtl_minus_mse": (
                    delta.mean()
                ),
                "n_folds_mtl_better": int(
                    np.sum(delta > 0)
                    if direction == "higher"
                    else np.sum(delta < 0)
                ),
                "n_outer_folds": len(delta),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--smiles-col",
        default="smiles",
    )
    parser.add_argument(
        "--target-col",
        default="logBB",
    )
    parser.add_argument(
        "--class-col",
        default="bbb_class",
    )
    parser.add_argument(
        "--fold-col",
        default="outer_fold",
    )
    parser.add_argument(
        "--id-col",
        default="sample_id",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--margin-buffer",
        type=float,
        default=0.25,
    )

    # These values are used directly when --hpo-trials is 0.
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=(
            "Maximum epochs for final epoch selection."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=30,
        help=(
            "Optuna trials per model per outer fold. "
            "Use 0 to disable HPO."
        ),
    )
    parser.add_argument(
        "--hpo-epochs",
        type=int,
        default=75,
        help=(
            "Maximum epochs per HPO trial. "
            "Use the same value as --epochs for the "
            "strictest comparison."
        ),
    )
    parser.add_argument(
        "--hpo-patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--hpo-progress-bar",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--tune-loss", action="store_true", help="Tune MTL beta and margin buffer using the methods search spaces.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this trainer's existing result files and checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hpo_trials < 0 or args.inner_folds < 2 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid training/CV settings.")
    if args.hpo_trials and (args.hpo_epochs < 1 or args.hpo_patience < 1):
        raise ValueError("HPO epochs and patience must be positive.")
    if args.tune_loss and args.hpo_trials == 0:
        raise ValueError("--tune-loss requires --hpo-trials > 0.")
    if not np.isfinite([args.alpha, args.beta, args.margin_buffer, args.threshold, args.learning_rate, args.weight_decay, args.min_delta]).all() or args.alpha <= 0 or args.beta < 0 or args.margin_buffer < 0 or args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("Invalid loss/optimizer parameters.")
    if args.layers < 1 or args.hidden_dim < 1 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid GNN architecture parameters.")
    column_roles = [args.smiles_col, args.target_col, args.class_col, args.fold_col, args.id_col]
    if len(set(column_roles)) != len(column_roles):
        raise ValueError("Input column options must name distinct columns for SMILES, target, class, fold, and ID.")
    source = pd.read_csv(args.input, low_memory=False)
    required_columns = {args.smiles_col, args.target_col, args.class_col, args.fold_col}
    if args.id_col != "sample_id":
        required_columns.add(args.id_col)
    missing_columns = required_columns.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Missing requested input columns: {sorted(missing_columns)}")
    rename = {args.smiles_col: "smiles", args.target_col: "logBB", args.class_col: "bbb_class",
              args.fold_col: "outer_fold", args.id_col: "sample_id"}
    df = validate_bbb_dataframe(source.rename(columns=rename))
    if not df.columns.is_unique:
        raise ValueError("Custom column names collided with the canonical input columns.")
    outer_folds = sorted(df.outer_fold.unique())
    splits_by_fold = {fold: dataframe_inner_splits(df, fold, np.flatnonzero(df.outer_fold.to_numpy() != fold),
                       args.inner_folds, args.seed + 1000 * pos) for pos, fold in enumerate(outer_folds)}
    model_dir = args.output_dir / "fitted_models"
    owned_checkpoints = [path for name in MODEL_NAMES.values() for path in model_dir.glob(f"{name}_outer_*.pt")]
    if owned_checkpoints and not args.overwrite:
        raise FileExistsError(f"GNN checkpoints already exist in {model_dir}. Use a new --output-dir or --overwrite.")
    prepare_output_directory(args.output_dir,
        ["gnn_predictions.csv", "gnn_fold_metrics.csv", "gnn_pooled_metrics.csv", "gnn_mtl_vs_mse_comparison.csv",
         "gnn_training_history.csv", "inner_folds_used.csv", "gnn_hpo_trials.csv", "gnn_hpo_selected_parameters.json", "run_manifest.json"],
        [], args.overwrite)
    for path in owned_checkpoints:
        path.unlink()
    model_dir.mkdir(exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}; creating molecular graphs", flush=True)
    graphs = [smiles_to_graph(str(row.smiles), float(row.logBB), int(row.bbb_class), i)
              for i, row in df.iterrows()]
    prediction_parts, metric_rows, history_rows, hpo_rows, fold_rows = [], [], [], [], []
    selections = {}
    for fold_position, fold in enumerate(outer_folds):
        train_idx = np.flatnonzero(df.outer_fold.to_numpy() != fold)
        test_idx = np.flatnonzero(df.outer_fold.to_numpy() == fold)
        seed = args.seed + 1000 * fold_position
        splits = splits_by_fold[fold]
        for inner, (_, va) in enumerate(splits):
            fold_rows.extend({"outer_fold_model": int(fold), "sample_id": df.iloc[i].sample_id,
                              "inner_fold": inner} for i in train_idx[va])
        for mode, model_name in MODEL_NAMES.items():
            print(f"[{model_name}] outer fold {fold}", flush=True)
            if args.hpo_trials:
                model_args, rows, selection = tune_hyperparameters(
                    graphs=graphs, outer_indices=train_idx, inner_splits=splits, df=df,
                    loss_mode=mode, model_name=model_name, outer_fold=fold,
                    args=args, device=device, seed=seed)
                hpo_rows.extend(rows)
                selections[f"{model_name}_outer_{fold}"] = selection
            else:
                model_args = copy.copy(args)
            # Select the training duration across every inner fold, using the
            # requested final epoch budget; never use the outer test fold.
            _, epochs, history = evaluate_inner_cv(graphs, train_idx, splits, df, mode, model_args, device, seed)
            final_epochs = max(1, int(np.median(epochs)))
            hyperparameters = {k: getattr(model_args, k) for k in
                               ("hidden_dim", "layers", "dropout", "batch_size", "learning_rate", "weight_decay")}
            history_rows.extend({"model": model_name, "outer_fold": int(fold), "selected_epoch": final_epochs,
                                 **hyperparameters, **row} for row in history)
            fit_idx = train_idx[df.iloc[train_idx]["_has_logbb"].to_numpy(bool)] if mode == "mse" else train_idx
            model = fit_final_model([graphs[i] for i in fit_idx], mode, final_epochs, model_args, device, seed)
            prediction = predict(model, [graphs[i] for i in test_idx], model_args, device)
            part = pd.DataFrame({"sample_id": df.iloc[test_idx].sample_id.to_numpy(), "row_index": df.iloc[test_idx]["_source_row"].to_numpy(),
                                 "smiles": df.iloc[test_idx].smiles.to_numpy(), "outer_fold": int(fold),
                                 "model": model_name, "model_task": "regression",
                                 "true_logBB": df.iloc[test_idx].logBB.to_numpy(float),
                                 "bbb_class": df.iloc[test_idx].bbb_class.to_numpy(int),
                                 "has_logBB": df.iloc[test_idx]["_has_logbb"].to_numpy(bool),
                                 "classification_only": ~df.iloc[test_idx]["_has_logbb"].to_numpy(bool),
                                 "predicted_logBB": prediction, "score_nonpenetrant": -prediction,
                                 "predicted_bbb_class": (prediction < model_args.threshold).astype(int),
                                 "classification_threshold": model_args.threshold})
            part["absolute_error"] = np.abs(part.true_logBB - part.predicted_logBB)
            part["wrong_side"] = part.bbb_class != part.predicted_bbb_class
            for key, value in hyperparameters.items():
                part[f"hpo_{key}"] = value
            prediction_parts.append(part)
            metric_rows.append({"model": model_name, "outer_fold": int(fold), "selected_epoch": final_epochs,
                                **hyperparameters, **evaluate_predictions(part, model_args.threshold)})
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_name": model_name, "outer_fold": int(fold), **hyperparameters,
                        "threshold": model_args.threshold, "alpha": model_args.alpha, "beta": model_args.beta,
                        "margin_buffer": model_args.margin_buffer, "selected_epoch": final_epochs,
                        "hpo_enabled": args.hpo_trials > 0, "hpo_trials": args.hpo_trials,
                        "loss_version": "half_sum_squared_error_plus_class_mean_squared_hinges_v1",
                        "optimizer_updates_per_epoch": 1, "inner_best_epochs": epochs},
                       model_dir / f"{model_name}_outer_{fold}.pt")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["model", "row_index"])
    fold_metrics = pd.DataFrame(metric_rows)
    pooled = pd.DataFrame([{"model": name, **evaluate_predictions(group, args.threshold)}
                           for name, group in predictions.groupby("model")])
    comparison = build_comparison_table(fold_metrics)
    for name, frame in (("gnn_predictions.csv", predictions), ("gnn_fold_metrics.csv", fold_metrics),
                        ("gnn_pooled_metrics.csv", pooled), ("gnn_mtl_vs_mse_comparison.csv", comparison),
                        ("gnn_training_history.csv", pd.DataFrame(history_rows)),
                        ("inner_folds_used.csv", pd.DataFrame(fold_rows))):
        frame.to_csv(args.output_dir / name, index=False)
    if hpo_rows:
        pd.DataFrame(hpo_rows).to_csv(args.output_dir / "gnn_hpo_trials.csv", index=False)
    (args.output_dir / "gnn_hpo_selected_parameters.json").write_text(json.dumps(selections, indent=2))
    manifest = {"input": str(args.input.resolve()), "models": MODEL_NAMES, "architecture": "GINE",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "outer_folds": [int(f) for f in outer_folds], "device": str(device),
                "loss": "alpha/2*sum_measured(error^2) + beta*sum_class_means(squared_margin_violation)",
                "loss_scaling": "none", "updates_per_epoch": 1,
                "batching": "additive gradient accumulation, then gradient clipping and one AdamW step",
                "inner_cv": "all saved inner folds, or class-by-measured-status stratification",
                "hpo_sampler": "TPE", "hpo_objectives": ["mean inner MSE", "mean inner balanced accuracy"],
                "selection": "normalized Pareto distance; median inner best epoch for final refit",
                "comparison_scope": "objective plus classification-only supervision; model hyperparameters tuned independently"}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Completed successfully.")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
