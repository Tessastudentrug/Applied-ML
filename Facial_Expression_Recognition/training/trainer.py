"""
Training and evaluation utilities for facial expression recognition models.

Includes:
- evaluate(): model evaluation on validation/test set
- fit(): full training loop with early stopping and metrics tracking
"""

import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, class_weights=None
) -> dict:
    """
    Evaluate model performance on a dataset.

    Computes:
    - Loss (cross entropy)
    - Accuracy
    - Macro F1-score

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation set
        device: CPU or GPU device
        class_weights: Optional class weights for imbalanced datasets

    Returns:
        Dictionary containing loss, accuracy, F1 score, predictions, and labels
    """
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            pred = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / max(1, n),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights=None,
    optimizer=None,
    lr: float = 1e-3,
    max_epochs: int = 20,
    weight_decay: float = 0.0,
    clip_grad_norm: float = None,
    patience: int = 3,
    save_path: str = None,
    trial=None,
    global_step_offset: int = 0,
) -> list:
    """
    Train a PyTorch model with validation monitoring and early stopping.

    Features:
    - Supports custom optimizer or AdamW fallback
    - Tracks accuracy and loss per epoch
    - Uses macro F1 for early stopping
    - Optional gradient clipping
    - Early stopping with best model restore

    Args:
        model: PyTorch model
        train_loader: training dataset loader
        val_loader: validation dataset loader
        device: CPU or GPU device
        class_weights: optional weights for imbalanced classes
        optimizer: optional pre-defined optimizer
        lr: learning rate (if optimizer not provided)
        max_epochs: maximum number of epochs
        weight_decay: L2 regularization
        clip_grad_norm: max gradient norm (optional)
        patience: early stopping patience based on F1 score
        save_path: optional path to save best model
        trial: 

    Returns:
        List of training history dictionaries per epoch
    """
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    if optimizer is not None:
        optim = optimizer
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_f1 = 0.0
    bad_epochs = 0
    hist = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.perf_counter()

        total_loss = 0.0
        n = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optim.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / max(1, n)
        train_acc = correct / max(1, n)
        val = evaluate(model, val_loader, device, class_weights)
        dt = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1": val["f1"],
            "time_s": dt,
        }
        hist.append(record)
        if trial is not None:
            trial.report(val["f1"], epoch + global_step_offset)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['f1']:.4f} | "
            f"time {dt:.1f}s"
        )

        if patience is not None:
            if val["f1"] > best_val_f1 + 1e-4:
                best_val_f1 = val["f1"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            torch.save(best_state, save_path)

    return hist
