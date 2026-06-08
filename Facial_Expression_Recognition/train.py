"""
Training script for facial expression recognition models.

Supports training:
- CNN baseline model
- EfficientNet transfer learning model

Handles:
- Dataset loading
- Model selection
- Optimizer setup
- Training loop execution
- Model saving
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from Facial_Expression_Recognition.training.trainer import fit, evaluate
from Facial_Expression_Recognition.data.data import get_dataloaders, get_datasets
from Facial_Expression_Recognition.models.cnn import CNNImageClassifier
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier


def get_model(model_name, num_classes=7, dropout=0.5):
    """
    Initialize a model based on the selected architecture.

    Args:
        model_name: 'cnn' or 'effnet'
        num_classes: Number of output emotion classes

    Returns:
        Initialized PyTorch model
    """
    if model_name == "cnn":
        return CNNImageClassifier(
            num_classes=num_classes,
            dropout=dropout,
            )

    if model_name == "effnet":
        return EfficientNetClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )

def get_optimizer(model, model_name, optimizer_name, lr, weight_decay):
    if model_name == "effnet":
        if optimizer_name != "AdamW":
            raise ValueError("Use AdamW for effnet")

        return torch.optim.AdamW(
            [
                {"params": model.model.features.parameters(), "lr": lr * 0.1},
                {"params": model.model.classifier.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")

def save_test_results(model, test_loader, device, model_name, suffix=""):
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"TEST | "
        f"loss {test_metrics['loss']:.4f} | "
        f"acc {test_metrics['acc']:.4f} | "
        f"f1 {test_metrics['f1']:.4f}"
    )

    os.makedirs("reports", exist_ok=True)

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"reports/{model_name}{suffix}_classification_report.csv")

    with open(f"reports/{model_name}{suffix}_test_summary.txt", "w") as f:
        f.write("Final Test Set Evaluation\n")
        f.write("=========================\n\n")
        f.write(f"Model: {model_name}{suffix}\n")
        f.write(f"Test loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"Test F1-score: {test_metrics['f1']:.4f}\n\n")
        f.write(classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            zero_division=0,
        ))

    ConfusionMatrixDisplay.from_predictions(
        test_metrics["y_true"],
        test_metrics["y_pred"],
    )

    plt.tight_layout()
    plt.savefig(f"reports/{model_name}{suffix}_confusion_matrix.png")
    plt.close()

    print(f"Saved report to reports/{model_name}{suffix}_test_summary.txt")
    print(f"Saved CSV report to reports/{model_name}{suffix}_classification_report.csv")
    print(f"Saved confusion matrix to reports/{model_name}{suffix}_confusion_matrix.png")


def save_training_curves(hist, model_name):
    os.makedirs("reports", exist_ok=True)

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(
        f"reports/{model_name}_training_history.csv",
        index=False,
    )

    # Loss
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{model_name}_loss_curve.png", dpi=300)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_acc"], label="Train accuracy")
    plt.plot(hist_df["epoch"], hist_df["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{model_name}_accuracy_curve.png", dpi=300)
    plt.close()

    # F1
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["val_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"{model_name.upper()} Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{model_name}_f1_curve.png", dpi=300)
    plt.close()

def main():
    """
    Main training pipeline.

    Steps:
        - Parse CLI arguments
        - Select device (CPU/GPU)
        - Load model
        - Create data loaders
        - Define optimizer
        - Train model using fit()
        - Save trained weights
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "effnet"],
        default="cnn",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--use_best", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224 if args.model == "effnet" else 64

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        image_size=img_size,
        train_split=0.8,
    )

    if args.tune:
        dataset_train, dataset_eval = get_datasets(image_size=img_size) 

        labels = np.array(dataset_train.labels)
        indices = np.arange(len(dataset_train))

        train_val_idx, test_idx, train_val_labels, _ = train_test_split(
            indices,
            labels,
            test_size=0.1,
            stratify=labels,
            random_state=7,
        )

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
            dropout = trial.suggest_float("dropout", 0.2, 0.6)

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
            fold_f1_scores = []
            epochs_per_fold = 8

            for fold, (train_ix, val_ix) in enumerate(
                skf.split(train_val_idx, train_val_labels)
            ):
                fold_train_loader = DataLoader(
                    Subset(dataset_train, train_val_idx[train_ix]),
                    batch_size=32,
                    shuffle=True,
                )

                fold_val_loader = DataLoader(
                    Subset(dataset_eval, train_val_idx[val_ix]),
                    batch_size=32,
                    shuffle=False,
                )

                model = get_model(args.model, dropout=dropout).to(device)

                optimizer = get_optimizer(
                    model=model,
                    model_name=args.model,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay,
                )

                hist = fit(
                    model=model,
                    train_loader=fold_train_loader,
                    val_loader=fold_val_loader,
                    device=device,
                    optimizer=optimizer,
                    max_epochs=epochs_per_fold,
                    patience=2,
                    trial=trial,
                    global_step_offset=fold * epochs_per_fold,
                )

                fold_f1_scores.append(hist[-1]["val_f1"])

            return np.mean(fold_f1_scores)

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=5,
            ),
        )

        study.optimize(objective, n_trials=args.n_trials)

        print("Best F1:", study.best_value)
        print("Best params:", study.best_params)

        best_params = study.best_params

        final_train_idx, final_val_idx, final_train_labels, final_val_labels = train_test_split(
            train_val_idx,
            train_val_labels,
            test_size=0.111111,
            stratify=train_val_labels,
            random_state=7,
        )

        final_train_loader = DataLoader(
            Subset(dataset_train, final_train_idx),
            batch_size=32,
            shuffle=True,
        )

        final_val_loader = DataLoader(
            Subset(dataset_eval, final_val_idx),
            batch_size=32,
            shuffle=False,
        )

        test_loader = DataLoader(
            Subset(dataset_eval, test_idx),
            batch_size=32,
            shuffle=False,
        )

        model = get_model(
            args.model,
            dropout=best_params.get("dropout", 0.5),
        ).to(device)

        optimizer = get_optimizer(
            model=model,
            model_name=args.model,
            optimizer_name=best_params["optimizer"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        hist = fit(
            model=model,
            train_loader=final_train_loader,
            val_loader=final_val_loader,
            device=device,
            optimizer=optimizer,
            max_epochs=30,
            patience=5,
            save_path=f"models/{args.model}_optuna.pth",
        )
      
        save_training_curves(hist, f"{args.model}_optuna")
        save_path = f"models/{args.model}_optuna.pth"
        model.load_state_dict(torch.load(save_path, map_location=device))

        print(f"Saved model to {save_path}")

        save_test_results(
            model,
            test_loader,
            device,
            args.model,
            suffix="_optuna",)
        return

    model = get_model(args.model).to(device)

    if args.model == "effnet":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.model.features.parameters(), "lr": 3e-5},
                {"params": model.model.classifier.parameters(), "lr": 3e-4},
            ],
            weight_decay=1e-3,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    _ = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        max_epochs=30,
        patience=5,
        save_path=f"models/{args.model}.pth",
    )

    print(f"Saved model to models/{args.model}.pth")
 
if __name__ == "__main__":
    main()