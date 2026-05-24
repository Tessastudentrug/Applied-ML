import torch

from data.data import get_dataloaders
from models.cnn import CNNImageClassifier
from models.effnet import EfficientNetClassifier
from training.trainer import fit
import argparse


def get_model(model_name, num_classes=7):
    if model_name == "cnn":
        return CNNImageClassifier(num_classes=num_classes)

    if model_name == "effnet":
        return EfficientNetClassifier(num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "effnet"],
        default="cnn",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=32,
        image_size=64,
        train_split=0.8,
    )


    shistory = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-4,
        max_epochs=20,
        weight_decay=1e-4,
        patience=5,)
    torch.save(model.state_dict(), f"../models/{args.model}.pth")


if __name__ == "__main__":
    main()
