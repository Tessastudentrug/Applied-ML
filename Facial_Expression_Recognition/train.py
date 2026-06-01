import argparse

import torch
from Facial_Expression_Recognition.training.trainer import fit

from Facial_Expression_Recognition.data.data import get_dataloaders
from Facial_Expression_Recognition.models.cnn import CNNImageClassifier
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier


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

    img_size = 224 if args.model == "effnet" else 64

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=32,
        image_size=img_size,
        train_split=0.8,
    )

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
    )

    torch.save(model.state_dict(), f"../models/{args.model}.pth")


if __name__ == "__main__":
    main()