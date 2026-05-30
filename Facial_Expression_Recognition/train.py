import argparse

import torch
from training.trainer import fit

from data.data import get_dataloaders
from models.cnn import CNNImageClassifier
from models.effnet import EfficientNetClassifier


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

    # 1. DYNAMIC IMAGE SIZE
    # Automatically use 224 for EffNet, but keep 64 for CNN so nothing breaks!
    img_size = 224 if args.model == "effnet" else 64

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=32,
        image_size=img_size,  # Uses the dynamic variable here
        train_split=0.8,
    )

    # 2. SETUP OPTIMIZER WITH DIFFERENTIAL LEARNING RATES
    if args.model == "effnet":
        # Protect the ImageNet weights with 1e-5, train the new classifier with 1e-3
        optimizer = torch.optim.Adam(
            [
                {"params": model.model.features.parameters(), "lr": 1e-5},
                {"params": model.model.classifier.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
    else:
        # Standard optimizer for CNN
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 3. RUN TRAINING
    _ = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,  # Pass our custom optimizer into fit()
        max_epochs=20,
        patience=5,
    )

    torch.save(model.state_dict(), f"../models/{args.model}.pth")


if __name__ == "__main__":
    main()
