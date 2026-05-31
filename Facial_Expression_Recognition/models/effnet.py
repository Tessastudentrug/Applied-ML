import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        # Setup the custom classifier with the Dropout restored
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),  # Re-added the 0.5 Dropout
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # EfficientNet expects 3 channels (RGB),
        #  so we copy the grayscale channel 3 times
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
