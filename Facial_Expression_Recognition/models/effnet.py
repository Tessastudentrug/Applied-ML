from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        # 1. First, freeze the entire backbone so we don't destroy ImageNet weights
        for param in self.model.features.parameters():
            param.requires_grad = False

        # 2. Now, UNFREEZE the last 4 layers/blocks of the features
        # This allows the highest-level filters to adapt to faces
        for block in self.model.features[-4:]:
            for param in block.parameters():
                param.requires_grad = True

        # 3. Setup the custom classifier with the Dropout restored
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),                       # Re-added the 0.5 Dropout
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # EfficientNet expects 3 channels (RGB), so we copy the grayscale channel 3 times
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)