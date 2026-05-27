from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7):

        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),                       # Dropout
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # EfficientNet expects 3 channels (RGB), triple greyscale?
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
