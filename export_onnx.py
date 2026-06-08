"""
ONNX Export Script for Edge AI Deployment

This script takes the hyperparameter-tuned PyTorch model (effnet_optuna.pth)
and translates it into an ONNX format (effnet_combined.onnx). 

Converting it to ONNX allows the index.html web app to run the AI completely 
offline using the user's local device, ensuring a privacy-first architecture.
"""

import torch
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier

print("Grabbing the newly tuned Optuna model...")

model = EfficientNetClassifier(num_classes=7) 
model.load_state_dict(torch.load("models/effnet_optuna.pth", map_location="cpu"))
model.eval() 

dummy_input = torch.randn(1, 1, 224, 224)

print("Translating PyTorch into ONNX for the browser...")

onnx_path = "effnet_combined.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Success! The model was exported directly to {onnx_path}!")