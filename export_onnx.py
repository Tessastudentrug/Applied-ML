import torch
# Import the correct class name!
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier

# 1. Initialize the model using the exact class name Ryan wrote
model = EfficientNetClassifier(num_classes=7) 

# Load your trained weights
model.load_state_dict(torch.load("models/effnet.pth", map_location="cpu"))
model.eval()

# 2. Create a dummy input matching your pipeline (Batch=1, Channels=1 (Grayscale), Size=224x224)
dummy_input = torch.randn(1, 1, 224, 224)

# 3. Export to ONNX format
onnx_path = "models/effnet.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Success! Model exported to {onnx_path}")