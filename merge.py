"""
ONNX Model Merge Script

This script loads the split ONNX model files (the main .onnx file and its 
external .data file) and merges them into a single, self-contained .onnx file.
This simplifies deployment by allowing the web application to load a single file.
"""

import onnx

print("Loading the split ONNX model and external data...")
model = onnx.load("effnet_combined.onnx", load_external_data=True)

print("Merging into a single, self-contained file...")
onnx.save(model, "effnet_combined_merged.onnx")

print("Success! The merged model was saved as 'effnet_combined_merged.onnx'.")