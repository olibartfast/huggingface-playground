import torch
from transformers import VitPoseForPoseEstimation, VitPoseConfig
import onnx
import onnxruntime
import numpy as np

# Load the model
model_name = "usyd-community/vitpose-base-simple"
print(f"Loading model: {model_name}")

try:
    model = VitPoseForPoseEstimation.from_pretrained(model_name)
except OSError:
    # Fallback to a different checkpoint if that one doesn't exist or isn't accessible
    # This is a common one, but let's try to be robust.
    # If the user didn't specify, we pick a standard one.
    print(f"Could not load {model_name}, trying another if available or failing.")
    raise

model.eval()

# Create dummy input
# ViTPose usually takes standard image sizes, e.g., 256x192
height = 256
width = 192
dummy_input = torch.randn(1, 3, height, width)

# Export to ONNX
onnx_path = "vitpose_base.onnx"
print(f"Exporting to {onnx_path}...")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Export complete.")

# Verify ONNX model
print("Verifying ONNX model...")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Run inference with ONNX Runtime to check
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

print("ONNX Runtime inference successful.")
print("Output shape:", ort_outs[0].shape)
