import onnx
import torch
from ultralytics import YOLO
from util.config import PYTORCH_MODEL_PATH, ONNX_MODEL_PATH

try:
    yolo_model = YOLO(str(PYTORCH_MODEL_PATH))
    print("Model loaded")
except Exception as e:
    print("Couldnt find model")
    raise e
torch_model = yolo_model.model.eval()

example_inputs = torch.randn(1, 3, 640, 640)

# Legacy exporter (dynamo=False) for yolov8 (too old)
torch.onnx.export(
    torch_model,
    example_inputs,
    str(ONNX_MODEL_PATH),  # Export to models/cat_class_v1.onnx
    export_params=True,
    opset_version=14,  # Recommended for YOLOv8 apparently
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output0'],
    dynamic_axes={
        'images': {0: 'batch_size'},   # Input batch can vary
        'output0': {0: 'batch_size'}   # Output batch can vary
    },
)

print(f"ONNX model exported to: {ONNX_MODEL_PATH}")