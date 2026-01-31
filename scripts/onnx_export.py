import onnx
import torch
from ultralytics import YOLO
from pathlib import Path

# Backout from scripts and grab best model
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "runs" / "detect" / "train" / "weights" / "best.pt"

try:
    yolo_model = YOLO(str(MODEL_PATH))
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
    "cat_class_basic.onnx",
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