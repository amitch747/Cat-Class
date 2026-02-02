"""
Script for comparing best.onnx to best.py
"""
import onnxruntime
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import random
import cv2
from util.config import (PYTORCH_MODEL_PATH, 
                    ONNX_MODEL_PATH, 
                    DATA_DIR,
                    CLASS_NAMES, 
                    CLASS_COLORS,
                    CONFIDENCE_THRESHOLD,
                    NMS_IOU_THRESHOLD)
   
# Get 10 random images
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMAGES_DIR = DATA_DIR / 'test' / 'images'
image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
random.seed(1)
trial_paths = random.sample(image_files, 10)

# Get onnx and torch models
yolo_model = YOLO(PYTORCH_MODEL_PATH)
torch_model = yolo_model.model.eval()

# Setup onnx inference session
ort_sesh = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]) # CPU is better for testing?

# Process images for yolo
def process_for_yolo(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    # re = cv2.resize(img, (640, 640))
    # cv2.imshow("re",re)
    # cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1)) #HWC to CHW
    return img
trial_images = np.array([process_for_yolo(p) for p in trial_paths])

# Need a dict that matches model input (can be more than 1) to input data
onnxruntime_input = {
    input_info.name: data 
    for input_info, data in zip(ort_sesh.get_inputs(), [trial_images])
}

print(f"Input shape: {trial_images.shape}") # 10, 3, 640, 640
print(f"Sample image: {trial_images[0].shape}") # 3, 640, 640
print(f"Onnx input length: {len(onnxruntime_input)}") # Should be 1

# Run trial_images through both models
onnxruntime_outputs = ort_sesh.run(None, onnxruntime_input)[0]
with torch.no_grad():
    torch_input = torch.from_numpy(trial_images) # Convert to torch tensors first
    torch_outputs = torch_model(torch_input)


if isinstance(torch_outputs, (list, tuple)):
    # In YOLOv8, the first element [0] is usually the (Batch, 5, 8400) tensor
    torch_predictions = torch_outputs[0] 
else:
    torch_predictions = torch_outputs


# 10 for batch, 7 for 3 classes and 4 box dimensions, 8400 anchor points
print(f"Torch Predictions Shape: {torch_predictions.shape}")
print(f"ONNX Outputs Shape: {onnxruntime_outputs.shape}")

assert len(torch_predictions) == len(onnxruntime_outputs)


# Compare output |actual - expected| <= atol + rtol * |expected|
for torch_pred, onnx_out in zip(torch_predictions, onnxruntime_outputs):
    torch.testing.assert_close(
        torch_pred, 
        torch.from_numpy(onnx_out), 
        rtol=1e-03, # Relative tolerance
        atol=1e-03  # Absolute tolerance
    ) 
print("PyTorch and ONNX Runtime output are equal within tolerance")

predictions = onnxruntime_outputs[0].T  #(8700, 7)

scores = predictions[:, 4:] # Grab class scores (last 3 elements) for each anchor point (8400, 3)
max_scores = np.max(scores, axis=1)  # (8400,) highest score for the 3 cats
class_ids = np.argmax(scores, axis=1)  # (8400,) which cat has highest score

# Filter for weak class scores
mask = max_scores > CONFIDENCE_THRESHOLD
final_boxes_raw = predictions[mask, :4] # For the remaining anchor points, 
final_scores_raw = max_scores[mask]
final_class_ids = class_ids[mask]
# print(final_scores_raw)
print(final_class_ids)

if len(final_boxes_raw) > 0:
    # CXCYWH 
    center_x  = final_boxes_raw[:, 0]
    center_y = final_boxes_raw[:, 1]
    width = final_boxes_raw[:, 2]
    height = final_boxes_raw[:, 3]
    # to XY-XY
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) # stack as columns
    #print(boxes_xyxy.shape)

    # Grab original image 
    img = cv2.imread(str(trial_paths[0]))
    h, w = img.shape[:2]
    img_copy = img.copy()

    # Show image with initial bounding boxes
    print(f"Bboxes before NMS: {len(boxes_xyxy)}")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 128, 0), (0, 128, 255)]
    for i, box in enumerate(boxes_xyxy):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        color = colors[i % len(colors)]  # Cycle through colors
        cv2.rectangle(img_copy, p1, p2, color, 1, cv2.LINE_AA)
        # Add box number
        cv2.putText(img_copy, str(i+1), (p1[0]+i, p1[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow("Before NMS", img_copy)


    # Filter out the extra boxes (IoU over best box)
    import torchvision
    keep = torchvision.ops.nms(torch.from_numpy(boxes_xyxy), 
                               torch.from_numpy(final_scores_raw), 
                               NMS_IOU_THRESHOLD)
    
    # Reshape incase 1 box returns shape of (4,)
    final_boxes = boxes_xyxy[keep].reshape(-1, 4)
    final_scores = final_scores_raw[keep].reshape(-1)
    final_class_ids = final_class_ids[keep].reshape(-1)


    print(f"Bboxes after NMS: {len(final_boxes)}")
    for box, score, id in zip(final_boxes, final_scores, final_class_ids):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        cv2.rectangle(img, p1, p2, CLASS_COLORS[id], 2)
        cv2.putText(img, CLASS_NAMES[id], (p1[0], p1[1]-7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[id], 1)
        cv2.putText(img, f"{(score):.3f}", (p1[0], p2[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[id], 1)
        
    cv2.imshow("After NSM", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No cats detected above the 0.5 threshold")