import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import time
from util.config import ONNX_MODEL_PATH, CLASS_NAMES, CLASS_COLORS, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD

# Setup Onnx Session
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # Try to get GPU
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
input_name = onnx_session.get_inputs()[0].name
print(f"input_name: {input_name}")

# Setup camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # WSL2 exposes USB cams as V4L2
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

print(f"Current provider: {onnx_session.get_providers()[0]}")

while True:
    ret, frame = cap.read()
    if not ret: break

    display_frame = frame.copy()
    start_time = time.time()
    h, w = frame.shape[:2]

    img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA) # INTER_AREA avoids aliasing when shrinking
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # H,W,C -> C,H,W
    img = np.transpose(img, (2, 0, 1))
    # Need a batch dimension first
    img = np.expand_dims(img, axis=0)

    outputs = onnx_session.run(None, {input_name: img})[0] # start of list is actual output
    # shape = (1, 7, 8400)
    # print(outputs.shape)
    predictions = outputs[0].T # select single batch, transpose (8400, 7)

    scores = predictions[:, 4:] # grab cat scores
    max_scores = np.max(scores, axis=1) # highest confidence in anchor point 
    class_ids = np.argmax(scores, axis=1)  # which cat is it

    confidence_mask = max_scores > CONFIDENCE_THRESHOLD
    if np.any(confidence_mask):
        final_boxes_cxcywh = predictions[confidence_mask, :4]
        final_scores = max_scores[confidence_mask]
        final_class_ids = class_ids[confidence_mask]
        # CXCYWH 
        center_x  = final_boxes_cxcywh[:, 0]
        center_y = final_boxes_cxcywh[:, 1]
        width = final_boxes_cxcywh[:, 2]
        height = final_boxes_cxcywh[:, 3]
        # to XY-XY
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        final_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) # stack as columns

        # NMS
        keep = torchvision.ops.nms(torch.from_numpy(final_boxes_xyxy), torch.from_numpy(final_scores), NMS_IOU_THRESHOLD)
        # keep should be the indexs of anchor point with a cat

        final_boxes = final_boxes_xyxy[keep].reshape(-1, 4)
        final_scores = final_scores[keep].reshape(-1)
        final_class_ids = final_class_ids[keep].reshape(-1)


        for box, score, id in zip(final_boxes, final_scores, final_class_ids):
            p1 = (int(box[0] / 640 * w), int(box[1] / 640 * h))
            p2 = (int(box[2] / 640 * w), int(box[3] / 640 * h))
            cv2.rectangle(display_frame, p1, p2, CLASS_COLORS[id], 2)
            cv2.putText(display_frame, CLASS_NAMES[id], (p1[0], p1[1]-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[id], 1)
            cv2.putText(display_frame, f"{(score):.3f}", (p1[0], p2[1]+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[id], 1)

    latency = (time.time() - start_time) * 1000
    cv2.putText(display_frame, f"{latency:.2f}ms | GPU", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

    cv2.imshow("CatClass", display_frame)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()