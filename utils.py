import cv2
import numpy as np
from config import (CLASS_COLORS, 
                    CLASS_NAMES, 
                    CONFIDENCE_THRESHOLD, 
                    NMS_IOU_THRESHOLD,
                    IMG_SIZE)


def preprocess_frame(frame):
    """
    Process cam frame for inference
    """
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # H,W,C -> C,H,W
    img = np.transpose(img, (2, 0, 1)) # Note the image is a view and non-contiguous
    # Add batch dim
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img) 


def postprocess_frame(model_outputs, frame_height, frame_width):
    """
    Filter output tensor and convert bbox coords to frame size
    """
    predictions = model_outputs[0].T # select actual output and transpose (8400, 7)

    scores = predictions[:, 4:] # grab cat scores
    max_scores = np.max(scores, axis=1) # highest confidence in anchor point 
    class_ids = np.argmax(scores, axis=1)  # which cat is it
    
    confidence_mask = max_scores > CONFIDENCE_THRESHOLD
    if np.any(confidence_mask):
        filtered_bboxes_cxcywh = predictions[confidence_mask, :4]
        filtered_scores = max_scores[confidence_mask]
        filtered_ids = class_ids[confidence_mask]

        center_x  = filtered_bboxes_cxcywh[:, 0]
        center_y = filtered_bboxes_cxcywh[:, 1]
        width = filtered_bboxes_cxcywh[:, 2]
        height = filtered_bboxes_cxcywh[:, 3]
        # to XY-XY and relative to frame
        x1 = (center_x - width / 2) * frame_width / IMG_SIZE
        y1 = (center_y - height / 2) * frame_height / IMG_SIZE
        x2 = (center_x + width / 2) * frame_width / IMG_SIZE
        y2 = (center_y + height / 2) * frame_height / IMG_SIZE
        bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) # stack as columns


        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(boxes_xywh, filtered_scores.tolist(),
                                0.0, NMS_IOU_THRESHOLD)
        if len(keep) == 0:
            return np.array([]), np.array([]), np.array([])
        
        keep = np.array(keep).flatten()

        # Pytorch version of NMS. 
        # keep = torchvision.ops.nms(
        #     torch.from_numpy(bboxes_xyxy).cuda(),
        #     torch.from_numpy(filtered_scores).cuda(),
        #     NMS_IOU_THRESHOLD
        # ).cpu().numpy()

        # Reshape incase 1 box returns shape of (4,)
        final_bboxes = bboxes_xyxy[keep].reshape(-1, 4)
        final_scores = filtered_scores[keep].reshape(-1)
        final_class_ids = filtered_ids[keep].reshape(-1)

        return final_bboxes, final_scores, final_class_ids
    return np.array([]), np.array([]), np.array([])



def draw_bboxes(frame, boxes, scores, ids):
    """
    Draws bounding boxes on cats.
    Assume data is already post-processed and ready for openCV
    """
    for box, score, id in zip(boxes, scores, ids):
        color = CLASS_COLORS[id]
        name = CLASS_NAMES[id]
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{(score):.2f}", (x2, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
