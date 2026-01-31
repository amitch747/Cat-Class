from config import CLASS_COLORS, CLASS_NAMES
import cv2


def draw_bboxes(frame, boxes, scores, ids):
    """
    Draws bounding boxes on cats.
    Assume data is already post-processed and ready for openCV
    

    :param boxes: Description
    :param scores: Description
    :param ids: Description
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
