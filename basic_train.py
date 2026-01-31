from ultralytics import YOLO
import wandb

wandb.init(project="catclass",
           name="yolov8n-basic-run",
)

model = YOLO("yolov8n.pt")

results = model.train(
    data="configs/dataset.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
)

wandb.finish()