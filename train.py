from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/dataset.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    mosaic=0.0,
    project="runs",
    name="yolov8n-bodies",
)
