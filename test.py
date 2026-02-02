from ultralytics import YOLO

model = YOLO("models/best.pt")

metrics = model.val(data="configs/dataset.yaml", split="test")

print(f"Test mAP50: {metrics.box.map50}")
print(f"Test mAP50-95: {metrics.box.map}")