from ultralytics import YOLO

# Load pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train at higher resolution
model.train(
    data="data_tiles_balanced.yaml",
    imgsz=1024,
    epochs=150,
    batch=2,
    device=0,
    workers=0,
    patience=40,
    project="runs/detect/runs",
    name="tile_balanced_aug_1024",
    exist_ok=True
)