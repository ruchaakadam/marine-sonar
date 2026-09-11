from ultralytics import YOLO

# Load pretrained YOLO11n
model = YOLO("yolo11n.pt")

# Train
model.train(
    data="yolo_dataset_groupval/data.yaml",

    imgsz=1024,
    epochs=150,
    batch=4,

    device=0,
    workers=0,

    patience=40,

    # Augmentation
    degrees=5,
    translate=0.05,
    scale=0.20,
    fliplr=0.5,

    project="runs/detect/runs",
    name="groupval_1024",
    exist_ok=True,
)