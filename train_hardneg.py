from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data_hardneg.yaml",

    imgsz=640,

    epochs=150,

    batch=8,

    device=0,

    workers=0,

    patience=40,

    project="runs/detect",
    name="tile_hardneg_train",

    exist_ok=True,

    pretrained=True,

    optimizer="auto",

    verbose=True
)