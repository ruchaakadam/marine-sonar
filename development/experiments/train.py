from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
MODEL = PROJECT_ROOT / "yolo11n.pt"

print("Dataset:", DATA_YAML)
print("Base model:", MODEL)

model = YOLO(str(MODEL))

model.train(
    data=str(DATA_YAML),
    epochs=20,
    imgsz=416,
    batch=1,
    device="mps",
    workers=0,
    project=str(PROJECT_ROOT / "runs" / "detect"),
    name="marine_reproducible",
)

print("Training complete.")
