from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
MODEL = PROJECT_ROOT / "runs" / "detect" / "marine_reproducible" / "weights" / "best.pt"

print("Dataset:", DATA_YAML)
print("Model:", MODEL)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

if not MODEL.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL}\n"
        "Train the model first or change MODEL to an existing best.pt."
    )

model = YOLO(str(MODEL))

results = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=416,
    batch=1,
    device="mps",
    workers=0,
    plots=True,
)

print("\nValidation complete.")
print("mAP50:", float(results.box.map50))
print("mAP50-95:", float(results.box.map))
