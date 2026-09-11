from pathlib import Path
from ultralytics import YOLO

MODEL = r"D:\sih\marine-sonar\runs\detect\runs\detect\runs\groupval_1024\weights\best.pt"
DATA = Path(r"D:\sih\marine-sonar\yolo_dataset_groupval\test")

model = YOLO(MODEL)

thresholds = [0.001, 0.005, 0.010, 0.020, 0.050,
              0.100, 0.150, 0.200, 0.250, 0.300,
              0.400, 0.500]

IOU_THRESHOLD = 0.5

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def load_gt(label_file, w, h):
    boxes = []

    if not label_file.exists():
        return boxes

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        boxes.append((x1, y1, x2, y2))

    return boxes


images = sorted((DATA / "images").glob("*"))

print("=" * 75)
print("GROUPVAL 1024 TEST THRESHOLD ANALYSIS")
print("=" * 75)
print("Images:", len(images))
print()

# Run prediction once at extremely low confidence.
results = model.predict(
    source=str(DATA / "images"),
    imgsz=1024,
    device=0,
    conf=0.001,
    iou=0.5,
    workers=0,
    verbose=False
)

all_data = []

for idx, result in enumerate(results):
    img_path = Path(result.path)

    h, w = result.orig_shape

    label_file = DATA / "labels" / f"{img_path.stem}.txt"

    gt = load_gt(label_file, w, h)

    preds = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, conf in zip(xyxy, confs):
            preds.append((box, float(conf)))

    all_data.append((gt, preds))

print("Predictions collected.")
print()

print("=" * 75)
print("CONF | TP | FP | PRECISION | RECALL | F1")
print("=" * 75)

best = None

for threshold in thresholds:

    total_tp = 0
    total_fp = 0
    total_gt = 0

    for gt, preds in all_data:

        selected = [
            box for box, conf in preds
            if conf >= threshold
        ]

        total_gt += len(gt)

        matched_gt = set()

        # Highest-confidence predictions first
        for pred in selected:

            best_iou = 0.0
            best_gt = -1

            for gi, gt_box in enumerate(gt):

                if gi in matched_gt:
                    continue

                score = iou(pred, gt_box)

                if score > best_iou:
                    best_iou = score
                    best_gt = gi

            if best_iou >= IOU_THRESHOLD:
                total_tp += 1
                matched_gt.add(best_gt)
            else:
                total_fp += 1

    fn = total_gt - total_tp

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    recall = (
        total_tp / total_gt
        if total_gt > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print(
        f"{threshold:0.3f} | "
        f"{total_tp:2d} | "
        f"{total_fp:3d} | "
        f"{precision:0.4f} | "
        f"{recall:0.4f} | "
        f"{f1:0.4f}"
    )

    if best is None or f1 > best["f1"]:
        best = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

print()
print("=" * 75)
print("BEST F1 THRESHOLD")
print("=" * 75)

print(f"Confidence : {best['threshold']:.3f}")
print(f"Precision  : {best['precision']:.6f}")
print(f"Recall     : {best['recall']:.6f}")
print(f"F1         : {best['f1']:.6f}")

print("=" * 75)
print("DONE")
print("=" * 75)