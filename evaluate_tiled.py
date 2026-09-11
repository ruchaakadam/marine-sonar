from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
import torchvision
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

MODEL = Path(
    "runs/detect/runs/detect/runs/tile_balanced_aug/weights/best.pt"
)

TEST_IMAGES = Path("yolo_dataset/test/images")
TEST_LABELS = Path("yolo_dataset/test/labels")

TILE_H = 1728
OVERLAP = 432
STRIDE = TILE_H - OVERLAP

IMGSZ = 640
PRED_CONF = 0.001
IOU_NMS = 0.50
DEVICE = 0

CONF_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

model = YOLO(str(MODEL))


# ============================================================
# IOU
# ============================================================

def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores):

    if not boxes:
        return []

    bt = torch.tensor(
        boxes,
        dtype=torch.float32
    )

    st = torch.tensor(
        scores,
        dtype=torch.float32
    )

    keep = torchvision.ops.nms(
        bt,
        st,
        IOU_NMS
    )

    return keep.tolist()


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_gt(label_path, W, H):

    gt = []

    if not label_path.exists():
        return gt

    for line in label_path.read_text().splitlines():

        if not line.strip():
            continue

        p = line.split()

        if len(p) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, p)

        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H

        gt.append(
            [x1, y1, x2, y2]
        )

    return gt


# ============================================================
# TILED PREDICTION
# ============================================================

def predict_tiled(img_path):

    img = Image.open(img_path).convert("RGB")

    W, H = img.size

    boxes = []
    scores = []

    y0 = 0

    while y0 < H:

        if H <= TILE_H:
            y_start = 0
        elif y0 + TILE_H >= H:
            y_start = H - TILE_H
        else:
            y_start = y0

        y_end = min(
            y_start + TILE_H,
            H
        )

        tile = img.crop(
            (0, y_start, W, y_end)
        )

        results = model.predict(
            source=tile,
            imgsz=IMGSZ,
            conf=PRED_CONF,
            iou=0.5,
            device=DEVICE,
            verbose=False
        )

        r = results[0]

        if r.boxes is not None:

            for b, s in zip(
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.conf.cpu().tolist()
            ):

                x1, ty1, x2, ty2 = b

                boxes.append(
                    [
                        x1,
                        ty1 + y_start,
                        x2,
                        ty2 + y_start
                    ]
                )

                scores.append(
                    float(s)
                )

        if y_end >= H:
            break

        y0 += STRIDE

    keep = nms(
        boxes,
        scores
    )

    return [
        (boxes[i], scores[i])
        for i in keep
    ]


# ============================================================
# MATCH
# ============================================================

def evaluate(
    all_data,
    confidence,
    iou_threshold=0.50
):

    total_gt = 0
    total_predictions = 0

    tp = 0
    fp = 0

    for item in all_data:

        gt = item["gt"]

        preds = [
            p for p in item["preds"]
            if p[1] >= confidence
        ]

        total_gt += len(gt)
        total_predictions += len(preds)

        used = set()

        # highest confidence first
        preds.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for box, score in preds:

            best_iou = 0.0
            best_idx = -1

            for i, g in enumerate(gt):

                if i in used:
                    continue

                v = box_iou(
                    box,
                    g
                )

                if v > best_iou:
                    best_iou = v
                    best_idx = i

            if best_iou >= iou_threshold:

                tp += 1
                used.add(best_idx)

            else:

                fp += 1

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / total_gt
        if total_gt > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return (
        total_predictions,
        tp,
        fp,
        precision,
        recall,
        f1
    )


# ============================================================
# RUN
# ============================================================

print()
print("=" * 70)
print("TILED CONFIDENCE SWEEP")
print("=" * 70)

print("MODEL:", MODEL)
print("TILE:", TILE_H)
print("OVERLAP:", OVERLAP)
print("STRIDE:", STRIDE)
print()

images = sorted(
    TEST_IMAGES.glob("*.png")
)

print(
    "TEST IMAGES:",
    len(images)
)

all_data = []

for n, img_path in enumerate(
    images,
    1
):

    img = Image.open(img_path)

    W, H = img.size

    label_path = (
        TEST_LABELS /
        f"{img_path.stem}.txt"
    )

    gt = load_gt(
        label_path,
        W,
        H
    )

    preds = predict_tiled(
        img_path
    )

    all_data.append(
        {
            "name": img_path.name,
            "gt": gt,
            "preds": preds
        }
    )

    print(
        f"{n:3d}/{len(images)} "
        f"{img_path.stem:35s} "
        f"GT={len(gt):2d} "
        f"PRED={len(preds):3d}"
    )


# ============================================================
# CONFIDENCE RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "CONF | PRED | TP | FP | PRECISION | RECALL | F1"
)
print("-" * 70)

best = None

for conf in CONF_THRESHOLDS:

    (
        total,
        tp,
        fp,
        precision,
        recall,
        f1
    ) = evaluate(
        all_data,
        conf,
        0.50
    )

    print(
        f"{conf:5.3f} | "
        f"{total:4d} | "
        f"{tp:2d} | "
        f"{fp:4d} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

    if best is None or f1 > best[1]:

        best = (
            conf,
            f1,
            precision,
            recall
        )


print()
print("=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Confidence : {best[0]:.3f}"
)

print(
    f"Precision  : {best[2]:.6f}"
)

print(
    f"Recall     : {best[3]:.6f}"
)

print(
    f"F1         : {best[1]:.6f}"
)

print("=" * 70)