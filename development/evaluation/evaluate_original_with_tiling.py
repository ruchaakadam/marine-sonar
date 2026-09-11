from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import math


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"D:\sih\marine-sonar\runs\detect\runs\detect\runs"
    r"\tile_balanced_aug_1024\weights\best.pt"
)

IMAGE_DIR = Path(r"D:\sih\marine-sonar\yolo_dataset\test\images")
LABEL_DIR = Path(r"D:\sih\marine-sonar\yolo_dataset\test\labels")

# Same tiling geometry used to create the training tiles
TILE_SIZE = 1728
OVERLAP = 432
STRIDE = TILE_SIZE - OVERLAP

# Model inference resolution
IMGSZ = 1024

# Generate predictions at very low confidence.
# We will calculate metrics at multiple thresholds later.
PRED_CONF = 0.001

IOU_MATCH = 0.50

# NMS used after mapping tile predictions back to original image
NMS_IOU = 0.50

DEVICE = 0

# Confidence thresholds to report
CONF_THRESHOLDS = [
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
    0.400,
    0.500,
]


# ============================================================
# HELPERS
# ============================================================

def load_labels(label_path, width, height):
    """
    Load YOLO labels and convert them to absolute xyxy coordinates.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append({
            "cls": cls,
            "box": [x1, y1, x2, y2],
        })

    return boxes


def make_starts(length, tile_size, stride):
    """
    Generate tile start positions while guaranteeing that the
    final tile reaches the image edge.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size

    if starts[-1] != last:
        starts.append(last)

    return starts


def box_iou(a, b):
    """
    IoU between two xyxy boxes.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def nms(predictions, iou_threshold=0.50):
    """
    Simple class-aware NMS.
    predictions:
        [{"cls": 0, "conf": 0.5, "box": [...]}, ...]
    """
    if not predictions:
        return []

    final = []

    classes = sorted(set(p["cls"] for p in predictions))

    for cls in classes:
        cls_preds = [
            p for p in predictions
            if p["cls"] == cls
        ]

        cls_preds.sort(
            key=lambda x: x["conf"],
            reverse=True
        )

        while cls_preds:
            best = cls_preds.pop(0)
            final.append(best)

            remaining = []

            for p in cls_preds:
                if box_iou(best["box"], p["box"]) < iou_threshold:
                    remaining.append(p)

            cls_preds = remaining

    final.sort(
        key=lambda x: x["conf"],
        reverse=True
    )

    return final


def match_detections(predictions, ground_truths, conf_threshold, iou_threshold):
    """
    Calculate TP / FP / FN for one image.
    """
    preds = [
        p for p in predictions
        if p["conf"] >= conf_threshold
    ]

    preds.sort(
        key=lambda x: x["conf"],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred in preds:
        best_iou = 0.0
        best_gt = None

        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue

            if pred["cls"] != gt["cls"]:
                continue

            iou = box_iou(
                pred["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


def calculate_ap(all_predictions, all_ground_truths, iou_threshold):
    """
    Calculate AP using confidence-ranked detections.
    """
    detections = []

    total_gt = 0

    for image_id, gts in all_ground_truths.items():
        total_gt += len(gts)

    for image_id, preds in all_predictions.items():
        for pred in preds:
            detections.append(
                (
                    pred["conf"],
                    image_id,
                    pred
                )
            )

    detections.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if total_gt == 0:
        return 0.0

    matched = {
        image_id: set()
        for image_id in all_ground_truths
    }

    tp_list = []
    fp_list = []

    for conf, image_id, pred in detections:

        gts = all_ground_truths.get(
            image_id,
            []
        )

        best_iou = 0.0
        best_gt = None

        for i, gt in enumerate(gts):

            if i in matched[image_id]:
                continue

            if pred["cls"] != gt["cls"]:
                continue

            iou = box_iou(
                pred["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt is not None and best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched[image_id].add(best_gt)
        else:
            tp_list.append(0)
            fp_list.append(1)

    if not tp_list:
        return 0.0

    tp_cum = []
    fp_cum = []

    t = 0
    f = 0

    for tp, fp in zip(tp_list, fp_list):
        t += tp
        f += fp

        tp_cum.append(t)
        fp_cum.append(f)

    precisions = []
    recalls = []

    for t, f in zip(tp_cum, fp_cum):
        precision = t / max(t + f, 1)
        recall = t / max(total_gt, 1)

        precisions.append(precision)
        recalls.append(recall)

    # Precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(
            precisions[i],
            precisions[i + 1]
        )

    # 101-point interpolation
    ap = 0.0

    for r in [i / 100 for i in range(101)]:

        possible = [
            p
            for p, rec in zip(
                precisions,
                recalls
            )
            if rec >= r
        ]

        if possible:
            ap += max(possible)

    ap /= 101.0

    return ap


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("TILED ORIGINAL-IMAGE EVALUATION")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Images:")
print(IMAGE_DIR)

print()
print("Labels:")
print(LABEL_DIR)

print()
print("Tile size:", TILE_SIZE)
print("Overlap:", OVERLAP)
print("Stride:", STRIDE)
print("Inference imgsz:", IMGSZ)

model = YOLO(str(MODEL_PATH))


# ============================================================
# PROCESS DATASET
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print()
print("Images found:", len(image_paths))

if not image_paths:
    raise RuntimeError(
        "No test images found."
    )


all_predictions = {}
all_ground_truths = {}

total_tiles = 0


for image_number, image_path in enumerate(image_paths, 1):

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    ground_truths = load_labels(
        label_path,
        width,
        height
    )

    all_ground_truths[image_path.stem] = ground_truths

    x_starts = make_starts(
        width,
        TILE_SIZE,
        STRIDE
    )

    y_starts = make_starts(
        height,
        TILE_SIZE,
        STRIDE
    )

    image_predictions = []

    for y0 in y_starts:

        for x0 in x_starts:

            x1 = min(
                x0 + TILE_SIZE,
                width
            )

            y1 = min(
                y0 + TILE_SIZE,
                height
            )

            tile = image.crop(
                (x0, y0, x1, y1)
            )

            results = model.predict(
                source=tile,
                imgsz=IMGSZ,
                conf=PRED_CONF,
                iou=0.50,
                device=DEVICE,
                verbose=False
            )

            total_tiles += 1

            result = results[0]

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()

            tile_w, tile_h = tile.size

            for box, conf, cls in zip(
                boxes,
                confs,
                classes
            ):

                tx1, ty1, tx2, ty2 = box

                # Convert tile coordinates back
                # into original-image coordinates.
                ox1 = tx1 + x0
                oy1 = ty1 + y0
                ox2 = tx2 + x0
                oy2 = ty2 + y0

                # Clip to original image.
                ox1 = max(0.0, min(float(width), ox1))
                oy1 = max(0.0, min(float(height), oy1))
                ox2 = max(0.0, min(float(width), ox2))
                oy2 = max(0.0, min(float(height), oy2))

                if ox2 <= ox1 or oy2 <= oy1:
                    continue

                image_predictions.append({
                    "cls": int(cls),
                    "conf": float(conf),
                    "box": [
                        ox1,
                        oy1,
                        ox2,
                        oy2
                    ]
                })

    # Merge overlapping predictions from neighboring tiles.
    image_predictions = nms(
        image_predictions,
        NMS_IOU
    )

    all_predictions[image_path.stem] = image_predictions

    print(
        f"{image_number:3d}/{len(image_paths)} "
        f"{image_path.stem:30s} "
        f"GT={len(ground_truths):2d} "
        f"PRED={len(image_predictions):3d}"
    )


# ============================================================
# SUMMARY
# ============================================================

total_gt = sum(
    len(x)
    for x in all_ground_truths.values()
)

total_predictions = sum(
    len(x)
    for x in all_predictions.values()
)

print()
print("=" * 70)
print("TILED INFERENCE SUMMARY")
print("=" * 70)

print("Images:", len(image_paths))
print("Tiles processed:", total_tiles)
print("GT boxes:", total_gt)
print("Predictions after NMS:", total_predictions)


# ============================================================
# CONFIDENCE TABLE
# ============================================================

print()
print("=" * 90)
print(
    "CONF | PRED | TP | FP | FN | PRECISION | RECALL | F1"
)
print("-" * 90)

best_f1 = -1
best_threshold = None
best_p = 0
best_r = 0

for threshold in CONF_THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_id in all_ground_truths:

        tp, fp, fn = match_detections(
            all_predictions.get(
                image_id,
                []
            ),
            all_ground_truths[image_id],
            threshold,
            IOU_MATCH
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(
        total_tp + total_fp,
        1
    )

    recall = total_tp / max(
        total_tp + total_fn,
        1
    )

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    pred_count = sum(
        sum(
            1
            for p in all_predictions.get(
                image_id,
                []
            )
            if p["conf"] >= threshold
        )
        for image_id in all_ground_truths
    )

    print(
        f"{threshold:0.3f} | "
        f"{pred_count:4d} | "
        f"{total_tp:2d} | "
        f"{total_fp:3d} | "
        f"{total_fn:2d} | "
        f"{precision:0.4f} | "
        f"{recall:0.4f} | "
        f"{f1:0.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_p = precision
        best_r = recall


# ============================================================
# AP / mAP
# ============================================================

print()
print("=" * 70)
print("AP RESULTS")
print("=" * 70)

aps = []

for iou in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]:

    ap = calculate_ap(
        all_predictions,
        all_ground_truths,
        iou
    )

    aps.append(ap)

    print(
        f"IoU {iou:.2f}: AP={ap:.6f}"
    )


map50 = aps[0]
map50_95 = sum(aps) / len(aps)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FINAL TILED ORIGINAL TEST")
print("=" * 70)

print(
    f"mAP50    : {map50:.6f}"
)

print(
    f"mAP50-95 : {map50_95:.6f}"
)

print()
print("BEST F1 THRESHOLD")
print(
    f"Confidence : {best_threshold:.3f}"
)
print(
    f"Precision  : {best_p:.6f}"
)
print(
    f"Recall     : {best_r:.6f}"
)
print(
    f"F1         : {best_f1:.6f}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)