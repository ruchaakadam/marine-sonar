from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# CONFIGURATION
# ============================================================

GT = Path("yolo_dataset_groupval/test/labels")

# IMPORTANT:
# Use predictions generated with conf=0.001 so we can
# evaluate different thresholds afterwards.

PR = Path(r"runs/detect/runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels")

CONF = 0.05
IOU_T = 0.50


# ============================================================
# BOX UTILITIES
# ============================================================

def xywh_to_xyxy(x, y, w, h):
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ============================================================
# READ LABEL FILE
# ============================================================

def read_gt(path):
    """
    Read YOLO ground-truth labels.

    Returns:
        [(class_id, box), ...]
    """

    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        if len(v) < 5:
            continue

        cls = int(float(v[0]))

        box = xywh_to_xyxy(
            float(v[1]),
            float(v[2]),
            float(v[3]),
            float(v[4])
        )

        boxes.append((cls, box))

    return boxes


def read_predictions(path):
    """
    Read YOLO prediction labels.

    Expected format:
        class x y w h confidence

    Returns:
        [(class_id, box, confidence), ...]
    """

    predictions = []

    if not path.exists():
        return predictions

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        if len(v) < 6:
            continue

        cls = int(float(v[0]))

        box = xywh_to_xyxy(
            float(v[1]),
            float(v[2]),
            float(v[3]),
            float(v[4])
        )

        confidence = float(v[5])

        predictions.append(
            (cls, box, confidence)
        )

    return predictions


# ============================================================
# GROUP NAME
# ============================================================

def group_name(filename):
    """
    Example:

    Artificial_Reef_01
        -> Artificial_Reef

    Lucinda_van_Valkenburg_05
        -> Lucinda_van_Valkenburg
    """

    return re.sub(
        r"_\d+$",
        "",
        Path(filename).stem
    )


# ============================================================
# STATISTICS
# ============================================================

stats = defaultdict(
    lambda: {
        "images": 0,
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
)


# ============================================================
# MAIN EVALUATION
# ============================================================

for gt_file in sorted(GT.glob("*.txt")):

    group = group_name(gt_file.name)

    stats[group]["images"] += 1

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_boxes = read_gt(gt_file)

    stats[group]["gt"] += len(gt_boxes)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred_file = PR / gt_file.name

    predictions = read_predictions(pred_file)

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    predictions = [
        p for p in predictions
        if p[2] >= CONF
    ]

    # Highest confidence first
    predictions.sort(
        key=lambda x: x[2],
        reverse=True
    )

    # --------------------------------------------------------
    # Match predictions to GT
    # --------------------------------------------------------

    matched = set()

    tp = 0
    fp = 0

    for pred_class, pred_box, confidence in predictions:

        best_iou = 0.0
        best_index = -1

        for i, (gt_class, gt_box) in enumerate(gt_boxes):

            # Already matched GT
            if i in matched:
                continue

            # IMPORTANT:
            # Prediction and GT must have same class
            if pred_class != gt_class:
                continue

            score = iou(
                pred_box,
                gt_box
            )

            if score > best_iou:
                best_iou = score
                best_index = i

        # ----------------------------------------------------
        # True positive
        # ----------------------------------------------------

        if best_iou >= IOU_T:

            tp += 1
            matched.add(best_index)

        # ----------------------------------------------------
        # False positive
        # ----------------------------------------------------

        else:

            fp += 1

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    fn = len(gt_boxes) - len(matched)

    stats[group]["tp"] += tp
    stats[group]["fp"] += fp
    stats[group]["fn"] += fn


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 110)
print("ORIGINAL MODEL - GROUP ERROR ANALYSIS")
print("=" * 110)

print(f"Prediction folder : {PR}")
print(f"Confidence        : {CONF}")
print(f"IoU threshold     : {IOU_T}")

print("=" * 110)

print(
    f"{'GROUP':28} "
    f"{'IMG':>5} "
    f"{'GT':>5} "
    f"{'TP':>5} "
    f"{'FP':>5} "
    f"{'FN':>5} "
    f"{'PREC':>9} "
    f"{'RECALL':>9} "
    f"{'F1':>9}"
)

print("-" * 110)


# ============================================================
# TOTALS
# ============================================================

total_images = 0
total_gt = 0
total_tp = 0
total_fp = 0
total_fn = 0


# ============================================================
# PER-GROUP RESULTS
# ============================================================

for group in sorted(stats):

    s = stats[group]

    images = s["images"]
    gt = s["gt"]
    tp = s["tp"]
    fp = s["fp"]
    fn = s["fn"]

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"{group:28} "
        f"{images:5d} "
        f"{gt:5d} "
        f"{tp:5d} "
        f"{fp:5d} "
        f"{fn:5d} "
        f"{precision:9.3f} "
        f"{recall:9.3f} "
        f"{f1:9.3f}"
    )

    total_images += images
    total_gt += gt
    total_tp += tp
    total_fp += fp
    total_fn += fn


# ============================================================
# OVERALL METRICS
# ============================================================

precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0.0
)

recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ============================================================
# TOTAL
# ============================================================

print("-" * 110)

print(
    f"{'TOTAL':28} "
    f"{total_images:5d} "
    f"{total_gt:5d} "
    f"{total_tp:5d} "
    f"{total_fp:5d} "
    f"{total_fn:5d} "
    f"{precision:9.3f} "
    f"{recall:9.3f} "
    f"{f1:9.3f}"
)

print("=" * 110)

print()
print("FINAL METRICS")
print("=" * 50)
print(f"Confidence : {CONF:.3f}")
print(f"IoU        : {IOU_T:.2f}")
print(f"TP         : {total_tp}")
print(f"FP         : {total_fp}")
print(f"FN         : {total_fn}")
print(f"Precision  : {precision:.6f}")
print(f"Recall     : {recall:.6f}")
print(f"F1         : {f1:.6f}")
print("=" * 50)