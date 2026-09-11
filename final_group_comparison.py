from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# DATA
# ============================================================

GT = Path("yolo_dataset_groupval/test/labels")

ORIGINAL_PR = Path(
    r"runs/detect/runs/group_metrics/groupval_1024_conf010_txt/labels"
)

ONLINE_PR = Path(
    r"runs/detect/runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels"
)

# Best thresholds found from threshold analysis
ORIGINAL_CONF = 0.100
ONLINE_CONF = 0.050

IOU_T = 0.50


# ============================================================
# FUNCTIONS
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

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def group_name(filename):

    return re.sub(r"_\d+$", "", Path(filename).stem)


def read_gt(path):

    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        if len(v) < 5:
            continue

        boxes.append(
            xywh_to_xyxy(
                float(v[1]),
                float(v[2]),
                float(v[3]),
                float(v[4])
            )
        )

    return boxes


def read_predictions(path, confidence_threshold):

    predictions = []

    if not path.exists():
        return predictions

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        if len(v) < 6:
            continue

        confidence = float(v[5])

        if confidence < confidence_threshold:
            continue

        box = xywh_to_xyxy(
            float(v[1]),
            float(v[2]),
            float(v[3]),
            float(v[4])
        )

        predictions.append((box, confidence))

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions


def calculate_model(prediction_folder, confidence):

    stats = defaultdict(
        lambda: {
            "img": 0,
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    for gt_file in sorted(GT.glob("*.txt")):

        group = group_name(gt_file.name)

        gt_boxes = read_gt(gt_file)

        pred_file = prediction_folder / gt_file.name

        predictions = read_predictions(
            pred_file,
            confidence
        )

        stats[group]["img"] += 1
        stats[group]["gt"] += len(gt_boxes)

        matched = set()

        tp = 0
        fp = 0

        for pred_box, pred_conf in predictions:

            best_iou = 0.0
            best_index = -1

            for i, gt_box in enumerate(gt_boxes):

                if i in matched:
                    continue

                score = iou(pred_box, gt_box)

                if score > best_iou:
                    best_iou = score
                    best_index = i

            if best_iou >= IOU_T:

                tp += 1
                matched.add(best_index)

            else:

                fp += 1

        fn = len(gt_boxes) - len(matched)

        stats[group]["tp"] += tp
        stats[group]["fp"] += fp
        stats[group]["fn"] += fn

    return stats


def metrics(s):

    tp = s["tp"]
    fp = s["fp"]
    fn = s["fn"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f1


# ============================================================
# RUN
# ============================================================

original = calculate_model(
    ORIGINAL_PR,
    ORIGINAL_CONF
)

online = calculate_model(
    ONLINE_PR,
    ONLINE_CONF
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 125)
print("FINAL PER-GROUP COMPARISON")
print("=" * 125)

print(f"Original confidence      : {ORIGINAL_CONF:.3f}")
print(f"Online augmentation conf : {ONLINE_CONF:.3f}")
print(f"IoU threshold             : {IOU_T:.2f}")

print()
print(
    f"{'GROUP':28}"
    f"{'ORIG F1':>10}"
    f"{'AUG F1':>10}"
    f"{'CHANGE':>10}"
    f"{'ORIG R':>10}"
    f"{'AUG R':>10}"
    f"{'ORIG P':>10}"
    f"{'AUG P':>10}"
)

print("-" * 125)


# ============================================================
# GROUP RESULTS
# ============================================================

for group in sorted(original.keys()):

    op, or_, of1 = metrics(original[group])
    ap, ar, af1 = metrics(online[group])

    f1_change = af1 - of1

    print(
        f"{group:28}"
        f"{of1:10.3f}"
        f"{af1:10.3f}"
        f"{f1_change:+10.3f}"
        f"{or_:10.3f}"
        f"{ar:10.3f}"
        f"{op:10.3f}"
        f"{ap:10.3f}"
    )


# ============================================================
# TOTALS
# ============================================================

def total_stats(stats):

    total = {
        "img": 0,
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }

    for s in stats.values():

        for key in total:

            total[key] += s[key]

    return total


orig_total = total_stats(original)
aug_total = total_stats(online)

op, or_, of1 = metrics(orig_total)
ap, ar, af1 = metrics(aug_total)


print()
print("=" * 125)
print("TOTAL COMPARISON")
print("=" * 125)

print()
print(
    f"{'METRIC':20}"
    f"{'ORIGINAL':>15}"
    f"{'ONLINE AUG':>15}"
    f"{'CHANGE':>15}"
)

print("-" * 70)

for key in ["tp", "fp", "fn"]:

    old = orig_total[key]
    new = aug_total[key]

    print(
        f"{key.upper():20}"
        f"{old:15d}"
        f"{new:15d}"
        f"{new - old:+15d}"
    )

print()

for name, old, new in [
    ("Precision", op, ap),
    ("Recall", or_, ar),
    ("F1", of1, af1)
]:

    print(
        f"{name:20}"
        f"{old:15.6f}"
        f"{new:15.6f}"
        f"{new - old:+15.6f}"
    )


# ============================================================
# GROUP IMPROVEMENT SUMMARY
# ============================================================

print()
print("=" * 125)
print("GROUP IMPROVEMENT SUMMARY")
print("=" * 125)

improved = []
worsened = []
unchanged = []

for group in sorted(original.keys()):

    _, _, old_f1 = metrics(original[group])
    _, _, new_f1 = metrics(online[group])

    if new_f1 > old_f1:
        improved.append((group, old_f1, new_f1))

    elif new_f1 < old_f1:
        worsened.append((group, old_f1, new_f1))

    else:
        unchanged.append((group, old_f1))


print()
print("IMPROVED GROUPS")
print("-" * 60)

for group, old, new in improved:

    print(
        f"{group:28} "
        f"{old:.3f} -> {new:.3f} "
        f"({new-old:+.3f})"
    )


print()
print("WORSENED GROUPS")
print("-" * 60)

for group, old, new in worsened:

    print(
        f"{group:28} "
        f"{old:.3f} -> {new:.3f} "
        f"({new-old:+.3f})"
    )


print()
print("UNCHANGED GROUPS")
print("-" * 60)

for group, value in unchanged:

    print(
        f"{group:28} "
        f"F1 = {value:.3f}"
    )


print()
print("=" * 125)
print("ANALYSIS COMPLETE")
print("=" * 125)