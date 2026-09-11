from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# CONFIGURATION
# ============================================================

GT_DIR = Path(
    r"yolo_dataset_groupval\test\labels"
)

ORIGINAL_DIR = Path(
    r"runs\FINAL_RESULTS\original_same_threshold_conf001_txt\labels"
)

ONLINE_AUG_DIR = Path(
    r"runs\detect\runs\FINAL_RESULTS\onlineaug_predictions_conf001_txt\labels"
)

CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.50


# ============================================================
# BOX FUNCTIONS
# ============================================================

def xywh_to_xyxy(x, y, w, h):
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    )


def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GROUP NAME
# ============================================================

def get_group_name(filename):

    stem = Path(filename).stem

    # Example:
    # Artificial_Reef_01 -> Artificial_Reef
    # Lucinda_van_Valkenburg_05 -> Lucinda_van_Valkenburg
    # Monrovia_08 -> Monrovia

    group = re.sub(r"_\d+$", "", stem)

    return group


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_gt_file(file_path):

    boxes = []

    if not file_path.exists():
        return boxes

    for line in file_path.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) < 5:
            continue

        try:

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

            boxes.append(
                xywh_to_xyxy(x, y, w, h)
            )

        except ValueError:
            continue

    return boxes


# ============================================================
# READ PREDICTIONS
# ============================================================

def read_prediction_file(file_path):

    predictions = []

    if not file_path.exists():
        return predictions

    for line in file_path.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) < 6:
            continue

        try:

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])
            confidence = float(values[5])

            # Apply confidence threshold
            if confidence < CONF_THRESHOLD:
                continue

            box = xywh_to_xyxy(
                x,
                y,
                w,
                h
            )

            predictions.append(
                (box, confidence)
            )

        except ValueError:
            continue

    # Highest confidence first
    predictions.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return predictions


# ============================================================
# MATCH PREDICTIONS TO GT
# ============================================================

def calculate_detection_stats(gt_boxes, predictions):

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_box, confidence in predictions:

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            current_iou = calculate_iou(
                pred_box,
                gt_box
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_gt_index = gt_index

        if best_iou >= IOU_THRESHOLD:

            tp += 1
            matched_gt.add(best_gt_index)

        else:

            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, fp, fn):

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

    return precision, recall, f1


# ============================================================
# CHECK PATHS
# ============================================================

print()
print("=" * 120)
print("DETAILED GROUP ERROR ANALYSIS")
print("=" * 120)

print()
print(f"GT folder       : {GT_DIR}")
print(f"Original folder : {ORIGINAL_DIR}")
print(f"Online AUG      : {ONLINE_AUG_DIR}")
print(f"Confidence      : {CONF_THRESHOLD:.3f}")
print(f"IoU threshold   : {IOU_THRESHOLD:.2f}")

print()
print("-" * 120)
print("PATH CHECK")
print("-" * 120)

print(f"GT exists       : {GT_DIR.exists()}")
print(f"Original exists : {ORIGINAL_DIR.exists()}")
print(f"Online exists   : {ONLINE_AUG_DIR.exists()}")


if not GT_DIR.exists():

    print()
    print("ERROR: GT folder does not exist.")
    raise SystemExit


if not ORIGINAL_DIR.exists():

    print()
    print("ERROR: Original prediction folder does not exist.")
    raise SystemExit


if not ONLINE_AUG_DIR.exists():

    print()
    print("ERROR: Online AUG prediction folder does not exist.")
    raise SystemExit


# ============================================================
# FILE COUNTS
# ============================================================

gt_files = sorted(GT_DIR.glob("*.txt"))

original_files = sorted(
    ORIGINAL_DIR.glob("*.txt")
)

online_files = sorted(
    ONLINE_AUG_DIR.glob("*.txt")
)

print()
print("-" * 120)
print("FILE COUNTS")
print("-" * 120)

print(f"GT files       : {len(gt_files)}")
print(f"Original files : {len(original_files)}")
print(f"Online AUG     : {len(online_files)}")


# ============================================================
# GROUP STATISTICS
# ============================================================

group_stats = defaultdict(
    lambda: {
        "img": 0,
        "gt": 0,

        "orig_tp": 0,
        "orig_fp": 0,
        "orig_fn": 0,

        "aug_tp": 0,
        "aug_fp": 0,
        "aug_fn": 0,
    }
)


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

for gt_file in gt_files:

    filename = gt_file.name

    group = get_group_name(filename)

    gt_boxes = read_gt_file(gt_file)

    original_file = ORIGINAL_DIR / filename
    online_file = ONLINE_AUG_DIR / filename

    original_predictions = read_prediction_file(
        original_file
    )

    online_predictions = read_prediction_file(
        online_file
    )

    orig_tp, orig_fp, orig_fn = calculate_detection_stats(
        gt_boxes,
        original_predictions
    )

    aug_tp, aug_fp, aug_fn = calculate_detection_stats(
        gt_boxes,
        online_predictions
    )

    group_stats[group]["img"] += 1
    group_stats[group]["gt"] += len(gt_boxes)

    group_stats[group]["orig_tp"] += orig_tp
    group_stats[group]["orig_fp"] += orig_fp
    group_stats[group]["orig_fn"] += orig_fn

    group_stats[group]["aug_tp"] += aug_tp
    group_stats[group]["aug_fp"] += aug_fp
    group_stats[group]["aug_fn"] += aug_fn


# ============================================================
# MAIN GROUP TABLE
# ============================================================

print()
print("=" * 150)
print("PER-GROUP ORIGINAL VS ONLINE AUGMENTATION")
print("=" * 150)

print()
print(
    f"{'GROUP':28} "
    f"{'GT':>4} "
    f"{'O_TP':>5} "
    f"{'O_FP':>5} "
    f"{'O_FN':>5} "
    f"{'O_F1':>8} "
    f"{'A_TP':>5} "
    f"{'A_FP':>5} "
    f"{'A_FN':>5} "
    f"{'A_F1':>8} "
    f"{'ΔF1':>9}"
)

print("-" * 150)


# Store changes for later ranking
improvements = []
worsened = []
unchanged = []


for group in sorted(group_stats):

    s = group_stats[group]

    orig_precision, orig_recall, orig_f1 = calculate_metrics(
        s["orig_tp"],
        s["orig_fp"],
        s["orig_fn"]
    )

    aug_precision, aug_recall, aug_f1 = calculate_metrics(
        s["aug_tp"],
        s["aug_fp"],
        s["aug_fn"]
    )

    change = aug_f1 - orig_f1

    print(
        f"{group:28} "
        f"{s['gt']:4d} "
        f"{s['orig_tp']:5d} "
        f"{s['orig_fp']:5d} "
        f"{s['orig_fn']:5d} "
        f"{orig_f1:8.3f} "
        f"{s['aug_tp']:5d} "
        f"{s['aug_fp']:5d} "
        f"{s['aug_fn']:5d} "
        f"{aug_f1:8.3f} "
        f"{change:+9.3f}"
    )

    record = (
        group,
        orig_f1,
        aug_f1,
        change
    )

    if change > 0.000001:
        improvements.append(record)

    elif change < -0.000001:
        worsened.append(record)

    else:
        unchanged.append(record)


# ============================================================
# TOTALS
# ============================================================

total_orig_tp = sum(
    s["orig_tp"]
    for s in group_stats.values()
)

total_orig_fp = sum(
    s["orig_fp"]
    for s in group_stats.values()
)

total_orig_fn = sum(
    s["orig_fn"]
    for s in group_stats.values()
)

total_aug_tp = sum(
    s["aug_tp"]
    for s in group_stats.values()
)

total_aug_fp = sum(
    s["aug_fp"]
    for s in group_stats.values()
)

total_aug_fn = sum(
    s["aug_fn"]
    for s in group_stats.values()
)


orig_p, orig_r, orig_f1 = calculate_metrics(
    total_orig_tp,
    total_orig_fp,
    total_orig_fn
)

aug_p, aug_r, aug_f1 = calculate_metrics(
    total_aug_tp,
    total_aug_fp,
    total_aug_fn
)


print()
print("-" * 150)

print(
    f"{'TOTAL':28} "
    f"{sum(s['gt'] for s in group_stats.values()):4d} "
    f"{total_orig_tp:5d} "
    f"{total_orig_fp:5d} "
    f"{total_orig_fn:5d} "
    f"{orig_f1:8.3f} "
    f"{total_aug_tp:5d} "
    f"{total_aug_fp:5d} "
    f"{total_aug_fn:5d} "
    f"{aug_f1:8.3f} "
    f"{aug_f1-orig_f1:+9.3f}"
)


# ============================================================
# METRIC COMPARISON
# ============================================================

print()
print("=" * 100)
print("OVERALL METRIC COMPARISON")
print("=" * 100)

print()
print(
    f"{'METRIC':15} "
    f"{'ORIGINAL':>15} "
    f"{'ONLINE AUG':>15} "
    f"{'CHANGE':>15}"
)

print("-" * 100)

print(
    f"{'TP':15} "
    f"{total_orig_tp:15d} "
    f"{total_aug_tp:15d} "
    f"{total_aug_tp-total_orig_tp:+15d}"
)

print(
    f"{'FP':15} "
    f"{total_orig_fp:15d} "
    f"{total_aug_fp:15d} "
    f"{total_aug_fp-total_orig_fp:+15d}"
)

print(
    f"{'FN':15} "
    f"{total_orig_fn:15d} "
    f"{total_aug_fn:15d} "
    f"{total_aug_fn-total_orig_fn:+15d}"
)

print()

print(
    f"{'Precision':15} "
    f"{orig_p:15.6f} "
    f"{aug_p:15.6f} "
    f"{aug_p-orig_p:+15.6f}"
)

print(
    f"{'Recall':15} "
    f"{orig_r:15.6f} "
    f"{aug_r:15.6f} "
    f"{aug_r-orig_r:+15.6f}"
)

print(
    f"{'F1':15} "
    f"{orig_f1:15.6f} "
    f"{aug_f1:15.6f} "
    f"{aug_f1-orig_f1:+15.6f}"
)


# ============================================================
# IMPROVED GROUPS
# ============================================================

print()
print("=" * 100)
print("GROUP IMPROVEMENT SUMMARY")
print("=" * 100)

print()
print("IMPROVED GROUPS")
print("-" * 80)

if improvements:

    improvements.sort(
        key=lambda x: x[3],
        reverse=True
    )

    for group, orig, aug, change in improvements:

        print(
            f"{group:28} "
            f"{orig:.3f} -> {aug:.3f} "
            f"({change:+.3f})"
        )

else:

    print("None")


# ============================================================
# WORSENED GROUPS
# ============================================================

print()
print("WORSENED GROUPS")
print("-" * 80)

if worsened:

    worsened.sort(
        key=lambda x: x[3]
    )

    for group, orig, aug, change in worsened:

        print(
            f"{group:28} "
            f"{orig:.3f} -> {aug:.3f} "
            f"({change:+.3f})"
        )

else:

    print("None")


# ============================================================
# UNCHANGED GROUPS
# ============================================================

print()
print("UNCHANGED GROUPS")
print("-" * 80)

if unchanged:

    for group, orig, aug, change in unchanged:

        print(
            f"{group:28} "
            f"F1 = {orig:.3f}"
        )

else:

    print("None")


# ============================================================
# BEST / WORST GROUP
# ============================================================

print()
print("=" * 100)
print("BEST AND WORST GROUPS")
print("=" * 100)

all_changes = improvements + worsened + unchanged

if all_changes:

    best_group = max(
        all_changes,
        key=lambda x: x[3]
    )

    worst_group = min(
        all_changes,
        key=lambda x: x[3]
    )

    print()
    print(
        f"Best improvement : {best_group[0]} "
        f"({best_group[1]:.3f} -> {best_group[2]:.3f}, "
        f"{best_group[3]:+.3f})"
    )

    print(
        f"Worst change     : {worst_group[0]} "
        f"({worst_group[1]:.3f} -> {worst_group[2]:.3f}, "
        f"{worst_group[3]:+.3f})"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

print()

if aug_r > orig_r:
    print("✓ Online augmentation improves recall.")
else:
    print("✗ Online augmentation reduces recall.")

if aug_p > orig_p:
    print("✓ Online augmentation improves precision.")
else:
    print("✗ Online augmentation reduces precision.")

if aug_f1 > orig_f1:
    print("✓ Online augmentation improves overall F1.")
elif aug_f1 < orig_f1:
    print("✗ Online augmentation reduces overall F1.")
else:
    print("= Online augmentation produces the same overall F1.")


print()
print(
    f"Correct detections (TP): "
    f"{total_orig_tp} -> {total_aug_tp} "
    f"({total_aug_tp-total_orig_tp:+d})"
)

print(
    f"Missed objects (FN): "
    f"{total_orig_fn} -> {total_aug_fn} "
    f"({total_aug_fn-total_orig_fn:+d})"
)

print(
    f"False detections (FP): "
    f"{total_orig_fp} -> {total_aug_fp} "
    f"({total_aug_fp-total_orig_fp:+d})"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print()
print("=" * 100)
print("FINAL CONCLUSION")
print("=" * 100)

print()

if aug_f1 > orig_f1:

    print(
        "At the same confidence threshold, online augmentation "
        "improves overall F1, mainly because it increases true "
        "detections and reduces missed objects."
    )

    print(
        "However, the increase in false positives reduces precision."
    )

elif aug_f1 < orig_f1:

    print(
        "At the same confidence threshold, online augmentation "
        "reduces overall F1."
    )

    print(
        "The augmentation does not provide enough additional "
        "correct detections to compensate for its errors."
    )

else:

    print(
        "At the same confidence threshold, online augmentation "
        "does not change overall F1."
    )


print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)