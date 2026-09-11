from pathlib import Path
from collections import defaultdict
import re
import csv


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

OUTPUT_CSV = Path(
    r"runs\FINAL_RESULTS\per_image_error_analysis.csv"
)


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

    if union <= 0:
        return 0.0

    return inter / union


def get_group(filename):

    stem = Path(filename).stem

    return re.sub(r"_\d+$", "", stem)


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

        try:

            boxes.append(
                xywh_to_xyxy(
                    float(v[1]),
                    float(v[2]),
                    float(v[3]),
                    float(v[4])
                )
            )

        except ValueError:
            continue

    return boxes


def read_predictions(path):

    predictions = []

    if not path.exists():
        return predictions

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        if len(v) < 6:
            continue

        try:

            confidence = float(v[5])

            if confidence < CONF_THRESHOLD:
                continue

            box = xywh_to_xyxy(
                float(v[1]),
                float(v[2]),
                float(v[3]),
                float(v[4])
            )

            predictions.append(
                (box, confidence)
            )

        except ValueError:
            continue

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(gt_boxes, predictions):

    matched = set()

    tp = 0
    fp = 0

    matched_ious = []
    fp_confidences = []

    for pred_box, confidence in predictions:

        best_iou = 0.0
        best_index = -1

        for i, gt_box in enumerate(gt_boxes):

            if i in matched:
                continue

            score = iou(pred_box, gt_box)

            if score > best_iou:

                best_iou = score
                best_index = i

        if best_iou >= IOU_THRESHOLD:

            tp += 1
            matched.add(best_index)
            matched_ious.append(best_iou)

        else:

            fp += 1
            fp_confidences.append(confidence)

    fn = len(gt_boxes) - len(matched)

    return (
        tp,
        fp,
        fn,
        matched_ious,
        fp_confidences
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 110)
print("PER-IMAGE ERROR ANALYSIS")
print("=" * 110)

print()
print(f"GT folder       : {GT_DIR}")
print(f"Original folder : {ORIGINAL_DIR}")
print(f"Online AUG      : {ONLINE_AUG_DIR}")
print(f"Confidence      : {CONF_THRESHOLD:.3f}")
print(f"IoU threshold   : {IOU_THRESHOLD:.2f}")

print()
print("-" * 110)
print("PATH CHECK")
print("-" * 110)

print(f"GT exists       : {GT_DIR.exists()}")
print(f"Original exists : {ORIGINAL_DIR.exists()}")
print(f"Online AUG     : {ONLINE_AUG_DIR.exists()}")


if not GT_DIR.exists():
    print("\nERROR: GT folder does not exist.")
    raise SystemExit

if not ORIGINAL_DIR.exists():
    print("\nERROR: Original prediction folder does not exist.")
    raise SystemExit

if not ONLINE_AUG_DIR.exists():
    print("\nERROR: Online AUG prediction folder does not exist.")
    raise SystemExit


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS IMAGES
# ============================================================

rows = []

for gt_file in sorted(GT_DIR.glob("*.txt")):

    filename = gt_file.name
    group = get_group(filename)

    gt_boxes = read_gt(gt_file)

    orig_file = ORIGINAL_DIR / filename
    aug_file = ONLINE_AUG_DIR / filename

    orig_predictions = read_predictions(
        orig_file
    )

    aug_predictions = read_predictions(
        aug_file
    )

    (
        orig_tp,
        orig_fp,
        orig_fn,
        orig_ious,
        orig_fp_conf
    ) = analyze_image(
        gt_boxes,
        orig_predictions
    )

    (
        aug_tp,
        aug_fp,
        aug_fn,
        aug_ious,
        aug_fp_conf
    ) = analyze_image(
        gt_boxes,
        aug_predictions
    )

    orig_f1_den = (
        2 * orig_tp +
        orig_fp +
        orig_fn
    )

    aug_f1_den = (
        2 * aug_tp +
        aug_fp +
        aug_fn
    )

    orig_f1 = (
        2 * orig_tp / orig_f1_den
        if orig_f1_den > 0
        else 0.0
    )

    aug_f1 = (
        2 * aug_tp / aug_f1_den
        if aug_f1_den > 0
        else 0.0
    )

    rows.append({
        "image": filename,
        "group": group,
        "gt": len(gt_boxes),

        "original_tp": orig_tp,
        "original_fp": orig_fp,
        "original_fn": orig_fn,
        "original_f1": orig_f1,

        "aug_tp": aug_tp,
        "aug_fp": aug_fp,
        "aug_fn": aug_fn,
        "aug_f1": aug_f1,

        "tp_change": aug_tp - orig_tp,
        "fp_change": aug_fp - orig_fp,
        "fn_change": aug_fn - orig_fn,

        "f1_change": aug_f1 - orig_f1,

        "aug_fp_max_conf": (
            max(aug_fp_conf)
            if aug_fp_conf
            else 0.0
        ),

        "aug_fp_avg_conf": (
            sum(aug_fp_conf) / len(aug_fp_conf)
            if aug_fp_conf
            else 0.0
        ),

        "aug_matched_iou_avg": (
            sum(aug_ious) / len(aug_ious)
            if aug_ious
            else 0.0
        )
    })


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = list(rows[0].keys()) if rows else []

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SORTING
# ============================================================

most_fp = sorted(
    rows,
    key=lambda x: x["fp_change"],
    reverse=True
)

most_improved = sorted(
    rows,
    key=lambda x: x["f1_change"],
    reverse=True
)

most_worsened = sorted(
    rows,
    key=lambda x: x["f1_change"]
)

most_fn = sorted(
    rows,
    key=lambda x: x["aug_fn"],
    reverse=True
)


# ============================================================
# TOP FALSE-POSITIVE IMAGES
# ============================================================

print()
print("=" * 110)
print("TOP IMAGES WITH INCREASED FALSE POSITIVES")
print("=" * 110)

print()
print(
    f"{'IMAGE':38} "
    f"{'GROUP':25} "
    f"{'O_FP':>5} "
    f"{'A_FP':>5} "
    f"{'CHANGE':>8}"
)

print("-" * 110)

for row in most_fp[:20]:

    if row["fp_change"] <= 0:
        break

    print(
        f"{row['image'][:38]:38} "
        f"{row['group'][:25]:25} "
        f"{row['original_fp']:5d} "
        f"{row['aug_fp']:5d} "
        f"{row['fp_change']:+8d}"
    )


# ============================================================
# TOP IMPROVED IMAGES
# ============================================================

print()
print("=" * 110)
print("TOP IMPROVED IMAGES")
print("=" * 110)

print()
print(
    f"{'IMAGE':38} "
    f"{'GROUP':25} "
    f"{'O_F1':>7} "
    f"{'A_F1':>7} "
    f"{'CHANGE':>8}"
)

print("-" * 110)

for row in most_improved[:20]:

    if row["f1_change"] <= 0:
        break

    print(
        f"{row['image'][:38]:38} "
        f"{row['group'][:25]:25} "
        f"{row['original_f1']:7.3f} "
        f"{row['aug_f1']:7.3f} "
        f"{row['f1_change']:+8.3f}"
    )


# ============================================================
# TOP WORST IMAGES
# ============================================================

print()
print("=" * 110)
print("TOP WORSENED IMAGES")
print("=" * 110)

print()
print(
    f"{'IMAGE':38} "
    f"{'GROUP':25} "
    f"{'O_F1':>7} "
    f"{'A_F1':>7} "
    f"{'CHANGE':>8}"
)

print("-" * 110)

for row in most_worsened[:20]:

    if row["f1_change"] >= 0:
        break

    print(
        f"{row['image'][:38]:38} "
        f"{row['group'][:25]:25} "
        f"{row['original_f1']:7.3f} "
        f"{row['aug_f1']:7.3f} "
        f"{row['f1_change']:+8.3f}"
    )


# ============================================================
# MOST MISSED OBJECTS
# ============================================================

print()
print("=" * 110)
print("IMAGES WITH MOST MISSED OBJECTS AFTER ONLINE AUGMENTATION")
print("=" * 110)

print()
print(
    f"{'IMAGE':38} "
    f"{'GROUP':25} "
    f"{'GT':>4} "
    f"{'A_TP':>5} "
    f"{'A_FN':>5}"
)

print("-" * 110)

for row in most_fn[:20]:

    print(
        f"{row['image'][:38]:38} "
        f"{row['group'][:25]:25} "
        f"{row['gt']:4d} "
        f"{row['aug_tp']:5d} "
        f"{row['aug_fn']:5d}"
    )


# ============================================================
# HIGH-CONFIDENCE FALSE POSITIVES
# ============================================================

high_fp = sorted(
    rows,
    key=lambda x: x["aug_fp_max_conf"],
    reverse=True
)

print()
print("=" * 110)
print("IMAGES WITH HIGHEST-CONFIDENCE FALSE POSITIVES")
print("=" * 110)

print()
print(
    f"{'IMAGE':38} "
    f"{'GROUP':25} "
    f"{'MAX FP CONF':>12} "
    f"{'AVG FP CONF':>12} "
    f"{'A_FP':>5}"
)

print("-" * 110)

for row in high_fp[:20]:

    if row["aug_fp_max_conf"] <= 0:
        break

    print(
        f"{row['image'][:38]:38} "
        f"{row['group'][:25]:25} "
        f"{row['aug_fp_max_conf']:12.3f} "
        f"{row['aug_fp_avg_conf']:12.3f} "
        f"{row['aug_fp']:5d}"
    )


# ============================================================
# GLOBAL TOTALS
# ============================================================

orig_tp = sum(r["original_tp"] for r in rows)
orig_fp = sum(r["original_fp"] for r in rows)
orig_fn = sum(r["original_fn"] for r in rows)

aug_tp = sum(r["aug_tp"] for r in rows)
aug_fp = sum(r["aug_fp"] for r in rows)
aug_fn = sum(r["aug_fn"] for r in rows)

orig_precision = (
    orig_tp / (orig_tp + orig_fp)
    if orig_tp + orig_fp > 0
    else 0
)

orig_recall = (
    orig_tp / (orig_tp + orig_fn)
    if orig_tp + orig_fn > 0
    else 0
)

orig_f1 = (
    2 * orig_precision * orig_recall /
    (orig_precision + orig_recall)
    if orig_precision + orig_recall > 0
    else 0
)

aug_precision = (
    aug_tp / (aug_tp + aug_fp)
    if aug_tp + aug_fp > 0
    else 0
)

aug_recall = (
    aug_tp / (aug_tp + aug_fn)
    if aug_tp + aug_fn > 0
    else 0
)

aug_f1 = (
    2 * aug_precision * aug_recall /
    (aug_precision + aug_recall)
    if aug_precision + aug_recall > 0
    else 0
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 110)
print("FINAL SUMMARY")
print("=" * 110)

print()

print(
    f"Images analyzed       : {len(rows)}"
)

print(
    f"Original TP           : {orig_tp}"
)

print(
    f"Online AUG TP         : {aug_tp}"
)

print(
    f"Original FP           : {orig_fp}"
)

print(
    f"Online AUG FP         : {aug_fp}"
)

print(
    f"Original FN           : {orig_fn}"
)

print(
    f"Online AUG FN         : {aug_fn}"
)

print()

print(
    f"Original Precision    : {orig_precision:.6f}"
)

print(
    f"Online AUG Precision  : {aug_precision:.6f}"
)

print(
    f"Original Recall       : {orig_recall:.6f}"
)

print(
    f"Online AUG Recall     : {aug_recall:.6f}"
)

print(
    f"Original F1           : {orig_f1:.6f}"
)

print(
    f"Online AUG F1         : {aug_f1:.6f}"
)

print()

print(
    f"TP change             : {aug_tp - orig_tp:+d}"
)

print(
    f"FP change             : {aug_fp - orig_fp:+d}"
)

print(
    f"FN change             : {aug_fn - orig_fn:+d}"
)

print(
    f"F1 change             : {aug_f1 - orig_f1:+.6f}"
)

print()
print("-" * 110)

print(
    f"CSV saved to          : {OUTPUT_CSV}"
)

print("-" * 110)

print()
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)