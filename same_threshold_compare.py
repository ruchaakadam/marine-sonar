from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# CONFIGURATION
# ============================================================

GT = Path("yolo_dataset_groupval/test/labels")

ORIGINAL_PR = Path(
    "runs/FINAL_RESULTS/original_same_threshold_conf001_txt/labels"
)

ONLINE_PR = Path(
    "runs/detect/runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels"
)
IOU_T = 0.50

# Thresholds to compare
THRESHOLDS = [
    0.010,
    0.050,
    0.100,
    0.150,
    0.200,
]


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


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# READ LABEL FILE
# ============================================================

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


def read_predictions(path):

    predictions = []

    if not path.exists():
        return predictions

    for line in path.read_text().splitlines():

        if not line.strip():
            continue

        v = line.split()

        # YOLO prediction format:
        # class x y w h confidence

        if len(v) < 6:
            continue

        box = xywh_to_xyxy(
            float(v[1]),
            float(v[2]),
            float(v[3]),
            float(v[4])
        )

        confidence = float(v[5])

        predictions.append(
            (box, confidence)
        )

    return predictions


# ============================================================
# GROUP NAME
# ============================================================

def group_name(filename):

    # Example:
    # Artificial_Reef_01 -> Artificial_Reef
    # Monrovia_05 -> Monrovia
    # Lucinda_van_Valkenburg_07 -> Lucinda_van_Valkenburg

    return re.sub(
        r"_\d+$",
        "",
        Path(filename).stem
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(prediction_folder, threshold):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    stats = defaultdict(
        lambda: {
            "images": 0,
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    prediction_files = 0

    if prediction_folder.exists():

        prediction_files = len(
            list(
                prediction_folder.glob("*.txt")
            )
        )

    for gt_file in GT.glob("*.txt"):

        group = group_name(gt_file.name)

        stats[group]["images"] += 1

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        gt_boxes = read_gt(gt_file)

        stats[group]["gt"] += len(gt_boxes)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        pred_file = (
            prediction_folder
            / gt_file.name
        )

        predictions = read_predictions(
            pred_file
        )

        # ----------------------------------------------------
        # Apply confidence threshold
        # ----------------------------------------------------

        predictions = [
            (box, conf)
            for box, conf in predictions
            if conf >= threshold
        ]

        # Highest confidence first
        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ----------------------------------------------------
        # Match predictions to GT
        # ----------------------------------------------------

        matched_gt = set()

        tp = 0
        fp = 0

        for pred_box, confidence in predictions:

            best_iou = 0.0
            best_gt = -1

            for i, gt_box in enumerate(gt_boxes):

                if i in matched_gt:
                    continue

                score = iou(
                    pred_box,
                    gt_box
                )

                if score > best_iou:

                    best_iou = score
                    best_gt = i

            if (
                best_gt >= 0
                and best_iou >= IOU_T
            ):

                tp += 1
                matched_gt.add(best_gt)

            else:

                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        stats[group]["tp"] += tp
        stats[group]["fp"] += fp
        stats[group]["fn"] += fn

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction_files": prediction_files,
        "stats": stats
    }


# ============================================================
# PRINT MODEL ANALYSIS
# ============================================================

def print_model_results(
    title,
    prediction_folder
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"Prediction folder : {prediction_folder}"
    )

    print(
        f"GT files          : "
        f"{len(list(GT.glob('*.txt')))}"
    )

    print(
        f"Prediction files  : "
        f"{len(list(prediction_folder.glob('*.txt')))}"
    )

    print()

    print(
        f"{'CONF':>7} | "
        f"{'TP':>4} | "
        f"{'FP':>5} | "
        f"{'FN':>4} | "
        f"{'PRECISION':>9} | "
        f"{'RECALL':>7} | "
        f"{'F1':>7}"
    )

    print("-" * 90)

    results = {}

    for threshold in THRESHOLDS:

        result = evaluate(
            prediction_folder,
            threshold
        )

        results[threshold] = result

        print(
            f"{threshold:7.3f} | "
            f"{result['tp']:4d} | "
            f"{result['fp']:5d} | "
            f"{result['fn']:4d} | "
            f"{result['precision']:9.4f} | "
            f"{result['recall']:7.4f} | "
            f"{result['f1']:7.4f}"
        )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_threshold = max(
        results,
        key=lambda x: results[x]["f1"]
    )

    best = results[best_threshold]

    print()
    print("=" * 90)
    print("BEST F1 THRESHOLD")
    print("=" * 90)

    print(
        f"Confidence : {best_threshold:.3f}"
    )

    print(
        f"TP         : {best['tp']}"
    )

    print(
        f"FP         : {best['fp']}"
    )

    print(
        f"FN         : {best['fn']}"
    )

    print(
        f"Precision  : {best['precision']:.6f}"
    )

    print(
        f"Recall     : {best['recall']:.6f}"
    )

    print(
        f"F1         : {best['f1']:.6f}"
    )

    print("=" * 90)

    return results


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 90)
print("ORIGINAL MODEL VS ONLINE AUGMENTATION")
print("=" * 90)

print(
    f"GT folder       : {GT}"
)

print(
    f"Original folder : {ORIGINAL_PR}"
)

print(
    f"Online AUG      : {ONLINE_PR}"
)

print(
    f"IoU threshold   : {IOU_T}"
)

print()


# ============================================================
# PATH CHECK
# ============================================================

print("=" * 90)
print("PATH CHECK")
print("=" * 90)

print(
    f"GT exists       : {GT.exists()}"
)

print(
    f"Original exists : {ORIGINAL_PR.exists()}"
)

print(
    f"Online AUG exists: {ONLINE_PR.exists()}"
)

if not GT.exists():

    print()
    print("ERROR: GT folder does not exist.")
    print(GT)
    raise SystemExit


if not ORIGINAL_PR.exists():

    print()
    print("ERROR: ORIGINAL prediction folder does not exist.")
    print(ORIGINAL_PR)
    raise SystemExit


if not ONLINE_PR.exists():

    print()
    print("ERROR: ONLINE AUG prediction folder does not exist.")
    print(ONLINE_PR)
    raise SystemExit


gt_count = len(
    list(GT.glob("*.txt"))
)

original_count = len(
    list(ORIGINAL_PR.glob("*.txt"))
)

online_count = len(
    list(ONLINE_PR.glob("*.txt"))
)

print()

print(
    f"GT files       : {gt_count}"
)

print(
    f"Original files : {original_count}"
)

print(
    f"Online AUG     : {online_count}"
)

if gt_count != 120:

    print(
        "\nWARNING: Expected 120 GT files."
    )

if original_count != 120:

    print(
        "\nWARNING: Original model does not have 120 prediction files."
    )

if online_count != 120:

    print(
        "\nWARNING: Online augmentation does not have 120 prediction files."
    )


# ============================================================
# ORIGINAL MODEL
# ============================================================

original_results = print_model_results(
    "ORIGINAL MODEL - SAME THRESHOLD ANALYSIS",
    ORIGINAL_PR
)


# ============================================================
# ONLINE AUGMENTATION
# ============================================================

online_results = print_model_results(
    "ONLINE AUGMENTATION - SAME THRESHOLD ANALYSIS",
    ONLINE_PR
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 100)
print("FINAL SAME-THRESHOLD COMPARISON")
print("=" * 100)

print(
    f"{'CONF':>7} | "
    f"{'ORIG F1':>8} | "
    f"{'AUG F1':>8} | "
    f"{'CHANGE':>9} | "
    f"{'ORIG P':>8} | "
    f"{'AUG P':>8} | "
    f"{'ORIG R':>8} | "
    f"{'AUG R':>8}"
)

print("-" * 100)

for threshold in THRESHOLDS:

    orig = original_results[threshold]
    aug = online_results[threshold]

    f1_change = (
        aug["f1"] - orig["f1"]
    )

    print(
        f"{threshold:7.3f} | "
        f"{orig['f1']:8.4f} | "
        f"{aug['f1']:8.4f} | "
        f"{f1_change:+9.4f} | "
        f"{orig['precision']:8.4f} | "
        f"{aug['precision']:8.4f} | "
        f"{orig['recall']:8.4f} | "
        f"{aug['recall']:8.4f}"
    )


# ============================================================
# BEST THRESHOLD FOR EACH MODEL
# ============================================================

best_orig_threshold = max(
    original_results,
    key=lambda x: original_results[x]["f1"]
)

best_aug_threshold = max(
    online_results,
    key=lambda x: online_results[x]["f1"]
)

orig_best = original_results[
    best_orig_threshold
]

aug_best = online_results[
    best_aug_threshold
]


print()
print("=" * 100)
print("BEST THRESHOLD SUMMARY")
print("=" * 100)

print()
print("ORIGINAL MODEL")

print(
    f"Confidence : {best_orig_threshold:.3f}"
)

print(
    f"TP         : {orig_best['tp']}"
)

print(
    f"FP         : {orig_best['fp']}"
)

print(
    f"FN         : {orig_best['fn']}"
)

print(
    f"Precision  : {orig_best['precision']:.6f}"
)

print(
    f"Recall     : {orig_best['recall']:.6f}"
)

print(
    f"F1         : {orig_best['f1']:.6f}"
)


print()
print("ONLINE AUGMENTATION")

print(
    f"Confidence : {best_aug_threshold:.3f}"
)

print(
    f"TP         : {aug_best['tp']}"
)

print(
    f"FP         : {aug_best['fp']}"
)

print(
    f"FN         : {aug_best['fn']}"
)

print(
    f"Precision  : {aug_best['precision']:.6f}"
)

print(
    f"Recall     : {aug_best['recall']:.6f}"
)

print(
    f"F1         : {aug_best['f1']:.6f}"
)


# ============================================================
# OVERALL IMPROVEMENT AT 0.05
# ============================================================

if 0.050 in original_results:

    orig = original_results[0.050]
    aug = online_results[0.050]

    print()
    print("=" * 100)
    print("COMPARISON AT CONFIDENCE = 0.050")
    print("=" * 100)

    print(
        f"{'METRIC':15} "
        f"{'ORIGINAL':>12} "
        f"{'ONLINE AUG':>14} "
        f"{'CHANGE':>12}"
    )

    print("-" * 60)

    print(
        f"{'TP':15} "
        f"{orig['tp']:12d} "
        f"{aug['tp']:14d} "
        f"{aug['tp'] - orig['tp']:+12d}"
    )

    print(
        f"{'FP':15} "
        f"{orig['fp']:12d} "
        f"{aug['fp']:14d} "
        f"{aug['fp'] - orig['fp']:+12d}"
    )

    print(
        f"{'FN':15} "
        f"{orig['fn']:12d} "
        f"{aug['fn']:14d} "
        f"{aug['fn'] - orig['fn']:+12d}"
    )

    print()

    print(
        f"{'Precision':15} "
        f"{orig['precision']:12.6f} "
        f"{aug['precision']:14.6f} "
        f"{aug['precision'] - orig['precision']:+12.6f}"
    )

    print(
        f"{'Recall':15} "
        f"{orig['recall']:12.6f} "
        f"{aug['recall']:14.6f} "
        f"{aug['recall'] - orig['recall']:+12.6f}"
    )

    print(
        f"{'F1':15} "
        f"{orig['f1']:12.6f} "
        f"{aug['f1']:14.6f} "
        f"{aug['f1'] - orig['f1']:+12.6f}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print()