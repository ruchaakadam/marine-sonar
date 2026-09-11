from pathlib import Path
import re
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path(
    r"yolo_dataset_groupval\test\images"
)

GT_DIR = Path(
    r"yolo_dataset_groupval\test\labels"
)

ORIGINAL_DIR = Path(
    r"runs\FINAL_RESULTS\original_same_threshold_conf001_txt\labels"
)

ONLINE_AUG_DIR = Path(
    r"runs\detect\runs\FINAL_RESULTS\onlineaug_predictions_conf001_txt\labels"
)

OUTPUT_DIR = Path(
    r"runs\FINAL_RESULTS\visual_error_analysis"
)

CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.50


# ============================================================
# FUNCTIONS
# ============================================================

def xywh_to_xyxy(x, y, w, h, img_w, img_h):

    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)

    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)

    return x1, y1, x2, y2


def calculate_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def read_gt(path, img_w, img_h):

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

            box = xywh_to_xyxy(
                float(v[1]),
                float(v[2]),
                float(v[3]),
                float(v[4]),
                img_w,
                img_h
            )

            boxes.append(box)

        except ValueError:
            continue

    return boxes


def read_predictions(path, img_w, img_h):

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
                float(v[4]),
                img_w,
                img_h
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


def evaluate(gt_boxes, predictions):

    matched = set()

    tp_boxes = []
    fp_boxes = []

    for pred_box, confidence in predictions:

        best_iou = 0.0
        best_index = -1

        for i, gt_box in enumerate(gt_boxes):

            if i in matched:
                continue

            score = calculate_iou(
                pred_box,
                gt_box
            )

            if score > best_iou:

                best_iou = score
                best_index = i

        if best_iou >= IOU_THRESHOLD:

            matched.add(best_index)

            tp_boxes.append(
                (pred_box, confidence, best_iou)
            )

        else:

            fp_boxes.append(
                (pred_box, confidence, best_iou)
            )

    fn_boxes = []

    for i, gt_box in enumerate(gt_boxes):

        if i not in matched:
            fn_boxes.append(gt_box)

    return tp_boxes, fp_boxes, fn_boxes


def draw_boxes(
    image,
    gt_boxes,
    tp_boxes,
    fp_boxes,
    fn_boxes,
    title
):

    output = image.copy()

    # --------------------------------------------------------
    # GT boxes
    # --------------------------------------------------------

    for box in gt_boxes:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

    # --------------------------------------------------------
    # TP boxes
    # --------------------------------------------------------

    for box, confidence, score in tp_boxes:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"TP {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # FP boxes
    # --------------------------------------------------------

    for box, confidence, score in fp_boxes:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            f"FP {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # FN boxes
    # --------------------------------------------------------

    for box in fn_boxes:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            3
        )

        cv2.putText(
            output,
            "FN",
            (x1, min(output.shape[0] - 5, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# PATH CHECK
# ============================================================

print()
print("=" * 100)
print("VISUAL ERROR ANALYSIS")
print("=" * 100)

print()
print(f"Images       : {IMAGE_DIR}")
print(f"GT           : {GT_DIR}")
print(f"Original     : {ORIGINAL_DIR}")
print(f"Online AUG   : {ONLINE_AUG_DIR}")
print(f"Output       : {OUTPUT_DIR}")

print()
print("-" * 100)
print("PATH CHECK")
print("-" * 100)

print(f"Images exists     : {IMAGE_DIR.exists()}")
print(f"GT exists         : {GT_DIR.exists()}")
print(f"Original exists   : {ORIGINAL_DIR.exists()}")
print(f"Online AUG exists : {ONLINE_AUG_DIR.exists()}")


if not IMAGE_DIR.exists():
    print("\nERROR: Image directory does not exist.")
    raise SystemExit

if not GT_DIR.exists():
    print("\nERROR: GT directory does not exist.")
    raise SystemExit

if not ORIGINAL_DIR.exists():
    print("\nERROR: Original prediction directory does not exist.")
    raise SystemExit

if not ONLINE_AUG_DIR.exists():
    print("\nERROR: Online AUG prediction directory does not exist.")
    raise SystemExit


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

(OUTPUT_DIR / "highest_fp").mkdir(
    exist_ok=True
)

(OUTPUT_DIR / "highest_fn").mkdir(
    exist_ok=True
)

(OUTPUT_DIR / "improved").mkdir(
    exist_ok=True
)

(OUTPUT_DIR / "worsened").mkdir(
    exist_ok=True
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff"
]

image_files = []

for ext in extensions:
    image_files.extend(
        IMAGE_DIR.glob(ext)
    )

image_files = sorted(image_files)


print()
print(f"Images found : {len(image_files)}")


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS
# ============================================================

for image_file in image_files:

    filename = image_file.name
    stem = image_file.stem

    gt_file = GT_DIR / f"{stem}.txt"

    original_file = ORIGINAL_DIR / f"{stem}.txt"

    aug_file = ONLINE_AUG_DIR / f"{stem}.txt"

    image = cv2.imread(
        str(image_file)
    )

    if image is None:
        continue

    img_h, img_w = image.shape[:2]

    gt_boxes = read_gt(
        gt_file,
        img_w,
        img_h
    )

    original_predictions = read_predictions(
        original_file,
        img_w,
        img_h
    )

    aug_predictions = read_predictions(
        aug_file,
        img_w,
        img_h
    )

    (
        orig_tp,
        orig_fp,
        orig_fn
    ) = evaluate(
        gt_boxes,
        original_predictions
    )

    (
        aug_tp,
        aug_fp,
        aug_fn
    ) = evaluate(
        gt_boxes,
        aug_predictions
    )

    orig_f1_den = (
        2 * len(orig_tp)
        + len(orig_fp)
        + len(orig_fn)
    )

    aug_f1_den = (
        2 * len(aug_tp)
        + len(aug_fp)
        + len(aug_fn)
    )

    orig_f1 = (
        2 * len(orig_tp) / orig_f1_den
        if orig_f1_den > 0
        else 0.0
    )

    aug_f1 = (
        2 * len(aug_tp) / aug_f1_den
        if aug_f1_den > 0
        else 0.0
    )

    results.append({
        "image": filename,
        "gt": len(gt_boxes),

        "orig_tp": len(orig_tp),
        "orig_fp": len(orig_fp),
        "orig_fn": len(orig_fn),
        "orig_f1": orig_f1,

        "aug_tp": len(aug_tp),
        "aug_fp": len(aug_fp),
        "aug_fn": len(aug_fn),
        "aug_f1": aug_f1,

        "fp_change": len(aug_fp) - len(orig_fp),
        "fn_change": len(aug_fn) - len(orig_fn),
        "f1_change": aug_f1 - orig_f1,

        "aug_fp_conf": (
            max(
                [x[1] for x in aug_fp]
            )
            if aug_fp
            else 0.0
        )
    })

    # ========================================================
    # CREATE ORIGINAL VISUAL
    # ========================================================

    original_visual = draw_boxes(
        image,
        gt_boxes,
        orig_tp,
        orig_fp,
        orig_fn,
        f"ORIGINAL | {filename}"
    )

    # ========================================================
    # CREATE AUGMENTATION VISUAL
    # ========================================================

    aug_visual = draw_boxes(
        image,
        gt_boxes,
        aug_tp,
        aug_fp,
        aug_fn,
        f"ONLINE AUG | {filename}"
    )

    # ========================================================
    # SIDE-BY-SIDE
    # ========================================================

    comparison = cv2.hconcat([
        original_visual,
        aug_visual
    ])

    # ========================================================
    # SAVE TEMPORARY BASE COMPARISON
    # ========================================================

    base_output = (
        OUTPUT_DIR /
        f"{stem}_comparison.jpg"
    )

    cv2.imwrite(
        str(base_output),
        comparison
    )


# ============================================================
# SELECT IMPORTANT IMAGES
# ============================================================

highest_fp = sorted(
    results,
    key=lambda x: x["fp_change"],
    reverse=True
)

highest_fn = sorted(
    results,
    key=lambda x: x["aug_fn"],
    reverse=True
)

improved = sorted(
    results,
    key=lambda x: x["f1_change"],
    reverse=True
)

worsened = sorted(
    results,
    key=lambda x: x["f1_change"]
)


# ============================================================
# COPY / ORGANIZE IMPORTANT IMAGES
# ============================================================

def organize_images(items, folder, condition):

    count = 0

    for row in items:

        if not condition(row):
            continue

        source = (
            OUTPUT_DIR /
            f"{Path(row['image']).stem}_comparison.jpg"
        )

        if not source.exists():
            continue

        destination = (
            folder /
            f"{count+1:02d}_{row['image']}_comparison.jpg"
        )

        # Read and write instead of shutil
        # to avoid filename issues
        img = cv2.imread(
            str(source)
        )

        if img is not None:

            cv2.imwrite(
                str(destination),
                img
            )

        count += 1

        if count >= 20:
            break


organize_images(
    highest_fp,
    OUTPUT_DIR / "highest_fp",
    lambda x: x["fp_change"] > 0
)

organize_images(
    highest_fn,
    OUTPUT_DIR / "highest_fn",
    lambda x: x["aug_fn"] > 0
)

organize_images(
    improved,
    OUTPUT_DIR / "improved",
    lambda x: x["f1_change"] > 0
)

organize_images(
    worsened,
    OUTPUT_DIR / "worsened",
    lambda x: x["f1_change"] < 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 110)
print("TOP FALSE-POSITIVE INCREASES")
print("=" * 110)

print()

for row in highest_fp[:20]:

    if row["fp_change"] <= 0:
        break

    print(
        f"{row['image']:35} "
        f"Original FP={row['orig_fp']:3d} "
        f"AUG FP={row['aug_fp']:3d} "
        f"Change={row['fp_change']:+3d}"
    )


print()
print("=" * 110)
print("TOP WORSENED IMAGES")
print("=" * 110)

print()

for row in worsened[:20]:

    if row["f1_change"] >= 0:
        break

    print(
        f"{row['image']:35} "
        f"Original F1={row['orig_f1']:.3f} "
        f"AUG F1={row['aug_f1']:.3f} "
        f"Change={row['f1_change']:+.3f}"
    )


print()
print("=" * 110)
print("TOP IMPROVED IMAGES")
print("=" * 110)

print()

for row in improved[:20]:

    if row["f1_change"] <= 0:
        break

    print(
        f"{row['image']:35} "
        f"Original F1={row['orig_f1']:.3f} "
        f"AUG F1={row['aug_f1']:.3f} "
        f"Change={row['f1_change']:+.3f}"
    )


print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print()
print(
    f"All comparisons : {OUTPUT_DIR}"
)

print(
    f"High FP cases   : {OUTPUT_DIR / 'highest_fp'}"
)

print(
    f"High FN cases   : {OUTPUT_DIR / 'highest_fn'}"
)

print(
    f"Improved cases  : {OUTPUT_DIR / 'improved'}"
)

print(
    f"Worsened cases  : {OUTPUT_DIR / 'worsened'}"
)

print()
print("=" * 110)
print("VISUAL ANALYSIS COMPLETE")
print("=" * 110)