from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"D:\sih\marine-sonar")

MODEL = ROOT / r"runs\detect\runs\detect\groupval_onlineaug_1024\weights\best.pt"

IMAGES = ROOT / r"yolo_dataset_groupval\test\images"
GT_LABELS = ROOT / r"yolo_dataset_groupval\test\labels"

OUTPUT = ROOT / r"runs\FINAL_RESULTS\confidence_nms_grid"

# Confidence values to test
CONF_VALUES = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
]

# NMS IoU values to test
NMS_VALUES = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

MATCH_IOU = 0.50

IMG_SIZE = 1024
DEVICE = 0


# ============================================================
# HELPERS
# ============================================================

def load_gt(label_file):
    """
    Load YOLO-format ground truth.

    Returns:
        list of [class_id, x1, y1, x2, y2]
    """

    objects = []

    if not label_file.exists():
        return objects

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        objects.append([
            cls,
            xc,
            yc,
            w,
            h
        ])

    return objects


def yolo_to_xyxy(obj, img_w, img_h):
    """
    Convert YOLO normalized box to pixel xyxy.
    """

    cls, xc, yc, w, h = obj

    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h

    return [
        cls,
        x1,
        y1,
        x2,
        y2
    ]


def box_iou(box1, box2):
    """
    IoU between two xyxy boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * \
            max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * \
            max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def match_predictions(gt_boxes, pred_boxes):
    """
    Greedy one-to-one matching.

    Prediction format:
        [class_id, confidence, x1, y1, x2, y2]

    GT format:
        [class_id, x1, y1, x2, y2]
    """

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0

    # Highest confidence first
    pred_boxes = sorted(
        pred_boxes,
        key=lambda x: x[1],
        reverse=True
    )

    for pred in pred_boxes:

        pred_cls = pred[0]

        pred_box = [
            pred[2],
            pred[3],
            pred[4],
            pred[5]
        ]

        best_iou = 0.0
        best_gt_idx = None

        for i, gt in enumerate(gt_boxes):

            if i in matched_gt:
                continue

            gt_cls = gt[0]

            if pred_cls != gt_cls:
                continue

            gt_box = [
                gt[1],
                gt[2],
                gt[3],
                gt[4]
            ]

            iou = box_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if (
            best_gt_idx is not None
            and best_iou >= MATCH_IOU
        ):
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


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
# PATH CHECK
# ============================================================

print("=" * 90)
print("CONFIDENCE × NMS GRID SEARCH")
print("=" * 90)

print()
print("Model       :", MODEL)
print("Images      :", IMAGES)
print("GT labels   :", GT_LABELS)
print("Match IoU   :", MATCH_IOU)
print("Confidence  :", CONF_VALUES)
print("NMS IoU     :", NMS_VALUES)

print()
print("=" * 90)
print("PATH CHECK")
print("=" * 90)

print("Model exists :", MODEL.exists())
print("Images exist:", IMAGES.exists())
print("GT exists   :", GT_LABELS.exists())

if not MODEL.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL}"
    )

if not IMAGES.exists():
    raise FileNotFoundError(
        f"Images folder not found:\n{IMAGES}"
    )

if not GT_LABELS.exists():
    raise FileNotFoundError(
        f"GT folder not found:\n{GT_LABELS}"
    )


# ============================================================
# IMAGE LIST
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

image_files = sorted([
    p for p in IMAGES.iterdir()
    if p.suffix.lower() in image_extensions
])

print()
print("Images found:", len(image_files))

if len(image_files) == 0:
    raise RuntimeError("No images found.")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")

model = YOLO(str(MODEL))

print("Model loaded.")


# ============================================================
# CACHE RAW PREDICTIONS
# ============================================================

print()
print("=" * 90)
print("RUNNING RAW PREDICTIONS")
print("=" * 90)

print()
print(
    "Important: predictions are generated once at conf=0.001."
)
print(
    "The confidence and NMS grid is then evaluated from the"
)
print(
    "cached detections."
)

raw_predictions = {}

for idx, image_path in enumerate(image_files, start=1):

    if idx % 10 == 0 or idx == 1:
        print(
            f"Processing image {idx}/{len(image_files)}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"WARNING: could not read {image_path}"
        )
        raw_predictions[image_path.name] = []
        continue

    h, w = image.shape[:2]

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=0.001,
        iou=0.50,
        save=False,
        verbose=False
    )

    detections = []

    if len(results) > 0:

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                xyxy = boxes.xyxy[i].cpu().numpy()

                confidence = float(
                    boxes.conf[i].cpu().item()
                )

                cls = int(
                    boxes.cls[i].cpu().item()
                )

                detections.append([
                    cls,
                    confidence,
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                ])

    raw_predictions[image_path.name] = detections


print()
print(
    "Raw prediction cache complete."
)


# ============================================================
# GRID SEARCH
# ============================================================

all_results = []

best_result = None

for nms_iou in NMS_VALUES:

    print()
    print("=" * 90)
    print(
        f"RUNNING NMS IoU = {nms_iou:.2f}"
    )
    print("=" * 90)

    for conf in CONF_VALUES:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        total_predictions = 0

        for image_path in image_files:

            image_name = image_path.name

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            h, w = image.shape[:2]

            # -----------------------------
            # Ground truth
            # -----------------------------

            gt_file = GT_LABELS / (
                image_path.stem + ".txt"
            )

            gt_raw = load_gt(gt_file)

            gt_boxes = [
                yolo_to_xyxy(
                    obj,
                    w,
                    h
                )
                for obj in gt_raw
            ]

            # -----------------------------
            # Raw detections
            # -----------------------------

            detections = raw_predictions.get(
                image_name,
                []
            )

            # Confidence filtering
            detections = [
                d for d in detections
                if d[1] >= conf
            ]

            # -----------------------------
            # NMS
            # -----------------------------

            pred_boxes = []

            if len(detections) > 0:

                boxes_xyxy = np.array([
                    [
                        d[2],
                        d[3],
                        d[4],
                        d[5]
                    ]
                    for d in detections
                ])

                scores = np.array([
                    d[1]
                    for d in detections
                ])

                class_ids = [
                    d[0]
                    for d in detections
                ]

                # Class-aware NMS
                final_indices = []

                unique_classes = sorted(
                    set(class_ids)
                )

                for cls in unique_classes:

                    cls_indices = [
                        i
                        for i, c
                        in enumerate(class_ids)
                        if c == cls
                    ]

                    cls_boxes = boxes_xyxy[
                        cls_indices
                    ]

                    cls_scores = scores[
                        cls_indices
                    ]

                    indices = cv2.dnn.NMSBoxes(
                        cls_boxes.tolist(),
                        cls_scores.tolist(),
                        score_threshold=conf,
                        nms_threshold=nms_iou
                    )

                    if len(indices) > 0:

                        indices = np.array(
                            indices
                        ).reshape(-1)

                        for local_idx in indices:

                            final_indices.append(
                                cls_indices[
                                    int(local_idx)
                                ]
                            )

                for i in final_indices:

                    d = detections[i]

                    pred_boxes.append(d)

            total_predictions += len(
                pred_boxes
            )

            # -----------------------------
            # Match
            # -----------------------------

            tp, fp, fn = match_predictions(
                gt_boxes,
                pred_boxes
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision, recall, f1 = calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )

        result = {
            "confidence": conf,
            "nms_iou": nms_iou,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predictions": total_predictions,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        all_results.append(result)

        if (
            best_result is None
            or f1 > best_result["f1"]
        ):
            best_result = result.copy()

        print(
            f"CONF={conf:.3f} | "
            f"NMS={nms_iou:.2f} | "
            f"TP={total_tp:3d} | "
            f"FP={total_fp:3d} | "
            f"FN={total_fn:3d} | "
            f"P={precision:.4f} | "
            f"R={recall:.4f} | "
            f"F1={f1:.4f}"
        )


# ============================================================
# SORT RESULTS
# ============================================================

all_results_sorted = sorted(
    all_results,
    key=lambda x: x["f1"],
    reverse=True
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 100)
print("FINAL CONFIDENCE × NMS RESULTS")
print("=" * 100)

print(
    "CONF | NMS  | TP | FP | FN | "
    "PRECISION | RECALL | F1"
)

print("-" * 100)

for r in all_results_sorted:

    print(
        f"{r['confidence']:.3f} | "
        f"{r['nms_iou']:.2f} | "
        f"{r['tp']:2d} | "
        f"{r['fp']:3d} | "
        f"{r['fn']:2d} | "
        f"{r['precision']:.4f}    | "
        f"{r['recall']:.4f} | "
        f"{r['f1']:.4f}"
    )


# ============================================================
# TOP 10
# ============================================================

print()
print("=" * 100)
print("TOP 10 SETTINGS")
print("=" * 100)

for rank, r in enumerate(
    all_results_sorted[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"CONF={r['confidence']:.3f}, "
        f"NMS={r['nms_iou']:.2f}, "
        f"TP={r['tp']}, "
        f"FP={r['fp']}, "
        f"FN={r['fn']}, "
        f"P={r['precision']:.4f}, "
        f"R={r['recall']:.4f}, "
        f"F1={r['f1']:.4f}"
    )


# ============================================================
# BEST RESULT
# ============================================================

print()
print("=" * 100)
print("BEST SETTING")
print("=" * 100)

print(
    f"Confidence : {best_result['confidence']:.3f}"
)

print(
    f"NMS IoU    : {best_result['nms_iou']:.2f}"
)

print(
    f"TP         : {best_result['tp']}"
)

print(
    f"FP         : {best_result['fp']}"
)

print(
    f"FN         : {best_result['fn']}"
)

print(
    f"Precision  : {best_result['precision']:.6f}"
)

print(
    f"Recall     : {best_result['recall']:.6f}"
)

print(
    f"F1         : {best_result['f1']:.6f}"
)


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = OUTPUT / "confidence_nms_grid_results.csv"

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "confidence,nms_iou,tp,fp,fn,"
        "predictions,precision,recall,f1\n"
    )

    for r in all_results:

        f.write(
            f"{r['confidence']:.4f},"
            f"{r['nms_iou']:.2f},"
            f"{r['tp']},"
            f"{r['fp']},"
            f"{r['fn']},"
            f"{r['predictions']},"
            f"{r['precision']:.6f},"
            f"{r['recall']:.6f},"
            f"{r['f1']:.6f}\n"
        )


# ============================================================
# SAVE BEST SETTING
# ============================================================

best_path = OUTPUT / "best_setting.txt"

with open(
    best_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BEST CONFIDENCE × NMS SETTING\n"
    )

    f.write(
        "================================\n"
    )

    for key, value in best_result.items():

        f.write(
            f"{key}: {value}\n"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "CSV saved to:"
)

print(
    csv_path
)

print()
print(
    "Best setting saved to:"
)

print(
    best_path
)

print()
print("=" * 100)
print("GRID SEARCH COMPLETE")
print("=" * 100)