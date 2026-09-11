from pathlib import Path
from ultralytics import YOLO
import numpy as np
import shutil

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"D:\sih\marine-sonar")

MODEL_PATH = ROOT / r"runs\detect\runs\detect\groupval_onlineaug_1024\weights\best.pt"

IMAGE_DIR = ROOT / r"yolo_dataset_groupval\test\images"
GT_DIR = ROOT / r"yolo_dataset_groupval\test\labels"

OUTPUT_ROOT = ROOT / r"runs\FINAL_RESULTS\nms_experiment"

CONF = 0.05
NMS_IOUS = [0.30, 0.40, 0.50, 0.60]

MATCH_IOU = 0.50

IMG_SIZE = 1024
DEVICE = 0


# ============================================================
# HELPERS
# ============================================================

def xywhn_to_xyxy(box, img_w, img_h):
    x, y, w, h = box

    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return [x1, y1, x2, y2]


def box_iou(a, b):
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


def read_gt(label_path, img_w, img_h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(parts[0])

            # This project is single-class, but keep class information.
            box = list(map(float, parts[1:5]))

            xyxy = xywhn_to_xyxy(box, img_w, img_h)

            boxes.append({
                "cls": cls,
                "box": xyxy
            })

    return boxes


def calculate_metrics(gt_boxes, pred_boxes):
    """
    Greedy one-to-one matching.
    Prediction must have IoU >= MATCH_IOU with an unused GT.
    """

    matched_gt = set()

    # Highest confidence first
    pred_boxes = sorted(
        pred_boxes,
        key=lambda x: x["conf"],
        reverse=True
    )

    tp = 0
    fp = 0

    for pred in pred_boxes:

        best_iou = 0.0
        best_gt = None

        for i, gt in enumerate(gt_boxes):

            if i in matched_gt:
                continue

            if pred["cls"] != gt["cls"]:
                continue

            iou = box_iou(pred["box"], gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= MATCH_IOU:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return tp, fp, fn, precision, recall, f1


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("ONLINE AUGMENTATION - NMS EXPERIMENT")
print("=" * 80)

print(f"Model        : {MODEL_PATH}")
print(f"Images       : {IMAGE_DIR}")
print(f"GT labels    : {GT_DIR}")
print(f"Confidence   : {CONF}")
print(f"Match IoU    : {MATCH_IOU}")
print(f"NMS IoUs     : {NMS_IOUS}")

print("=" * 80)
print("PATH CHECK")
print("=" * 80)

print(f"Model exists : {MODEL_PATH.exists()}")
print(f"Images exist : {IMAGE_DIR.exists()}")
print(f"GT exists    : {GT_DIR.exists()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found:\n{IMAGE_DIR}")

if not GT_DIR.exists():
    raise FileNotFoundError(f"GT folder not found:\n{GT_DIR}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print()
print("Loading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")
print()


# ------------------------------------------------------------
# Find images
# ------------------------------------------------------------

images = sorted(
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print(f"Images found : {len(images)}")

if len(images) == 0:
    raise RuntimeError("No images found.")


# ------------------------------------------------------------
# Results storage
# ------------------------------------------------------------

results = []

for nms_iou in NMS_IOUS:

    print()
    print("=" * 80)
    print(f"RUNNING NMS IoU = {nms_iou:.2f}")
    print("=" * 80)

    output_dir = OUTPUT_ROOT / f"conf{CONF:.3f}_nms{nms_iou:.2f}"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    processed = 0

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        source=str(IMAGE_DIR),
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=CONF,
        iou=nms_iou,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="predict",
        exist_ok=True,
        verbose=False
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Read predictions directly from result objects.
    # This avoids path confusion.
    # --------------------------------------------------------

    for result in predictions:

        image_path = Path(result.path)

        img_h, img_w = result.orig_shape

        gt_path = GT_DIR / f"{image_path.stem}.txt"

        gt_boxes = read_gt(
            gt_path,
            img_w,
            img_h
        )

        pred_boxes = []

        if result.boxes is not None and len(result.boxes) > 0:

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(
                xyxy,
                confs,
                classes
            ):
                pred_boxes.append({
                    "box": box.tolist(),
                    "conf": float(conf),
                    "cls": int(cls)
                })

        tp, fp, fn, precision, recall, f1 = calculate_metrics(
            gt_boxes,
            pred_boxes
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        processed += 1

        # ----------------------------------------------------
        # Save YOLO prediction txt
        # ----------------------------------------------------

        pred_txt = label_dir / f"{image_path.stem}.txt"

        with open(pred_txt, "w") as f:

            for pred in pred_boxes:

                x1, y1, x2, y2 = pred["box"]

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                f.write(
                    f"{pred['cls']} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f} "
                    f"{pred['conf']:.6f}\n"
                )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

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
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results.append({
        "nms": nms_iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print()
    print(f"Images     : {processed}")
    print(f"TP         : {total_tp}")
    print(f"FP         : {total_fp}")
    print(f"FN         : {total_fn}")
    print(f"Precision  : {precision:.6f}")
    print(f"Recall     : {recall:.6f}")
    print(f"F1         : {f1:.6f}")


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 80)
print("FINAL NMS COMPARISON")
print("=" * 80)

print(
    "NMS IoU | TP | FP | FN | Precision | Recall | F1"
)

print("-" * 80)

for r in results:

    print(
        f"{r['nms']:.2f}    | "
        f"{r['tp']:2d} | "
        f"{r['fp']:3d} | "
        f"{r['fn']:2d} | "
        f"{r['precision']:.4f}    | "
        f"{r['recall']:.4f} | "
        f"{r['f1']:.4f}"
    )


# ------------------------------------------------------------
# Best F1
# ------------------------------------------------------------

best = max(results, key=lambda x: x["f1"])

print()
print("=" * 80)
print("BEST NMS SETTING")
print("=" * 80)

print(f"NMS IoU    : {best['nms']:.2f}")
print(f"TP         : {best['tp']}")
print(f"FP         : {best['fp']}")
print(f"FN         : {best['fn']}")
print(f"Precision  : {best['precision']:.6f}")
print(f"Recall     : {best['recall']:.6f}")
print(f"F1         : {best['f1']:.6f}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)