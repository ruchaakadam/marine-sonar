from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

MODEL = Path(
    "runs/detect/runs/detect/runs/tile_balanced_aug/weights/best.pt"
)

SRC = Path("yolo_dataset_groupval")

POS = Path("yolo_dataset_groupval_tiles_positive")

DST = Path("yolo_dataset_groupval_tiles_hardneg")

IMGSZ = 640
CONF = 0.05
IOU = 0.50
DEVICE = 0

# Size of hard-negative crop
CROP_W = 864
CROP_H = 1728

MAX_TRAIN_NEG = 150
MAX_VAL_NEG = 30

random.seed(42)

model = YOLO(str(MODEL))


# ============================================================
# LOAD GT
# ============================================================

def load_gt(label_path, W, H):

    boxes = []

    if not label_path.exists():
        return boxes

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

        boxes.append(
            [x1, y1, x2, y2]
        )

    return boxes


# ============================================================
# IOU
# ============================================================

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

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0

    return inter / union


# ============================================================
# CHECK IF CROP CONTAINS REAL OBJECT
# ============================================================

def crop_overlaps_gt(
    crop,
    gt_boxes
):

    for gt in gt_boxes:

        if iou(crop, gt) > 0.05:
            return True

    return False


# ============================================================
# MINE HARD NEGATIVES
# ============================================================

def mine(split, maximum):

    image_dir = (
        SRC /
        split /
        "images"
    )

    label_dir = (
        SRC /
        split /
        "labels"
    )

    images = sorted(
        image_dir.glob("*.png")
    )

    candidates = []

    print()
    print("=" * 70)
    print(
        f"MINING {split.upper()} HARD NEGATIVES"
    )
    print("=" * 70)

    print(
        "Images:",
        len(images)
    )

    for n, img_path in enumerate(
        images,
        1
    ):

        img = Image.open(
            img_path
        ).convert("RGB")

        W, H = img.size

        label_path = (
            label_dir /
            f"{img_path.stem}.txt"
        )

        gt_boxes = load_gt(
            label_path,
            W,
            H
        )

        # Run model on original image
        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            continue

        for box, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist()
        ):

            x1, y1, x2, y2 = box

            score = float(score)

            # --------------------------------------------
            # Skip detections that overlap real GT.
            # --------------------------------------------

            is_real = False

            for gt in gt_boxes:

                if iou(
                    [x1, y1, x2, y2],
                    gt
                ) >= 0.10:

                    is_real = True
                    break

            if is_real:
                continue

            # --------------------------------------------
            # Center of false detection
            # --------------------------------------------

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # --------------------------------------------
            # Make crop around false detection
            # --------------------------------------------

            left = int(
                cx - CROP_W / 2
            )

            top = int(
                cy - CROP_H / 2
            )

            left = max(
                0,
                min(
                    left,
                    W - CROP_W
                )
            )

            top = max(
                0,
                min(
                    top,
                    H - CROP_H
                )
            )

            crop = [
                left,
                top,
                left + CROP_W,
                top + CROP_H
            ]

            # --------------------------------------------
            # Make absolutely sure crop has no GT
            # --------------------------------------------

            if crop_overlaps_gt(
                crop,
                gt_boxes
            ):
                continue

            candidates.append(
                (
                    img_path,
                    left,
                    top,
                    score
                )
            )

        if n % 10 == 0:

            print(
                f"Processed "
                f"{n}/{len(images)} "
                f"candidates={len(candidates)}"
            )

    # Highest confidence first
    candidates.sort(
        key=lambda x: x[3],
        reverse=True
    )

    # Remove duplicate crops
    unique = []
    seen = set()

    for item in candidates:

        img_path, left, top, score = item

        key = (
            img_path.name,
            left,
            top
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    candidates = unique[:maximum]

    print()
    print(
        "Hard-negative candidates:",
        len(candidates)
    )

    return candidates


# ============================================================
# SAVE HARD NEGATIVES
# ============================================================

def save_negatives(
    split,
    candidates
):

    out_img = (
        DST /
        split /
        "images"
    )

    out_lbl = (
        DST /
        split /
        "labels"
    )

    out_img.mkdir(
        parents=True,
        exist_ok=True
    )

    out_lbl.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    for (
        img_path,
        left,
        top,
        score
    ) in candidates:

        img = Image.open(
            img_path
        ).convert("RGB")

        crop = img.crop(
            (
                left,
                top,
                left + CROP_W,
                top + CROP_H
            )
        )

        stem = (
            f"HN_{count:04d}_"
            f"{img_path.stem}"
        )

        crop.save(
            out_img /
            f"{stem}.png"
        )

        # Empty YOLO label = background
        (
            out_lbl /
            f"{stem}.txt"
        ).write_text("")

        count += 1

    print(
        f"Saved {count} hard negatives"
    )

    return count


# ============================================================
# COPY POSITIVE TILES
# ============================================================

def copy_positive():

    for split in [
        "train",
        "val"
    ]:

        src_img = (
            POS /
            split /
            "images"
        )

        src_lbl = (
            POS /
            split /
            "labels"
        )

        dst_img = (
            DST /
            split /
            "images"
        )

        dst_lbl = (
            DST /
            split /
            "labels"
        )

        dst_img.mkdir(
            parents=True,
            exist_ok=True
        )

        dst_lbl.mkdir(
            parents=True,
            exist_ok=True
        )

        count = 0

        for img in src_img.glob("*.png"):

            shutil.copy2(
                img,
                dst_img / img.name
            )

            label = (
                src_lbl /
                f"{img.stem}.txt"
            )

            if label.exists():

                shutil.copy2(
                    label,
                    dst_lbl / label.name
                )

            count += 1

        print(
            split.upper(),
            "positive:",
            count
        )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("HARD NEGATIVE MINING V2")
print("=" * 70)

if DST.exists():

    shutil.rmtree(DST)

copy_positive()

train_candidates = mine(
    "train",
    MAX_TRAIN_NEG
)

save_negatives(
    "train",
    train_candidates
)

val_candidates = mine(
    "val",
    MAX_VAL_NEG
)

save_negatives(
    "val",
    val_candidates
)


# ============================================================
# FINAL COUNTS
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

for split in [
    "train",
    "val"
]:

    img_dir = (
        DST /
        split /
        "images"
    )

    lbl_dir = (
        DST /
        split /
        "labels"
    )

    images = list(
        img_dir.glob("*.png")
    )

    labels = list(
        lbl_dir.glob("*.txt")
    )

    backgrounds = 0

    for label in labels:

        if not label.read_text().strip():
            backgrounds += 1

    positives = (
        len(labels) -
        backgrounds
    )

    print()
    print(split.upper())
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Positive:", positives)
    print("Background:", backgrounds)

print()
print("Created:")
print(DST.resolve())
