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

# Existing positive-tile dataset
POS = Path("yolo_dataset_groupval_tiles_positive")

# New dataset
DST = Path("yolo_dataset_groupval_tiles_hardneg")

TILE_SIZE = 1728
OVERLAP = 432
STRIDE = TILE_SIZE - OVERLAP

IMGSZ = 640
PRED_CONF = 0.05
PRED_IOU = 0.50
DEVICE = 0

MAX_TRAIN_NEG = 150
MAX_VAL_NEG = 30

MIN_FALSE_AREA = 100
MAX_FALSE_AREA = 500000

random.seed(42)

model = YOLO(str(MODEL))


# ============================================================
# HELPERS
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


def intersection_over_area(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        1,
        (ax2 - ax1) * (ay2 - ay1)
    )

    return intersection / area_a


def overlaps_gt(pred_box, gt_boxes):

    for gt in gt_boxes:

        if intersection_over_area(
            pred_box,
            gt
        ) > 0.10:

            return True

    return False


def tile_has_gt(
    tile_x1,
    tile_y1,
    tile_x2,
    tile_y2,
    gt_boxes
):

    tile = [
        tile_x1,
        tile_y1,
        tile_x2,
        tile_y2
    ]

    for gt in gt_boxes:

        # intersection between tile and GT
        ix1 = max(tile[0], gt[0])
        iy1 = max(tile[1], gt[1])
        ix2 = min(tile[2], gt[2])
        iy2 = min(tile[3], gt[3])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        if iw * ih > 0:
            return True

    return False


def find_false_positive_tiles(
    split,
    maximum
):

    src_images = (
        SRC /
        split /
        "images"
    )

    src_labels = (
        SRC /
        split /
        "labels"
    )

    candidates = []

    images = sorted(
        src_images.glob("*.png")
    )

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

    for image_number, img_path in enumerate(
        images,
        1
    ):

        img = Image.open(
            img_path
        ).convert("RGB")

        W, H = img.size

        label_path = (
            src_labels /
            f"{img_path.stem}.txt"
        )

        gt_boxes = load_gt(
            label_path,
            W,
            H
        )

        y0 = 0

        while y0 < H:

            if H <= TILE_SIZE:
                y_start = 0
            elif y0 + TILE_SIZE >= H:
                y_start = H - TILE_SIZE
            else:
                y_start = y0

            y_end = min(
                y_start + TILE_SIZE,
                H
            )

            # ------------------------------------------------
            # We only want completely background tiles.
            # ------------------------------------------------

            if not tile_has_gt(
                0,
                y_start,
                W,
                y_end,
                gt_boxes
            ):

                tile = img.crop(
                    (
                        0,
                        y_start,
                        W,
                        y_end
                    )
                )

                results = model.predict(
                    source=tile,
                    imgsz=IMGSZ,
                    conf=PRED_CONF,
                    iou=PRED_IOU,
                    device=DEVICE,
                    verbose=False
                )

                result = results[0]

                if result.boxes is not None:

                    for box, conf in zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist()
                    ):

                        x1, y1, x2, y2 = box

                        area = (
                            max(0, x2 - x1)
                            *
                            max(0, y2 - y1)
                        )

                        if (
                            area >= MIN_FALSE_AREA
                            and
                            area <= MAX_FALSE_AREA
                        ):

                            candidates.append(
                                (
                                    img_path,
                                    y_start,
                                    float(conf)
                                )
                            )

                            break

            if y_end >= H:
                break

            y0 += STRIDE

        if image_number % 10 == 0:
            print(
                f"Processed "
                f"{image_number}/{len(images)} "
                f"candidates={len(candidates)}"
            )

    # Remove duplicate source/tile combinations
    unique = {}

    for img_path, y_start, conf in candidates:

        key = (
            img_path.name,
            y_start
        )

        if key not in unique:
            unique[key] = (
                img_path,
                y_start,
                conf
            )

    candidates = list(
        unique.values()
    )

    # Highest-confidence false positives first
    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    candidates = candidates[:maximum]

    print()
    print(
        "Hard-negative candidates:",
        len(candidates)
    )

    return candidates


def save_hard_negatives(
    split,
    candidates
):

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

    for img_path, y_start, conf in candidates:

        img = Image.open(
            img_path
        ).convert("RGB")

        W, H = img.size

        tile = img.crop(
            (
                0,
                y_start,
                W,
                min(
                    y_start + TILE_SIZE,
                    H
                )
            )
        )

        # Make sure tile is exactly TILE_SIZE high.
        # Pad if necessary.
        if tile.height < TILE_SIZE:

            padded = Image.new(
                "RGB",
                (
                    TILE_SIZE,
                    TILE_SIZE
                )
            )

            padded.paste(
                tile,
                (0, 0)
            )

            tile = padded

        stem = (
            f"HN_{count:04d}_"
            f"{img_path.stem}"
        )

        image_out = (
            dst_img /
            f"{stem}.png"
        )

        label_out = (
            dst_lbl /
            f"{stem}.txt"
        )

        tile.save(
            image_out
        )

        # EMPTY YOLO LABEL = BACKGROUND
        label_out.write_text("")

        count += 1

    print(
        f"Saved {count} hard-negative "
        f"tiles to {dst_img}"
    )

    return count


# ============================================================
# COPY POSITIVE DATA
# ============================================================

def copy_positive_data():

    print()
    print("=" * 70)
    print("COPYING POSITIVE TILES")
    print("=" * 70)

    for split in ["train", "val"]:

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

        for p in src_img.glob("*.png"):

            shutil.copy2(
                p,
                dst_img / p.name
            )

            label = (
                src_lbl /
                f"{p.stem}.txt"
            )

            if label.exists():

                shutil.copy2(
                    label,
                    dst_lbl / label.name
                )

            else:

                (
                    dst_lbl /
                    f"{p.stem}.txt"
                ).write_text("")

            count += 1

        print(
            split.upper(),
            "positive tiles:",
            count
        )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("HARD NEGATIVE DATASET BUILDER")
print("=" * 70)

if not MODEL.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL}"
    )

if not SRC.exists():

    raise FileNotFoundError(
        f"Source dataset not found:\n{SRC}"
    )

if not POS.exists():

    raise FileNotFoundError(
        f"Positive dataset not found:\n{POS}"
    )


# Start clean
if DST.exists():

    shutil.rmtree(DST)


# Copy the 68/22 positive tiles
copy_positive_data()


# Mine hard negatives from TRAIN
train_candidates = find_false_positive_tiles(
    "train",
    MAX_TRAIN_NEG
)

save_hard_negatives(
    "train",
    train_candidates
)


# Mine hard negatives from VAL
val_candidates = find_false_positive_tiles(
    "val",
    MAX_VAL_NEG
)

save_hard_negatives(
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

for split in ["train", "val"]:

    images = list(
        (
            DST /
            split /
            "images"
        ).glob("*.png")
    )

    labels = list(
        (
            DST /
            split /
            "labels"
        ).glob("*.txt")
    )

    background = 0

    for label in labels:

        if not label.read_text().strip():
            background += 1

    print()
    print(split.upper())
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Background:", background)
    print(
        "Positive:",
        len(images) - background
    )


print()
print("Created:")
print(DST.resolve())