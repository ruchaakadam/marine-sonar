from pathlib import Path
from PIL import Image, ImageEnhance
import shutil
import random

SRC = Path("yolo_dataset_groupval")
DST = Path("yolo_dataset_groupval_aug")

random.seed(42)


def read_labels(path):
    if not path.exists():
        return []

    lines = []
    for line in path.read_text().splitlines():
        if line.strip():
            parts = line.split()
            if len(parts) == 5:
                lines.append([float(x) for x in parts])

    return lines


def write_labels(path, labels):
    with open(path, "w") as f:
        for cls, x, y, w, h in labels:
            f.write(
                f"{int(cls)} {x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )


def horizontal_flip(labels):
    result = []

    for cls, x, y, w, h in labels:
        x = 1.0 - x
        result.append([cls, x, y, w, h])

    return result


def make_brightness_contrast(img):
    # Conservative sonar augmentation
    brightness = random.uniform(0.90, 1.10)
    contrast = random.uniform(0.90, 1.15)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    return img


# Clean previous output
if DST.exists():
    shutil.rmtree(DST)

# ---------------------------------------------------------
# Copy TRAIN originals
# ---------------------------------------------------------

src_img = SRC / "train" / "images"
src_lbl = SRC / "train" / "labels"

dst_img = DST / "train" / "images"
dst_lbl = DST / "train" / "labels"

dst_img.mkdir(parents=True, exist_ok=True)
dst_lbl.mkdir(parents=True, exist_ok=True)

original_images = 0
original_boxes = 0
positive_images = []

for img_path in sorted(src_img.glob("*.png")):

    label_path = src_lbl / f"{img_path.stem}.txt"

    shutil.copy2(img_path, dst_img / img_path.name)

    if label_path.exists():
        shutil.copy2(label_path, dst_lbl / label_path.name)

        labels = read_labels(label_path)

        if labels:
            positive_images.append((img_path, labels))
            original_boxes += len(labels)

    original_images += 1


# ---------------------------------------------------------
# Augment POSITIVE tiles only
# ---------------------------------------------------------

aug_images = 0
aug_boxes = 0

for img_path, labels in positive_images:

    img = Image.open(img_path).convert("L")

    # -------------------------------
    # AUGMENTATION 1: horizontal flip
    # -------------------------------

    flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flipped_name = f"{img_path.stem}_augflip.png"

    flipped.save(dst_img / flipped_name)

    flipped_labels = horizontal_flip(labels)

    write_labels(
        dst_lbl / f"{img_path.stem}_augflip.txt",
        flipped_labels
    )

    aug_images += 1
    aug_boxes += len(flipped_labels)

    # --------------------------------
    # AUGMENTATION 2: brightness/contrast
    # --------------------------------

    enhanced = make_brightness_contrast(img)
    enhanced_name = f"{img_path.stem}_augcontrast.png"

    enhanced.save(dst_img / enhanced_name)

    write_labels(
        dst_lbl / f"{img_path.stem}_augcontrast.txt",
        labels
    )

    aug_images += 1
    aug_boxes += len(labels)


# ---------------------------------------------------------
# Copy VAL unchanged
# ---------------------------------------------------------

for split in ["val", "test"]:

    s_img = SRC / split / "images"
    s_lbl = SRC / split / "labels"

    d_img = DST / split / "images"
    d_lbl = DST / split / "labels"

    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)

    for img in s_img.glob("*.png"):
        shutil.copy2(img, d_img / img.name)

    for lbl in s_lbl.glob("*.txt"):
        shutil.copy2(lbl, d_lbl / lbl.name)


# ---------------------------------------------------------
# data.yaml
# ---------------------------------------------------------

yaml = """path: D:/sih/marine-sonar/yolo_dataset_groupval_aug

train: train/images
val: val/images
test: test/images

names:
  0: target
"""

(DST / "data.yaml").write_text(yaml)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

total_train = len(list(dst_img.glob("*.png")))

print()
print("=" * 60)
print("AUGMENTED GROUP-VALIDATION DATASET")
print("=" * 60)

print()
print("ORIGINAL TRAIN IMAGES:", original_images)
print("ORIGINAL POSITIVE IMAGES:", len(positive_images))
print("ORIGINAL BOXES:", original_boxes)

print()
print("AUGMENTED IMAGES:", aug_images)
print("AUGMENTED BOXES:", aug_boxes)

print()
print("FINAL TRAIN IMAGES:", total_train)
print("FINAL TRAIN POSITIVE TILES:", len(positive_images) * 3)
print("FINAL TRAIN BOXES:", original_boxes + aug_boxes)

print()
print("VAL IMAGES:", len(list((DST / "val" / "images").glob("*.png"))))
print("TEST IMAGES:", len(list((DST / "test" / "images").glob("*.png"))))

print()
print("Created:")
print(DST.resolve())