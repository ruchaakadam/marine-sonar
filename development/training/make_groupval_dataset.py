from pathlib import Path
import shutil
import re

SRC = Path("yolo_dataset")
DST = Path("yolo_dataset_groupval")

# Entire groups held out for validation
VAL_GROUPS = {
    "DR_Hanna",
    "Isaac_M_Scott",
    "Montana",
    "Pewabic",
}


def get_group(stem):
    # Remove tile suffix, e.g. _tile02
    stem = re.sub(r"_tile\d+$", "", stem)

    # Remove image-number suffix, e.g. _03
    stem = re.sub(r"_\d+$", "", stem)

    return stem


def copy_split(src_split, dst_split, allowed_groups):
    src_img = SRC / src_split / "images"
    src_lbl = SRC / src_split / "labels"

    dst_img = DST / dst_split / "images"
    dst_lbl = DST / dst_split / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_boxes = 0

    for img in sorted(src_img.glob("*.png")):
        group = get_group(img.stem)

        if group not in allowed_groups:
            continue

        label = src_lbl / f"{img.stem}.txt"

        shutil.copy2(img, dst_img / img.name)

        if label.exists():
            shutil.copy2(label, dst_lbl / label.name)

            lines = [
                x for x in label.read_text().splitlines()
                if x.strip()
            ]
            copied_boxes += len(lines)

        copied_images += 1

    return copied_images, copied_boxes


# All groups appearing in original train
all_groups = set()

for p in (SRC / "train" / "images").glob("*.png"):
    all_groups.add(get_group(p.stem))

train_groups = all_groups - VAL_GROUPS

print("=" * 60)
print("GROUP-LEVEL TRAIN / VAL DATASET")
print("=" * 60)

print("\nValidation groups:")
for g in sorted(VAL_GROUPS):
    print(" ", g)

print("\nTraining groups:")
for g in sorted(train_groups):
    print(" ", g)

# Clean destination
if DST.exists():
    shutil.rmtree(DST)

# Create group-separated train/val
train_images, train_boxes = copy_split(
    "train",
    "train",
    train_groups
)

val_images, val_boxes = copy_split(
    "train",
    "val",
    VAL_GROUPS
)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print("\nTRAIN")
print("Images:", train_images)
print("Boxes:", train_boxes)

print("\nVAL")
print("Images:", val_images)
print("Boxes:", val_boxes)

# Copy original test unchanged
test_src_img = SRC / "test" / "images"
test_src_lbl = SRC / "test" / "labels"

test_dst_img = DST / "test" / "images"
test_dst_lbl = DST / "test" / "labels"

test_dst_img.mkdir(parents=True, exist_ok=True)
test_dst_lbl.mkdir(parents=True, exist_ok=True)

test_images = 0
test_boxes = 0

for img in test_src_img.glob("*.png"):
    shutil.copy2(img, test_dst_img / img.name)

    label = test_src_lbl / f"{img.stem}.txt"

    if label.exists():
        shutil.copy2(label, test_dst_lbl / label.name)
        test_boxes += len([
            x for x in label.read_text().splitlines()
            if x.strip()
        ])

    test_images += 1

print("\nTEST")
print("Images:", test_images)
print("Boxes:", test_boxes)

# Create data.yaml
yaml = """path: D:/sih/marine-sonar/yolo_dataset_groupval

train: train/images
val: val/images
test: test/images

names:
  0: target
"""

(DST / "data.yaml").write_text(yaml)

print("\nCreated:")
print(DST.resolve())
print(DST / "data.yaml")