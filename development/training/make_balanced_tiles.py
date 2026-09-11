from pathlib import Path
import shutil
import random

SRC = Path("yolo_dataset_groupval_tiles_full")
DST = Path("yolo_dataset_groupval_tiles_balanced")

random.seed(42)

for split in ["train", "val"]:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"

    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    positive = []

    for lbl in src_lbl.glob("*.txt"):
        lines = [x for x in lbl.read_text().splitlines() if x.strip()]

        if lines:
            positive.append(lbl)

    # Copy every positive tile
    for lbl in positive:
        img = src_img / f"{lbl.stem}.png"

        if img.exists():
            shutil.copy2(img, dst_img / img.name)
            shutil.copy2(lbl, dst_lbl / lbl.name)

    # Background tiles
    backgrounds = []

    for lbl in src_lbl.glob("*.txt"):
        lines = [x for x in lbl.read_text().splitlines() if x.strip()]

        if not lines:
            img = src_img / f"{lbl.stem}.png"

            if img.exists():
                backgrounds.append((img, lbl))

    # Train: equal number of backgrounds and positives
    # Val: keep all positive tiles, no backgrounds
    if split == "train":
        random.shuffle(backgrounds)
        backgrounds = backgrounds[:len(positive)]

    else:
        backgrounds = []

    for img, lbl in backgrounds:
        shutil.copy2(img, dst_img / img.name)
        shutil.copy2(lbl, dst_lbl / lbl.name)

    print()
    print(split.upper())
    print("Positive tiles:", len(positive))
    print("Background tiles:", len(backgrounds))
    print("Total tiles:", len(list(dst_img.glob("*.png"))))