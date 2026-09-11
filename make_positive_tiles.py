from pathlib import Path
from PIL import Image

SRC = Path("yolo_dataset_groupval_tiles_full")
DST = Path("yolo_dataset_groupval_tiles_positive")

for split in ["train", "val"]:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"

    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count_img = 0
    count_box = 0

    for label in sorted(src_lbl.glob("*.txt")):
        text = label.read_text().strip()

        # Skip background tiles
        if not text:
            continue

        img = src_img / f"{label.stem}.png"

        if not img.exists():
            print("Missing image:", img)
            continue

        (dst_lbl / label.name).write_text(text)
        Image.open(img).save(dst_img / img.name)

        count_img += 1
        count_box += len(text.splitlines())

    print()
    print(split.upper())
    print("Positive tiles:", count_img)
    print("Boxes:", count_box)

print()
print("Created:", DST.resolve())