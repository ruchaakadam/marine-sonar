from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("yolo_dataset_groupval_tiles_full")
OUT = Path("gt_check")

OUT.mkdir(exist_ok=True)

images = list((ROOT / "train" / "images").glob("*.png"))

count = 0

for img_path in images:
    label_path = ROOT / "train" / "labels" / f"{img_path.stem}.txt"

    if not label_path.exists():
        continue

    text = label_path.read_text().strip()

    if not text:
        continue

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    W, H = img.size

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, w, h = map(float, parts)

        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=6
        )

    img.save(OUT / img_path.name)

    count += 1

    if count >= 10:
        break

print("Created:", OUT.resolve())
print("Images:", count)