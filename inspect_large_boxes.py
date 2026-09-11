from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path("yolo_dataset_groupval")
img_dir = root / "train" / "images"
lbl_dir = root / "train" / "labels"

items = []

for label_path in sorted(lbl_dir.glob("*.txt")):
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)
        area = bw * bh

        if area > 0.25:
            img_path = img_dir / (label_path.stem + ".png")
            if img_path.exists():
                items.append((img_path, xc, yc, bw, bh, area))

thumb_w = 360
thumb_h = 280
cols = 4
rows = math.ceil(len(items) / cols)

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "black")

for i, (img_path, xc, yc, bw, bh, area) in enumerate(items):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    draw = ImageDraw.Draw(img)

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    draw.rectangle((x1, y1, x2, y2), outline="red", width=5)

    img.thumbnail((thumb_w - 10, thumb_h - 40))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(img, (x + 5, y + 35))

    ImageDraw.Draw(sheet).text(
        (x + 5, y + 5),
        f"{img_path.stem}  area={area:.3f}",
        fill="white"
    )

out = Path("large_box_review.jpg")
sheet.save(out, quality=95)

print("Created:", out.resolve())
print("Large boxes:", len(items))