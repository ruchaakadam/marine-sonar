from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

root = Path("yolo_dataset_groupval")
img_dir = root / "train" / "images"
lbl_dir = root / "train" / "labels"

images = sorted(img_dir.glob("*.png"))[:20]

thumb_w = 300
thumb_h = 220
cols = 4
rows = math.ceil(len(images) / cols)

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "black")

for i, img_path in enumerate(images):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    draw = ImageDraw.Draw(img)

    label_path = lbl_dir / (img_path.stem + ".txt")

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            draw.rectangle((x1, y1, x2, y2), outline="red", width=4)

    img.thumbnail((thumb_w - 10, thumb_h - 25))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(img, (x + 5, y + 20))

    ImageDraw.Draw(sheet).text(
        (x + 5, y + 2),
        img_path.stem,
        fill="white"
    )

out = Path("train_label_check.jpg")
sheet.save(out, quality=95)

print("Created:", out.resolve())
print("Images checked:", len(images))
