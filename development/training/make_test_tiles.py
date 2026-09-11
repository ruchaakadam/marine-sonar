from pathlib import Path
from PIL import Image

SRC = Path("yolo_dataset/test")
DST = Path("yolo_dataset_groupval_tiles_full/test")

TILE_H = 1728
OVERLAP = 432
STRIDE = TILE_H - OVERLAP

src_img = SRC / "images"
src_lbl = SRC / "labels"

dst_img = DST / "images"
dst_lbl = DST / "labels"

dst_img.mkdir(parents=True, exist_ok=True)
dst_lbl.mkdir(parents=True, exist_ok=True)

total_images = 0
total_boxes = 0
background_tiles = 0

for img_path in sorted(src_img.glob("*.png")):

    label_path = src_lbl / f"{img_path.stem}.txt"

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    labels = []

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            # Original image pixel coordinates
            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H

            labels.append((int(cls), x1, y1, x2, y2))

    tile_num = 0
    y0 = 0

    while True:

        y1 = min(y0 + TILE_H, H)
        tile = img.crop((0, y0, W, y1))

        new_labels = []

        for cls, box_x1, box_y1, box_x2, box_y2 in labels:

            ix1 = max(box_x1, 0)
            ix2 = min(box_x2, W)

            iy1 = max(box_y1, y0)
            iy2 = min(box_y2, y1)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)

            original_area = max(
                1,
                (box_x2 - box_x1) * (box_y2 - box_y1)
            )

            intersection = iw * ih

            # Same 30% rule as training/validation tiling
            if intersection / original_area < 0.30:
                continue

            tx1 = ix1
            tx2 = ix2
            ty1 = iy1 - y0
            ty2 = iy2 - y0

            tile_w, tile_h = tile.size

            new_xc = ((tx1 + tx2) / 2) / tile_w
            new_yc = ((ty1 + ty2) / 2) / tile_h
            new_bw = (tx2 - tx1) / tile_w
            new_bh = (ty2 - ty1) / tile_h

            new_labels.append(
                f"{cls} "
                f"{new_xc:.6f} "
                f"{new_yc:.6f} "
                f"{new_bw:.6f} "
                f"{new_bh:.6f}"
            )

        stem = f"{img_path.stem}_tile{tile_num:02d}"

        tile.save(dst_img / f"{stem}.png")

        (dst_lbl / f"{stem}.txt").write_text(
            "\n".join(new_labels) + ("\n" if new_labels else "")
        )

        total_images += 1
        total_boxes += len(new_labels)

        if not new_labels:
            background_tiles += 1

        tile_num += 1

        if y1 >= H:
            break

        y0 += STRIDE

print()
print("TEST")
print("Total tiles:", total_images)
print("Total boxes:", total_boxes)
print("Background tiles:", background_tiles)

print()
print("Created:")
print(DST.resolve())