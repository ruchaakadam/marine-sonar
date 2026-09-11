from pathlib import Path
from PIL import Image

SRC = Path("yolo_dataset_groupval")
DST = Path("yolo_dataset_groupval_tiles_full")

TILE_H = 1728
OVERLAP = 432
STRIDE = TILE_H - OVERLAP

for split in ["train", "val", "test"]:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"

    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

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

                x1 = (xc - bw / 2) * W
                y1 = (yc - bh / 2) * H
                x2 = (xc + bw / 2) * W
                y2 = (yc + bh / 2) * H

                labels.append((cls, x1, y1, x2, y2))

        tile_num = 0
        y0 = 0

        while y0 < H:

            y1 = min(y0 + TILE_H, H)

            if y1 - y0 >= TILE_H * 0.5:

                tile = img.crop((0, y0, W, y1))

                new_labels = []

                for cls, x1, box_y1, x2, box_y2 in labels:

                    ix1 = max(x1, 0)
                    ix2 = min(x2, W)

                    iy1 = max(box_y1, y0)
                    iy2 = min(box_y2, y1)

                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)

                    original_area = max(
                        1,
                        (x2 - x1) * (box_y2 - box_y1)
                    )

                    intersection = iw * ih

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
                        f"{int(cls)} "
                        f"{new_xc:.6f} "
                        f"{new_yc:.6f} "
                        f"{new_bw:.6f} "
                        f"{new_bh:.6f}"
                    )

                stem = f"{img_path.stem}_tile{tile_num:02d}"

                # SAVE EVERY TILE
                tile.save(dst_img / f"{stem}.png")

                # Empty label file = background tile
                (dst_lbl / f"{stem}.txt").write_text(
                    "\n".join(new_labels) +
                    ("\n" if new_labels else "")
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
    print(split.upper())
    print("Total tiles:", total_images)
    print("Total boxes:", total_boxes)
    print("Background tiles:", background_tiles)

print()
print("Created:")
print(DST.resolve())