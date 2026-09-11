import sqlite3
from pathlib import Path

import rasterio
from rasterio.windows import Window
from shapely import wkb
from PIL import Image
import numpy as np


# ============================================================
# PATHS
# ============================================================

S3 = Path(r"C:\Users\Aarya\Downloads\boulders\S3")
S2 = Path(r"C:\Users\Aarya\Downloads\boulders\S2")

OUTPUT = Path(r"C:\Users\Aarya\marine-sonar\rock_dataset")

SQLITE = S3 / "Training_stones.sqlite"

TIFFS = [
    S2 / "East_Training_Area.tif",
    S2 / "West_Training_Area.tif",
    S2 / "Central_Training_Area.tif",
]


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 512
ROCK_CLASS_ID = 0


def save_patch(src, geometry, index):
    minx, miny, maxx, maxy = geometry.bounds

    # Convert geographic coordinates to pixel coordinates
    row1, col1 = src.index(minx, maxy)
    row2, col2 = src.index(maxx, miny)

    center_col = int((col1 + col2) / 2)
    center_row = int((row1 + row2) / 2)

    half = PATCH_SIZE // 2

    col_off = max(0, center_col - half)
    row_off = max(0, center_row - half)

    col_off = min(col_off, src.width - PATCH_SIZE)
    row_off = min(row_off, src.height - PATCH_SIZE)

    window = Window(
        col_off,
        row_off,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    image = src.read(1, window=window)

    # Convert sonar raster to 8-bit image
    image = image.astype(np.float32)

    if image.max() > image.min():
        image = (
            (image - image.min())
            / (image.max() - image.min())
            * 255
        )

    image = image.astype(np.uint8)

    image_name = f"rock_{index:05d}.png"

    image_path = OUTPUT / "images" / image_name
    label_path = OUTPUT / "labels" / f"rock_{index:05d}.txt"

    Image.fromarray(image).save(image_path)

    # Convert rock geographic bounds to pixel coordinates
    x1 = (col1 - col_off)
    y1 = (row1 - row_off)
    x2 = (col2 - col_off)
    y2 = (row2 - row_off)

    # Clamp to patch
    x1 = max(0, min(PATCH_SIZE, x1))
    y1 = max(0, min(PATCH_SIZE, y1))
    x2 = max(0, min(PATCH_SIZE, x2))
    y2 = max(0, min(PATCH_SIZE, y2))

    width = x2 - x1
    height = y2 - y1

    if width <= 1 or height <= 1:
        return False

    center_x = (x1 + x2) / 2 / PATCH_SIZE
    center_y = (y1 + y2) / 2 / PATCH_SIZE
    norm_w = width / PATCH_SIZE
    norm_h = height / PATCH_SIZE

    with open(label_path, "w") as f:
        f.write(
            f"{ROCK_CLASS_ID} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{norm_w:.6f} "
            f"{norm_h:.6f}\n"
        )

    return True


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

(OUTPUT / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT / "labels").mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

conn = sqlite3.connect(SQLITE)

rows = conn.execute(
    """
    SELECT GEOMETRY
    FROM training_stones
    """
).fetchall()

conn.close()

print(f"Found {len(rows)} rock annotations.")


# ============================================================
# PROCESS ANNOTATIONS
# ============================================================

count = 0

opened_tiffs = []

for tif in TIFFS:
    print(f"Opening {tif.name}")
    opened_tiffs.append(rasterio.open(tif))


for row in rows:

    geometry = wkb.loads(row[0])

    minx, miny, maxx, maxy = geometry.bounds

    # Find the training raster containing this rock
    for src in opened_tiffs:

        if not (
            src.bounds.left <= minx <= src.bounds.right
            and src.bounds.left <= maxx <= src.bounds.right
            and src.bounds.bottom <= miny <= src.bounds.top
            and src.bounds.bottom <= maxy <= src.bounds.top
        ):
            continue

        try:
            if save_patch(src, geometry, count):
                count += 1
        except Exception as e:
            print(f"Skipped annotation: {e}")

        break


for src in opened_tiffs:
    src.close()


print()
print("==========================================")
print("ROCK EXTRACTION COMPLETE")
print("==========================================")
print(f"Rock samples created: {count}")
print(f"Output: {OUTPUT}")