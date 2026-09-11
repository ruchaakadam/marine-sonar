from pathlib import Path
from PIL import Image
import statistics

root = Path("yolo_dataset_groupval")
label_dir = root / "train" / "labels"
image_dir = root / "train" / "images"

areas = []
widths = []
heights = []

for p in sorted(label_dir.glob("*.txt")):
    for line in p.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        widths.append(bw)
        heights.append(bh)
        areas.append(bw * bh)

print("TRAIN LABEL FILES:", len(list(label_dir.glob("*.txt"))))
print("OBJECTS:", len(areas))

img = next(image_dir.glob("*.png"))
print("IMAGE SIZE:", Image.open(img).size)

print("MEAN WIDTH:", round(statistics.mean(widths), 4))
print("MEAN HEIGHT:", round(statistics.mean(heights), 4))
print("MEAN AREA:", round(statistics.mean(areas), 4))
print("MEDIAN AREA:", round(statistics.median(areas), 4))

print("AREA < 0.05:", sum(a < 0.05 for a in areas))
print("AREA < 0.10:", sum(a < 0.10 for a in areas))
print("AREA < 0.20:", sum(a < 0.20 for a in areas))
print("AREA >= 0.25:", sum(a >= 0.25 for a in areas))