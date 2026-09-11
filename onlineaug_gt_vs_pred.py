from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GT = Path("yolo_dataset_groupval/test/labels")
IMG = Path("yolo_dataset_groupval/test/images")
PR = Path(r"runs/detect/runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels")

OUT = Path(r"runs/FINAL_RESULTS/onlineaug_gt_vs_pred")
OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.05
IOU_T = 0.50

def read_boxes(path, pred=False):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue

        cls = int(float(p[0]))
        x, y, w, h = map(float, p[1:5])

        if pred:
            if len(p) < 6:
                continue
            conf = float(p[5])
            if conf < CONF:
                continue
        else:
            conf = 1.0

        boxes.append((cls, x, y, w, h, conf))

    return boxes

def iou(a, b):
    ax1 = a[1] - a[3]/2
    ay1 = a[2] - a[4]/2
    ax2 = a[1] + a[3]/2
    ay2 = a[2] + a[4]/2

    bx1 = b[1] - b[3]/2
    by1 = b[2] - b[4]/2
    bx2 = b[1] + b[3]/2
    by2 = b[2] + b[4]/2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2-ix1) * max(0, iy2-iy1)

    aa = max(0, ax2-ax1) * max(0, ay2-ay1)
    ab = max(0, bx2-bx1) * max(0, by2-by1)

    union = aa + ab - inter

    return inter / union if union > 0 else 0

def find_image(stem):
    for ext in [".png",".jpg",".jpeg",".JPG",".PNG",".JPEG"]:
        p = IMG / (stem + ext)
        if p.exists():
            return p
    return None

records = []

for gt_file in GT.glob("*.txt"):

    gt = read_boxes(gt_file)
    pr = read_boxes(PR / gt_file.name, pred=True)

    used = set()
    tp = 0
    fp = 0

    for pi, pb in sorted(
        enumerate(pr),
        key=lambda z: z[1][5],
        reverse=True
    ):
        matches = []

        for gi, gb in enumerate(gt):
            if gi in used:
                continue
            if pb[0] != gb[0]:
                continue

            v = iou(pb, gb)

            if v >= IOU_T:
                matches.append((v, gi))

        if matches:
            _, gi = max(matches)
            used.add(gi)
            tp += 1
        else:
            fp += 1

    fn = len(gt) - len(used)

    records.append((fp + fn, fp, fn, tp, gt_file))

records.sort(reverse=True)

print("=" * 80)
print("CREATING GT VS PREDICTION DIAGNOSTIC GALLERY")
print(f"Confidence = {CONF}")
print(f"IoU = {IOU_T}")
print("=" * 80)

for rank, (_, fp, fn, tp, gt_file) in enumerate(records[:15], 1):

    image_path = find_image(gt_file.stem)

    if image_path is None:
        continue

    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    W, H = im.size

    gt = read_boxes(gt_file)
    pr = read_boxes(PR / gt_file.name, pred=True)

    # GREEN = ground truth
    for b in gt:
        x, y, w, h = b[1:5]

        x1 = int((x-w/2)*W)
        y1 = int((y-h/2)*H)
        x2 = int((x+w/2)*W)
        y2 = int((y+h/2)*H)

        draw.rectangle([x1,y1,x2,y2], outline="lime", width=4)
        draw.text((x1,y1), "GT", fill="lime")

    # RED = predictions
    for b in pr:
        x, y, w, h, conf = b[1:6]

        x1 = int((x-w/2)*W)
        y1 = int((y-h/2)*H)
        x2 = int((x+w/2)*W)
        y2 = int((y+h/2)*H)

        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        draw.text((x1,max(0,y1-18)), f"P {conf:.2f}", fill="red")

    title = f"FP={fp} FN={fn} TP={tp}"

    draw.rectangle([0,0,min(W,500),30], fill="black")
    draw.text((5,5), title, fill="white")

    out = OUT / f"{rank:02d}_{gt_file.stem}_FP{fp}_FN{fn}.png"
    im.save(out)

    print(f"{rank:02d}: {gt_file.stem} | TP={tp} FP={fp} FN={fn}")

print()
print("DONE")
print(f"Gallery: {OUT.resolve()}")
