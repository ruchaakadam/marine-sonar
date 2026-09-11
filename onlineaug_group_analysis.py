from pathlib import Path
from collections import defaultdict
import re

GT = Path("yolo_dataset_groupval/test/labels")
PR = Path(r"runs/detect/runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels")

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
            score = float(p[5])
            if score < CONF:
                continue
        else:
            score = 1.0

        boxes.append((cls, x, y, w, h, score))

    return boxes

def iou(a, b):
    ax1, ay1 = a[1]-a[3]/2, a[2]-a[4]/2
    ax2, ay2 = a[1]+a[3]/2, a[2]+a[4]/2

    bx1, by1 = b[1]-b[3]/2, b[2]-b[4]/2
    bx2, by2 = b[1]+b[3]/2, b[2]+b[4]/2

    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)

    inter = max(0,ix2-ix1) * max(0,iy2-iy1)

    aa = max(0,ax2-ax1) * max(0,ay2-ay1)
    ab = max(0,bx2-bx1) * max(0,by2-by1)

    union = aa + ab - inter

    return inter / union if union > 0 else 0

def group_name(filename):
    # Remove the final _01, _02, etc.
    return re.sub(r'_\d+$', '', Path(filename).stem)

stats = defaultdict(lambda: {
    "images": 0,
    "gt": 0,
    "tp": 0,
    "fp": 0,
    "fn": 0
})

for gt_file in GT.glob("*.txt"):

    group = group_name(gt_file.name)
    stats[group]["images"] += 1

    gt = read_boxes(gt_file)
    pr = read_boxes(PR / gt_file.name, pred=True)

    stats[group]["gt"] += len(gt)

    used = set()

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

            score = iou(pb, gb)

            if score >= IOU_T:
                matches.append((score, gi))

        if matches:
            _, best_gi = max(matches)
            used.add(best_gi)
            stats[group]["tp"] += 1
        else:
            stats[group]["fp"] += 1

    stats[group]["fn"] += len(gt) - len(used)

print()
print("=" * 100)
print("ONLINE AUGMENTATION - GROUP ERROR ANALYSIS")
print(f"Confidence = {CONF}")
print(f"IoU = {IOU_T}")
print("=" * 100)

print(
    f"{'GROUP':28} {'IMG':>4} {'GT':>4} "
    f"{'TP':>4} {'FP':>5} {'FN':>4} "
    f"{'PREC':>8} {'RECALL':>8} {'F1':>8}"
)

print("-" * 100)

total = defaultdict(int)

for group, s in sorted(stats.items()):

    tp, fp, fn = s["tp"], s["fp"], s["fn"]

    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = (
        2*precision*recall/(precision+recall)
        if precision+recall else 0
    )

    print(
        f"{group:28} "
        f"{s['images']:4d} "
        f"{s['gt']:4d} "
        f"{tp:4d} "
        f"{fp:5d} "
        f"{fn:4d} "
        f"{precision:8.3f} "
        f"{recall:8.3f} "
        f"{f1:8.3f}"
    )

    for k in ["images","gt","tp","fp","fn"]:
        total[k] += s[k]

print("-" * 100)

tp, fp, fn = total["tp"], total["fp"], total["fn"]

precision = tp/(tp+fp) if tp+fp else 0
recall = tp/(tp+fn) if tp+fn else 0
f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

print(
    f"{'TOTAL':28} "
    f"{total['images']:4d} "
    f"{total['gt']:4d} "
    f"{tp:4d} "
    f"{fp:5d} "
    f"{fn:4d} "
    f"{precision:8.3f} "
    f"{recall:8.3f} "
    f"{f1:8.3f}"
)

print("=" * 100)
