from pathlib import Path
from collections import defaultdict

GT = Path(r"yolo_dataset_groupval/test/labels")
PR = Path(r"runs/FINAL_RESULTS/onlineaug_predictions_conf001_txt/labels")

def box(x):
    x, y, w, h = x
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    )


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-9)


# Ground truth
gts = {}

for p in GT.glob("*.txt"):
    items = []

    for line in p.read_text().splitlines():
        if not line.strip():
            continue

        v = line.split()

        if len(v) >= 5:
            cls = int(v[0])
            coords = list(map(float, v[1:5]))
            items.append((cls, box(coords)))

    gts[p.stem] = items


# Predictions
preds = {}

for p in PR.glob("*.txt"):
    items = []

    for line in p.read_text().splitlines():
        if not line.strip():
            continue

        v = line.split()

        if len(v) >= 6:
            cls = int(v[0])
            coords = list(map(float, v[1:5]))
            conf = float(v[5])

            items.append((cls, box(coords), conf))

    preds[p.stem] = items


thresholds = [
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
    0.400,
    0.500,
]

print()
print("=" * 70)
print("ONLINE AUGMENTATION — THRESHOLD ANALYSIS")
print("=" * 70)
print()
print("CONF | TP | FP | FN | PRECISION | RECALL | F1")
print("-" * 70)

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    TP = 0
    FP = 0
    FN = 0

    for stem, gt in gts.items():

        pred = [
            p for p in preds.get(stem, [])
            if p[2] >= threshold
        ]

        pred.sort(key=lambda x: x[2], reverse=True)

        used_gt = set()

        for pred_cls, pred_box, conf in pred:

            best_match = -1
            best_iou = 0

            for j, (gt_cls, gt_box) in enumerate(gt):

                if j in used_gt:
                    continue

                if pred_cls != gt_cls:
                    continue

                score = iou(pred_box, gt_box)

                if score >= 0.5 and score > best_iou:
                    best_iou = score
                    best_match = j

            if best_match >= 0:
                TP += 1
                used_gt.add(best_match)
            else:
                FP += 1

        FN += len(gt) - len(used_gt)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(
        f"{threshold:0.3f} | "
        f"{TP:2d} | "
        f"{FP:3d} | "
        f"{FN:3d} | "
        f"{precision:.4f} | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print()
print("=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)
print(f"Confidence : {best_threshold:.3f}")
print(f"F1         : {best_f1:.4f}")
print("=" * 70)