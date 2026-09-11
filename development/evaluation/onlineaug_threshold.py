from pathlib import Path

GT = Path("yolo_dataset_groupval/test/labels")
PR = Path(r"runs/detect/runs/FINAL_RESULTS/onlineaug_nms03_txt/labels")
def read_boxes(path, prediction=False, conf=0.0):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue

        cls = int(float(p[0]))
        x, y, w, h = map(float, p[1:5])

        if prediction:
            if len(p) < 6:
                continue
            score = float(p[5])
            if score < conf:
                continue
        else:
            score = 1.0

        boxes.append((cls, x, y, w, h, score))

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

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)

    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0

thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

print("=" * 75)
print("ONLINE AUGMENTATION - CORRECTED THRESHOLD ANALYSIS")
print("=" * 75)
print(f"GT files: {len(list(GT.glob('*.txt')))}")
print(f"Prediction files: {len(list(PR.glob('*.txt')))}")
print()

results = []

for conf in thresholds:
    TP = FP = FN = 0

    for gt_file in GT.glob("*.txt"):
        pr_file = PR / gt_file.name

        gt = read_boxes(gt_file)
        pred = read_boxes(pr_file, prediction=True, conf=conf)

        used = set()

        pred_sorted = sorted(
            enumerate(pred),
            key=lambda z: z[1][5],
            reverse=True
        )

        for pi, pb in pred_sorted:
            matches = []

            for gi, gb in enumerate(gt):
                if gi in used:
                    continue
                if pb[0] != gb[0]:
                    continue

                score = iou(pb, gb)

                if score >= 0.5:
                    matches.append((score, gi))

            if matches:
                _, best_gi = max(matches)
                used.add(best_gi)
                TP += 1
            else:
                FP += 1

        FN += len(gt) - len(used)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    results.append((f1, conf, TP, FP, FN, precision, recall))

    print(
        f"{conf:0.3f} | "
        f"TP {TP:3d} | FP {FP:3d} | FN {FN:3d} | "
        f"P {precision:.4f} | R {recall:.4f} | F1 {f1:.4f}"
    )

best = max(results)

print()
print("=" * 75)
print("BEST F1 THRESHOLD")
print("=" * 75)
print(f"Confidence : {best[1]:.3f}")
print(f"TP         : {best[2]}")
print(f"FP         : {best[3]}")
print(f"FN         : {best[4]}")
print(f"Precision  : {best[5]:.6f}")
print(f"Recall     : {best[6]:.6f}")
print(f"F1         : {best[0]:.6f}")
print("=" * 75)
