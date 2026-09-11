from ultralytics import YOLO

MODELS = {
    "groupval_onlineaug_1024": r"runs\detect\runs\detect\groupval_onlineaug_1024\weights\best.pt",
    "tile_balanced_aug": r"runs\detect\runs\detect\runs\tile_balanced_aug\weights\best.pt",
    "tile_balanced_aug_1024": r"runs\detect\runs\detect\runs\tile_balanced_aug_1024\weights\best.pt",
    "tile_balanced_train": r"runs\detect\runs\detect\runs\tile_balanced_train\weights\best.pt",
    "tile_positive_train": r"runs\detect\runs\detect\runs\tile_positive_train\weights\best.pt"
}

DATA = r"yolo_dataset_groupval\data.yaml"

def main():
    print("=" * 80)
    print("MARINE SONAR - SAME TEST SET MODEL COMPARISON")
    print("Test images : 120")
    print("Confidence  : 0.05")
    print("NMS IoU     : 0.45")
    print("Image size  : 640")
    print("Workers     : 0")
    print("=" * 80)

    results = []

    for name, path in MODELS.items():
        print(f"\n>>> TESTING: {name}")
        print(f"Model: {path}")

        try:
            model = YOLO(path)

            metrics = model.val(
                data=DATA,
                split="test",
                imgsz=640,
                conf=0.05,
                iou=0.45,
                device=0,
                workers=0,
                plots=False,
                verbose=False
            )

            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0
            )

            results.append(
                (name, precision, recall, f1, map50, map5095)
            )

            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1        : {f1:.4f}")
            print(f"mAP50     : {map50:.4f}")
            print(f"mAP50-95  : {map5095:.4f}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    results.sort(key=lambda x: x[5], reverse=True)

    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r[0]:<28} "
            f"P={r[1]:.4f} "
            f"R={r[2]:.4f} "
            f"F1={r[3]:.4f} "
            f"mAP50={r[4]:.4f} "
            f"mAP50-95={r[5]:.4f}"
        )

if __name__ == "__main__":
    main()
