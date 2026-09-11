from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import torchvision

# ============================================================
# SETTINGS
# ============================================================

MODEL = Path(
    "runs/detect/runs/detect/runs/tile_balanced_aug/weights/best.pt"
)

SOURCE = Path("yolo_dataset/test/images")

OUT = Path("runs/tiled_inference/balanced_aug")

TILE_H = 1728
OVERLAP = 432
STRIDE = TILE_H - OVERLAP

IMGSZ = 640
CONF = 0.05
IOU = 0.50

DEVICE = 0

OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL))


# ============================================================
# NMS
# ============================================================

def nms_boxes(boxes, scores, iou_threshold=0.5):

    if not boxes:
        return []

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    keep = torchvision.ops.nms(
        boxes_t,
        scores_t,
        iou_threshold
    )

    return keep.tolist()


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(img_path):

    img = Image.open(img_path).convert("RGB")

    W, H = img.size

    all_boxes = []
    all_scores = []

    y0 = 0

    while y0 < H:

        y1 = min(y0 + TILE_H, H)

        # Make the final tile exactly TILE_H high when possible
        if y1 == H and H >= TILE_H:
            y0_actual = H - TILE_H
        else:
            y0_actual = y0

        y1_actual = min(y0_actual + TILE_H, H)

        tile = img.crop(
            (0, y0_actual, W, y1_actual)
        )

        results = model.predict(
            source=tile,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:

            for box, score in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist()
            ):

                x1, ty1, x2, ty2 = box

                # Convert tile coordinates
                # back to original-image coordinates
                x1 = max(0, min(W, x1))
                x2 = max(0, min(W, x2))

                y1_box = max(
                    0,
                    min(H, ty1 + y0_actual)
                )

                y2_box = max(
                    0,
                    min(H, ty2 + y0_actual)
                )

                if x2 > x1 and y2_box > y1_box:

                    all_boxes.append(
                        [
                            x1,
                            y1_box,
                            x2,
                            y2_box
                        ]
                    )

                    all_scores.append(
                        float(score)
                    )

        if y1_actual >= H:
            break

        y0 += STRIDE

    # --------------------------------------------------------
    # NMS across all overlapping tiles
    # --------------------------------------------------------

    keep = nms_boxes(
        all_boxes,
        all_scores,
        iou_threshold=0.5
    )

    final_boxes = [
        all_boxes[i]
        for i in keep
    ]

    final_scores = [
        all_scores[i]
        for i in keep
    ]

    # --------------------------------------------------------
    # Draw predictions
    # --------------------------------------------------------

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    for box, score in zip(
        final_boxes,
        final_scores
    ):

        x1, y1, x2, y2 = box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=8
        )

        draw.text(
            (x1, max(0, y1 - 25)),
            f"{score:.2f}",
            fill="red"
        )

    out_path = OUT / img_path.name

    draw_img.save(out_path)

    print(
        f"{img_path.stem}: "
        f"tiles processed, "
        f"{len(final_boxes)} final detections"
    )


# ============================================================
# RUN
# ============================================================

images = sorted(
    SOURCE.glob("*.png")
)

print()
print("TILED INFERENCE")
print("Images:", len(images))
print("Tile height:", TILE_H)
print("Overlap:", OVERLAP)
print("Stride:", STRIDE)
print("Confidence:", CONF)
print()

for img_path in images:
    process_image(img_path)

print()
print("DONE")
print("Results:", OUT.resolve())