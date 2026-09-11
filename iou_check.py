from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/runs/detect/runs/train/sonar_yolo11n_640_aug/weights/best.pt")

image = "yolo_dataset/test/images/Barge_No_1_02.png"
label = "yolo_dataset/test/labels/Barge_No_1_02.txt"

W, H = Image.open(image).size

a = open(label).read().split()
x, y, w, h = map(float, a[1:5])

gt = [
    (x - w / 2) * W,
    (y - h / 2) * H,
    (x + w / 2) * W,
    (y + h / 2) * H
]

r = model.predict(
    source=image,
    conf=0.001,
    imgsz=640,
    device=0,
    verbose=False
)[0]

best_iou = 0
best_conf = 0
best_box = None

for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):

    ix1 = max(box[0], gt[0])
    iy1 = max(box[1], gt[1])
    ix2 = min(box[2], gt[2])
    iy2 = min(box[3], gt[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    union = box_area + gt_area - inter

    iou = inter / union if union > 0 else 0

    if iou > best_iou:
        best_iou = iou
        best_conf = float(conf)
        best_box = box

print("IMAGE SIZE:", W, "x", H)
print("GROUND TRUTH:", [round(v, 1) for v in gt])
print("BEST CONFIDENCE:", round(best_conf, 4))

if best_box is not None:
    print("BEST BOX:", [round(float(v), 1) for v in best_box])
else:
    print("BEST BOX: None")

print("BEST IoU:", round(best_iou, 4))