from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = r"D:\sih\marine-sonar\runs\detect\runs\detect\runs\groupval_1024\weights\best.pt"
IMG_DIR = Path(r"D:\sih\marine-sonar\yolo_dataset_groupval\test\images")
LBL_DIR = Path(r"D:\sih\marine-sonar\yolo_dataset_groupval\test\labels")

CONF = 0.10
IOU_THR = 0.50

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(area_a+area_b-inter+1e-9)

model = YOLO(MODEL)

TP=FP=FN=0
background_fp=0
per_image=[]

results = model.predict(
    source=str(IMG_DIR),
    imgsz=1024,
    device=0,
    conf=CONF,
    iou=IOU_THR,
    verbose=False
)

for r in results:
    name = Path(r.path).name
    label_file = LBL_DIR / (Path(r.path).stem + ".txt")

    gt=[]
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p=line.split()
            if len(p) != 5:
                continue
            cls,xc,yc,w,h=map(float,p)
            H,W=r.orig_shape
            gt.append([
                (xc-w/2)*W,
                (yc-h/2)*H,
                (xc+w/2)*W,
                (yc+h/2)*H,
                int(cls)
            ])

    preds=[]
    if r.boxes is not None:
        for box,cls,conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):
            preds.append([*box,float(cls),float(conf)])

    preds.sort(key=lambda x:x[5], reverse=True)
    matched=set()
    itp=ifp=ifn=0

    for p in preds:
        best_iou=0
        best_j=-1
        for j,g in enumerate(gt):
            if j in matched or int(p[4]) != int(g[4]):
                continue
            v=iou(p[:4],g[:4])
            if v>best_iou:
                best_iou=v
                best_j=j

        if best_iou >= IOU_THR:
            matched.add(best_j)
            itp += 1
        else:
            ifp += 1

    ifn=len(gt)-len(matched)

    TP += itp
    FP += ifp
    FN += ifn

    if not gt and ifp:
        background_fp += ifp

    per_image.append((name,itp,ifp,ifn))

precision=TP/(TP+FP) if TP+FP else 0
recall=TP/(TP+FN) if TP+FN else 0
f1=2*precision*recall/(precision+recall) if precision+recall else 0

print()
print("="*65)
print("FINAL ERROR ANALYSIS")
print("="*65)
print(f"Confidence : {CONF}")
print(f"IoU        : {IOU_THR}")
print()
print(f"TP         : {TP}")
print(f"FP         : {FP}")
print(f"FN         : {FN}")
print(f"Background FP : {background_fp}")
print()
print(f"Precision  : {precision:.6f}")
print(f"Recall     : {recall:.6f}")
print(f"F1         : {f1:.6f}")
print()
print("WORST FALSE-POSITIVE IMAGES")
for name,tp,fp,fn in sorted(per_image,key=lambda x:x[2],reverse=True)[:15]:
    if fp:
        print(f"{name:35} TP={tp:2} FP={fp:2} FN={fn:2}")

print("="*65)
