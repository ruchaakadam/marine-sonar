from io import BytesIO
from pathlib import Path

import json
import os
from fastapi import File, Form, UploadFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image,ImageDraw
from ultralytics import YOLO

from datetime import datetime

from fastapi.responses import StreamingResponse

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)

app = FastAPI(
    title="Marine Sonar API",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://marine-sonar.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Existing model is kept untouched as the fallback crab-pot detector.
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "best.pt"

# New SIH sonar detector.
DRISHTI_MODEL_PATH = BASE_DIR / "models" / "drishti.pt"

CRABPOT_MODEL_PATH = BASE_DIR / "models" / "crabpot_trained.pt"
ROCK_MODEL_PATH = BASE_DIR / "models" / "rock_trained.pt"

ACCURACY = 0.20
NMS_IOU = 0.40

print("=" * 70)
print("LOADING MARINE SONAR MODELS")
print("=" * 70)
print(f"DRISHTI model : {DRISHTI_MODEL_PATH}")
print(f"Fallback model: {FALLBACK_MODEL_PATH}")
print(f"Accuracy     : {ACCURACY}")
print(f"NMS IoU       : {NMS_IOU}")

if not DRISHTI_MODEL_PATH.exists():
    raise FileNotFoundError(f"DRISHTI model not found: {DRISHTI_MODEL_PATH}")

# Primary model
drishti_model = YOLO(str(DRISHTI_MODEL_PATH))

# Specialized models can be disabled on low-memory deployments.
# They remain enabled by default for local SIH testing.
ENABLE_SPECIALIZED_MODELS = os.getenv("ENABLE_SPECIALIZED_MODELS", "false").lower() == "true"

crabpot_model = None
rock_model = None

if ENABLE_SPECIALIZED_MODELS:
    if CRABPOT_MODEL_PATH.exists():
        crabpot_model = YOLO(str(CRABPOT_MODEL_PATH))
        print(f"Crab-pot classes: {crabpot_model.names}")
    else:
        print(f"Crab-pot model not found: {CRABPOT_MODEL_PATH}")

    if ROCK_MODEL_PATH.exists():
        rock_model = YOLO(str(ROCK_MODEL_PATH))
        print(f"Rock classes: {rock_model.names}")
    else:
        print(f"Rock model not found: {ROCK_MODEL_PATH}")

print(f"DRISHTI classes: {drishti_model.names}")
print(f"Specialized models enabled: {ENABLE_SPECIALIZED_MODELS}")
print("PRIMARY MODEL LOADED SUCCESSFULLY")

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


@app.get("/")
def root():
    return {
        "service": "marine-sonar-api",
        "status": "running",
        "models_loaded": True,
        "primary_model": "drishti",
        "specialized_models": ["crabpot_trained", "rock_trained"],
        "specialized_models_enabled": ENABLE_SPECIALIZED_MODELS,
        "fallback_model": "best",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "marine-sonar-api",
        "models_loaded": True,
        "primary_model": str(DRISHTI_MODEL_PATH),
        "crabpot_model": str(CRABPOT_MODEL_PATH),
        "rock_model": str(ROCK_MODEL_PATH),
        "specialized_models_enabled": ENABLE_SPECIALIZED_MODELS,
        "fallback_model": str(FALLBACK_MODEL_PATH),
        "accuracy": ACCURACY,
        "nms_iou": NMS_IOU,
    }


def normalize_detection(model, box):
    xyxy = box.xyxy[0].tolist()
    accuracy = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = str(model.names[class_id]).lower().replace("-", "_").replace(" ", "_")

    # The old single-class model calls its class "target".
    # In this project that existing class is treated as crab_pot.
    if class_name == "target":
        class_name = "crab_pot"

    return {
        "class_id": class_id,
        "class_name": class_name,
        "accuracy": round(accuracy, 4),
        "bbox": {
            "x1": round(xyxy[0], 2),
            "y1": round(xyxy[1], 2),
            "x2": round(xyxy[2], 2),
            "y2": round(xyxy[3], 2),
        },
    }


def run_model(model, image):
    results = model.predict(
        source=image,
        conf=ACCURACY,
        iou=NMS_IOU,
        imgsz=720,
        verbose=False,
    )

    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            detections.append(normalize_detection(model, box))

    return detections


def analyze_target(detections):
    if not detections:
        return {
            "status": "NO TARGET",
            "target": None,
            "possible_impact_object": None,
            "accuracy": 0,
            "impact_assessment": "NO CONFIDENT TARGET IDENTIFIED",
            "assessment": (
                "No confident target was identified in the sonar image. "
                "This does not prove the area is clear."
            ),
            "evidence": [],
            "recommended_action": (
                "Maintain monitoring and obtain another sonar scan if the "
                "area is operationally important."
            ),
        }

    best_detection = max(detections, key=lambda d: d["accuracy"])

    accuracy = best_detection["accuracy"]
    class_name = best_detection["class_name"]
    bbox = best_detection["bbox"]

    target_profiles = {
        "crab_pot": {
            "label": "Crab Pot",
            "risk": "MEDIUM",
            "action": (
                "Maintain safe clearance, reduce speed if necessary, and "
                "perform a secondary sonar scan before confirming the target."
            ),
            "assessment": (
                "The sonar evidence is consistent with a crab-pot target. "
                "The image alone does not prove vessel impact."
            ),
        },
        "submarine_pipeline": {
            "label": "Submarine Pipeline",
            "risk": "CRITICAL",
            "action": (
                "Do not cross the detected feature. Maintain safe clearance, "
                "verify its position with a secondary sonar scan, and "
                "correlate the location with navigation data."
            ),
            "assessment": (
                "The sonar model identifies a feature consistent with a "
                "submarine pipeline. Treat the target as a navigation hazard "
                "until independently verified."
            ),
        },
        "shipwreck": {
            "label": "Shipwreck",
            "risk": "HIGH",
            "action": (
                "Reduce speed, avoid the detected area, and perform a "
                "secondary sonar scan while recording the target position."
            ),
            "assessment": (
                "The sonar model identifies a structured feature consistent "
                "with a shipwreck. The image alone does not prove an impact."
            ),
        },
        "ghost_net": {
            "label": "Ghost Net",
            "risk": "HIGH",
            "action": (
                "Reduce speed, maintain clearance, perform a secondary scan, "
                "and flag the location for marine-operations review."
            ),
            "assessment": (
                "The sonar model identifies a feature consistent with an "
                "entangled or ghost-net target. Independent verification is "
                "recommended before treating it as confirmed."
            ),
        },
        "mine_cylinder": {
            "label": "Mine Cylinder",
            "risk": "CRITICAL",
            "action": (
                "Do not approach the target. Maintain maximum practical "
                "clearance, avoid crossing the area, and escalate for "
                "specialist verification."
            ),
            "assessment": (
                "The sonar model identifies a feature consistent with a "
                "mine-cylinder target. Treat it as a high-risk hazard until "
                "independently verified."
            ),
        },
        "rock": {
            "label": "Rock / Boulder",
            "risk": "MEDIUM",
            "action": (
                "Maintain safe clearance, reduce speed if necessary, and "
                "perform a secondary sonar scan to verify the feature and "
                "its position."
            ),
            "assessment": (
                "The sonar model identifies a feature consistent with a "
                "rock or boulder. The image alone does not establish its "
                "exact size, depth, or navigational clearance."
            ),
        },
    }

    profile = target_profiles.get(
        class_name,
        {
            "label": "Unknown Target",
            "risk": "MEDIUM",
            "action": (
                "Perform a secondary sonar scan and manually verify the "
                "target before taking impact-related action."
            ),
            "assessment": (
                "A sonar anomaly was detected, but the available model does "
                "not provide a supported target type."
            ),
        },
    )

    if accuracy >= 0.75:
        status = "HIGH ACCURACY"
    elif accuracy >= 0.50:
        status = "MEDIUM ACCURACY"
    else:
        status = "LOW ACCURACY"

    evidence = [
        f"Detected target type: {profile['label']}.",
        f"Sonar analysis accuracy: {round(accuracy * 100)}%.",
        f"Risk classification: {profile['risk']}.",
        (
            f"Detected image region: x1={bbox['x1']}, y1={bbox['y1']}, "
            f"x2={bbox['x2']}, y2={bbox['y2']}."
        ),
        "Sonar imagery alone does not prove that the vessel collided with the target.",
    ]

    return {
        "status": status,
        "target": profile["label"],
        "target_class": class_name,
        "risk_level": profile["risk"],
        "possible_impact_object": profile["label"],
        "accuracy": accuracy,
        "impact_assessment": (
            f"POSSIBLE {profile['label'].upper()} — "
            f"{profile['risk']} RISK"
        ),
        "assessment": profile["assessment"],
        "evidence": evidence,
        "recommended_action": profile["action"],
        "bounding_box": bbox,
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPEG, or WebP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Primary model: DRISHTI
    try:
        detections = run_model(drishti_model, image)
        model_used = "drishti"
    except Exception as e:
        detections = []
        model_used = "drishti_failed"
        print(f"DRISHTI inference failed: {e}")

    # Specialized crab-pot model. Avoid duplicates if DRISHTI already
    # identified a crab pot.
    if crabpot_model is not None:
        has_crab_pot = any(
            d.get("class_name") == "crab_pot"
            for d in detections
        )

        if not has_crab_pot:
            try:
                crabpot_detections = run_model(crabpot_model, image)
                detections.extend(crabpot_detections)
                if crabpot_detections:
                    model_used = f"{model_used}+crabpot"
            except Exception as e:
                print(f"Crab-pot inference failed: {e}")

    # Specialized rock model. DRISHTI does not contain a rock class,
    # so run the rock detector independently when available.
    if rock_model is not None:
        try:
            rock_detections = run_model(rock_model, image)
            detections.extend(rock_detections)
            if rock_detections:
                model_used = f"{model_used}+rock"
        except Exception as e:
            print(f"Rock inference failed: {e}")

    target_analysis = analyze_target(detections)

    return {
        "success": True,
        "filename": file.filename,
        "model": model_used,
        "accuracy_threshold": ACCURACY,
        "nms_iou_threshold": NMS_IOU,
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "detection_count": len(detections),
        "detections": detections,
        "target_analysis": target_analysis,
    }


def calculate_iou(box_a, box_b):
    ax1, ay1 = box_a["x1"], box_a["y1"]
    ax2, ay2 = box_a["x2"], box_a["y2"]

    bx1, by1 = box_b["x1"], box_b["y1"]
    bx2, by2 = box_b["x2"], box_b["y2"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def track_targets(previous_detections, current_detections):
    IOU_THRESHOLD = 0.30

    previous_detections = previous_detections or []
    current_detections = current_detections or []

    matches = []
    matched_previous = set()
    matched_current = set()

    for current_index, current in enumerate(current_detections):
        best_iou = 0.0
        best_previous_index = None

        for previous_index, previous in enumerate(previous_detections):
            if previous_index in matched_previous:
                continue

            iou = calculate_iou(previous["bbox"], current["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_previous_index = previous_index

        if (
            best_previous_index is not None
            and best_iou >= IOU_THRESHOLD
        ):
            matches.append({
                "status": "PERSISTENT TARGET",
                "previous_target": best_previous_index + 1,
                "current_target": current_index + 1,
                "accuracy": current["accuracy"],
                "iou": round(best_iou, 4),
                "possible_object": current.get(
                    "class_name", "unknown"
                ),
            })

            matched_previous.add(best_previous_index)
            matched_current.add(current_index)

    for current_index, current in enumerate(current_detections):
        if current_index not in matched_current:
            matches.append({
                "status": "NEW TARGET",
                "previous_target": None,
                "current_target": current_index + 1,
                "accuracy": current["accuracy"],
                "iou": 0,
                "possible_object": current.get(
                    "class_name", "unknown"
                ),
            })

    for previous_index, previous in enumerate(previous_detections):
        if previous_index not in matched_previous:
            matches.append({
                "status": "NO LONGER DETECTED",
                "previous_target": previous_index + 1,
                "current_target": None,
                "accuracy": previous["accuracy"],
                "iou": 0,
                "possible_object": previous.get(
                    "class_name", "unknown"
                ),
            })

    persistent_count = sum(
        1 for match in matches
        if match["status"] == "PERSISTENT TARGET"
    )

    new_count = sum(
        1 for match in matches
        if match["status"] == "NEW TARGET"
    )

    lost_count = sum(
        1 for match in matches
        if match["status"] == "NO LONGER DETECTED"
    )

    return {
        "success": True,
        "iou_threshold": IOU_THRESHOLD,
        "previous_scan_targets": len(previous_detections),
        "current_scan_targets": len(current_detections),
        "persistent_targets": persistent_count,
        "new_targets": new_count,
        "no_longer_detected": lost_count,
        "matches": matches,
    }


@app.post("/api/track")
async def track_scans(data: dict):
    previous_detections = data.get("previous_detections", [])
    current_detections = data.get("current_detections", [])

    if not isinstance(previous_detections, list):
        raise HTTPException(
            status_code=400,
            detail="previous_detections must be a list.",
        )

    if not isinstance(current_detections, list):
        raise HTTPException(
            status_code=400,
            detail="current_detections must be a list.",
        )

    try:
        return track_targets(
            previous_detections,
            current_detections,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Target tracking failed: {str(e)}",
        )
# ============================================================
# PDF INCIDENT REPORT
# ============================================================
@app.post("/api/report")
async def generate_report(
    data: str = Form(...),
    image: UploadFile = File(None),
):
    try:
        if isinstance(data, str):
            data = data.lstrip("\ufeff").strip()
            data = json.loads(data)

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=18,
        )

        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontSize=10,
            leading=15,
            spaceAfter=6,
        )

        story = []

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "MARINE SONAR<br/>INCIDENT & TARGET ANALYSIS REPORT",
                title_style,
            )
        )

        story.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
                body_style,
            )
        )

        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # SCAN INFORMATION
        # ----------------------------------------------------

        story.append(Paragraph("1. SCAN INFORMATION", heading_style))

        image_info = data.get("image", {})
        filename = data.get("filename", "Unknown")

        scan_data = [
            ["Image File", str(filename)],
            [
                "Image Size",
                f"{image_info.get('width', '—')} × "
                f"{image_info.get('height', '—')} px",
            ],
            ["Model Used", str(data.get("model", "Unknown"))],
            [
                "Accuracy Threshold",
                str(data.get("accuracy_threshold", "—")),
            ],
            [
                "NMS IoU Threshold",
                str(data.get("nms_iou_threshold", "—")),
            ],
        ]

        scan_table = Table(
            scan_data,
            colWidths=[170, 330],
        )

        scan_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0b2433")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )

        story.append(scan_table)
        # ----------------------------------------------------
        # SONAR IMAGE WITH DETECTION ANNOTATIONS
        # ----------------------------------------------------

        if image:
            image_bytes = await image.read()

            image_stream = BytesIO(image_bytes)

            pil_image = Image.open(image_stream).convert("RGB")

            draw = ImageDraw.Draw(pil_image)

            detections = data.get("detections", [])

            for detection in detections:
                box = detection.get("bbox", {})

                x1 = int(float(box.get("x1", 0)))
                y1 = int(float(box.get("y1", 0)))
                x2 = int(float(box.get("x2", 0)))
                y2 = int(float(box.get("y2", 0)))

                class_name = str(
                    detection.get(
                        "class_name",
                        "target"
                    )
                )

                accuracy = float(
                    detection.get(
                        "accuracy",
                        0
                    )
                )

                label = (
                    f"{class_name} "
                    f"{round(accuracy * 100)}%"
                )

                # Mark detected target coordinates
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=5,
                )

                # Label background
                text_box = draw.textbbox(
                    (x1, y1),
                    label
                )

                draw.rectangle(
                    [
                        text_box[0],
                        text_box[1],
                        text_box[2] + 8,
                        text_box[3] + 6,
                    ],
                    fill="red",
                )

                # Label text
                draw.text(
                    (x1 + 4, y1 + 2),
                    label,
                    fill="white",
                )

            annotated_stream = BytesIO()

            pil_image.save(
                annotated_stream,
                format="JPEG",
                quality=95,
            )

            annotated_stream.seek(0)

            sonar_image = ReportLabImage(
                annotated_stream,
                width=500,
                height=500,
                hAlign="CENTER",
            )

            story.append(
                Paragraph(
                    "SONAR IMAGE EVIDENCE",
                    heading_style,
                )
            )

            story.append(sonar_image)

            story.append(Spacer(1, 12))

        # ----------------------------------------------------
        # TARGET ANALYSIS
        # ----------------------------------------------------

        analysis = data.get("target_analysis", {})

        story.append(
            Paragraph("2. TARGET ANALYSIS", heading_style)
        )

        target_data = [
            ["Status", str(analysis.get("status", "—"))],
            ["Target", str(analysis.get("target", "No target"))],
            [
                "Possible Impact Object",
                str(
                    analysis.get(
                        "possible_impact_object",
                        "—"
                    )
                ),
            ],
            [
                "Accuracy",
                f"{round(float(analysis.get('accuracy', 0)) * 100)}%",
            ],
            [
                "Risk Level",
                str(analysis.get("risk_level", "—")),
            ],
            [
                "Impact Assessment",
                str(
                    analysis.get(
                        "impact_assessment",
                        "—"
                    )
                ),
            ],
        ]

        target_table = Table(
            target_data,
            colWidths=[170, 330],
        )

        target_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0b2433")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )

        story.append(target_table)

        # ----------------------------------------------------
        # SONAR ASSESSMENT
        # ----------------------------------------------------

        story.append(
            Paragraph("3. SONAR ASSESSMENT", heading_style)
        )

        assessment = analysis.get(
            "assessment",
            "No assessment available."
        )

        story.append(
            Paragraph(str(assessment), body_style)
        )

        # ----------------------------------------------------
        # RECOMMENDED ACTION
        # ----------------------------------------------------

        story.append(
            Paragraph("4. RECOMMENDED NAVIGATION ACTION", heading_style)
        )

        recommended_action = analysis.get(
            "recommended_action",
            "Maintain monitoring and verify the target."
        )

        story.append(
            Paragraph(
                f"<b>{recommended_action}</b>",
                body_style,
            )
        )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        story.append(
            Paragraph("5. DETECTION EVIDENCE", heading_style)
        )

        evidence = analysis.get("evidence", [])

        if evidence:
            for item in evidence:
                story.append(
                    Paragraph(
                        f"• {str(item)}",
                        body_style,
                    )
                )
        else:
            story.append(
                Paragraph(
                    "No additional evidence available.",
                    body_style,
                )
            )

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        story.append(
            Paragraph("6. DETECTED OBJECTS", heading_style)
        )

        detections = data.get("detections", [])

        if detections:
            detection_data = [
                ["#", "Object", "Accuracy", "Coordinates"]
            ]

            for index, detection in enumerate(
                detections,
                start=1
            ):
                bbox = detection.get("bbox", {})

                bbox_text = (
                    f"({bbox.get('x1', '—')}, "
                    f"{bbox.get('y1', '—')}) → "
                    f"({bbox.get('x2', '—')}, "
                    f"{bbox.get('y2', '—')})"
                    
                )

                detection_data.append(
                    [
                        str(index),
                        str(
                            detection.get(
                                "class_name",
                                "unknown"
                            )
                        ),
                        f"{round(float(detection.get('accuracy', 0)) * 100)}%",
                        bbox_text,
                    ]
                )

            detection_table = Table(
                detection_data,
                colWidths=[30, 120, 80, 270],
            )

            detection_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#0b2433"),
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white,
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey,
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold",
                        ),
                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            8,
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP",
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            6,
                        ),
                    ]
                )
            )

            story.append(detection_table)

        else:
            story.append(
                Paragraph(
                    "No objects were detected.",
                    body_style,
                )
            )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        story.append(Spacer(1, 18))

        story.append(
            Paragraph(
                "<b>IMPORTANT:</b> Sonar imagery and sonar analysis "
                "provide decision-support information. A detected "
                "target does not independently prove physical vessel "
                "impact. Secondary verification is recommended.",
                body_style,
            )
        )

        # Build PDF
        doc.build(story)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    'attachment; filename="marine_sonar_incident_report.pdf"'
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF report generation failed: {str(e)}",
        )