"""
AI Based Structural Crack and Defect Detection
Team: Safna Firoz, Jisa Asha Joseph, Ron Geo Roy, Vaishnav Shibu
Supervisor: Afia S. Hameed, Dept. of CE, Saintgits College of Engineering

Flask backend. Handles image upload, runs YOLOv8 inference when a trained
model (crack_model.pt) is available, otherwise falls back to an OpenCV
heuristic "demo mode" so the app is fully functional before training
finishes. Returns bounding boxes, severity score, an annotated image, and a
causes / prevention / repair / consequences report pulled from the
CRACK_ANALYSIS knowledge base.
"""

import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crack_model.pt")

# ---------------------------------------------------------------------------
# Class definitions (aligned with RDD2022 damage classes)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Longitudinal Crack",   # D00
    1: "Transverse Crack",     # D10
    2: "Alligator Crack",      # D20
    3: "Pothole",              # D40
}

# BGR colours for cv2.rectangle / cv2.putText, one per class
CLASS_COLORS = {
    "Longitudinal Crack": (0, 255, 170),   # neon green
    "Transverse Crack":   (255, 200, 0),   # amber
    "Alligator Crack":    (0, 140, 255),   # orange
    "Pothole":            (0, 0, 255),     # red
}

CRACK_ANALYSIS = {
    "Longitudinal Crack": {
        "causes": [
            "Fatigue of the road surface from repeated traffic loading along the direction of travel.",
            "Poor construction joints between paving lanes.",
            "Base or sub-base settlement running parallel to the pavement edge.",
            "Shrinkage of the asphalt binder over time.",
        ],
        "prevention": [
            "Ensure proper compaction of base and sub-base layers during construction.",
            "Use quality control checks on paving lane joints.",
            "Apply routine crack sealing before cracks widen.",
            "Monitor traffic loads and enforce axle-load limits.",
        ],
        "repair": [
            "Clean the crack of debris and vegetation.",
            "Apply hot-poured rubberised crack sealant for cracks under 10 mm.",
            "For wider cracks, mill and overlay the affected strip.",
            "Re-compact the base layer if settlement is confirmed.",
        ],
        "consequences": [
            "Water infiltration weakens the base layer, accelerating pavement failure.",
            "Crack widens under continued traffic, eventually forming a pothole.",
            "Reduced ride quality and increased vehicle wear, especially for two-wheelers.",
            "Full-depth reconstruction becomes necessary if untreated for multiple seasons.",
        ],
    },
    "Transverse Crack": {
        "causes": [
            "Thermal contraction of the asphalt in fluctuating temperatures.",
            "Reflective cracking from an underlying concrete slab joint.",
            "Aging and embrittlement of the asphalt binder.",
            "Poor mix design with insufficient binder content.",
        ],
        "prevention": [
            "Use polymer-modified binders that resist thermal cracking.",
            "Apply a stress-absorbing membrane interlayer over slab joints.",
            "Seal cracks early during routine maintenance cycles.",
            "Specify mix designs suited to local climate extremes.",
        ],
        "repair": [
            "Rout and seal the crack with a flexible sealant.",
            "Apply a fibre-reinforced overlay for widespread transverse cracking.",
            "Address underlying joint issues if reflective cracking is confirmed.",
        ],
        "consequences": [
            "Moisture ingress causes stripping of the asphalt binder from aggregate.",
            "Cracks propagate and interconnect, forming block cracking patterns.",
            "Ride comfort and vehicle handling deteriorate, raising accident risk.",
        ],
    },
    "Alligator Crack": {
        "causes": [
            "Structural fatigue from repeated heavy axle loads exceeding pavement design capacity.",
            "Inadequate pavement thickness for the traffic it carries.",
            "Poor drainage leading to sustained sub-grade weakening.",
            "Loss of base support from long-term water infiltration.",
        ],
        "prevention": [
            "Design pavement thickness according to projected traffic loads.",
            "Maintain effective surface and sub-surface drainage.",
            "Carry out periodic structural (deflection) surveys on high-traffic roads.",
            "Restrict overloaded vehicles on vulnerable stretches.",
        ],
        "repair": [
            "Full-depth patching of the affected area is required; sealing alone is insufficient.",
            "Excavate and reconstruct the base if sub-grade failure is present.",
            "Improve drainage around the repaired section before resurfacing.",
        ],
        "consequences": [
            "Rapid progression to potholes once fatigue cracking is present.",
            "Loss of structural integrity across the whole pavement section.",
            "Significantly higher repair cost if reconstruction is delayed.",
            "Elevated risk of vehicle damage and accidents, especially for motorcycles.",
        ],
    },
    "Pothole": {
        "causes": [
            "Water infiltration through existing cracks that weakens the base under traffic loading.",
            "Freeze-thaw and monsoon-driven wet-dry cycles accelerating material loss.",
            "Poor original construction quality or compaction.",
            "Delayed maintenance of earlier-stage cracking.",
        ],
        "prevention": [
            "Seal cracks promptly before water can penetrate the base.",
            "Ensure adequate camber and drainage so water does not pool on the surface.",
            "Carry out proactive resurfacing on roads showing early fatigue cracking.",
        ],
        "repair": [
            "Clean loose material from the pothole and square off the edges.",
            "Apply tack coat and compact hot-mix or cold-mix asphalt patch in layers.",
            "For recurring potholes, investigate and correct the underlying drainage issue.",
        ],
        "consequences": [
            "Immediate risk of vehicle damage (tyres, suspension, alignment) and motorcycle accidents.",
            "Rapid enlargement during monsoon season as edges erode further.",
            "High-cost emergency repairs versus low-cost preventive sealing.",
            "Liability and safety risk for municipal authorities if left unaddressed.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Optional YOLOv8 model (production path)
# ---------------------------------------------------------------------------
_model = None
_model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        from ultralytics import YOLO

        _model = YOLO(MODEL_PATH)
    except Exception as exc:  # ultralytics missing, or model file invalid
        _model_load_error = str(exc)
        _model = None


def run_model_analysis(img_bytes):
    """
    INPUT: raw image bytes from the HTTP upload.
    Decodes the image, runs detection (trained YOLOv8 model if available,
    otherwise an OpenCV heuristic demo pipeline), annotates the image, and
    returns a JSON-serialisable dict.
    """
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a JPG, PNG, or WEBP file.")

    if _model is not None:
        detections = _run_yolo_inference(img)
        mode = "model"
    else:
        detections = _run_demo_heuristic(img)
        mode = "demo"

    annotated = _annotate_image(img.copy(), detections)
    _, buf = cv2.imencode(".jpg", annotated)
    b64_image = base64.b64encode(buf).decode("utf-8")

    severity = _compute_severity(detections, img.shape)
    report = _build_report(detections)

    return {
        "mode": mode,
        "detections": detections,
        "num_detections": len(detections),
        "severity": severity,
        "annotated_image": b64_image,
        "report": report,
    }


def _run_yolo_inference(img):
    """Real inference path using a trained crack_model.pt (YOLOv8)."""
    results = _model.predict(img, conf=0.25, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = CLASS_NAMES.get(cls, "Unknown Defect")
        detections.append(
            {
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            }
        )
    return detections


def _run_demo_heuristic(img):
    """
    Demo-mode fallback using classical OpenCV image processing so the web
    app is fully usable before the YOLOv8 model has finished training.

    Pipeline: greyscale -> blur -> Canny edges -> contour extraction ->
    geometric heuristics (aspect ratio, area, elongation) to approximate
    a defect class for each candidate region.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.0015
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        candidates.append((area, x, y, bw, bh, c))

    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:6]  # keep the most prominent regions

    detections = []
    for area, x, y, bw, bh, c in candidates:
        aspect = bw / float(bh) if bh else 1.0
        fill_ratio = area / float(bw * bh) if bw * bh else 0
        edge_density = area / float(h * w)

        if fill_ratio > 0.55 and 0.6 < aspect < 1.7:
            class_name = "Pothole"
        elif fill_ratio > 0.35 and area > min_area * 4:
            class_name = "Alligator Crack"
        elif aspect >= 1.8:
            class_name = "Transverse Crack"
        else:
            class_name = "Longitudinal Crack"

        # Pseudo-confidence from edge density and fill ratio; demo mode only.
        confidence = float(np.clip(0.4 + fill_ratio * 0.3 + edge_density * 8, 0.35, 0.93))

        detections.append(
            {
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [x, y, x + bw, y + bh],
            }
        )

    return detections


def _annotate_image(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class"], (0, 255, 170))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
        label = f'{det["class"]} {det["confidence"] * 100:.1f}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 3, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA,
        )
    return img


def _compute_severity(detections, img_shape):
    if not detections:
        return {"score": 0, "category": "None", "urgency": "No defects detected."}

    h, w = img_shape[:2]
    frame_area = h * w
    weights = {"Pothole": 1.0, "Alligator Crack": 0.85, "Transverse Crack": 0.55, "Longitudinal Crack": 0.45}

    worst = 0.0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(frame_area)
        w_class = weights.get(det["class"], 0.5)
        score_contrib = (0.6 * det["confidence"] + 0.4 * min(box_area_ratio * 20, 1.0)) * w_class
        worst = max(worst, score_contrib)

    score = round(worst * 10, 1)
    if score >= 7.5:
        category, urgency = "Critical", "Immediate repair recommended — safety risk to vehicles."
    elif score >= 5:
        category, urgency = "High", "Schedule repair within 1-2 weeks."
    elif score >= 2.5:
        category, urgency = "Moderate", "Include in the next routine maintenance cycle."
    else:
        category, urgency = "Low", "Monitor; no immediate action required."

    return {"score": score, "category": category, "urgency": urgency}


def _build_report(detections):
    """Aggregate causes / prevention / repair / consequences for every
    unique defect class found, so the frontend can render the tabbed
    result panel."""
    seen = []
    for det in detections:
        if det["class"] not in seen:
            seen.append(det["class"])

    report = {"diagnosis": [], "causes": [], "prevention": [], "repair": [], "consequences": []}
    if not seen:
        report["diagnosis"].append("No structural cracks or defects were detected in this image.")
        return report

    for class_name in seen:
        info = CRACK_ANALYSIS.get(class_name)
        report["diagnosis"].append(class_name)
        if info:
            report["causes"].extend(info["causes"])
            report["prevention"].extend(info["prevention"])
            report["repair"].extend(info["repair"])
            report["consequences"].extend(info["consequences"])

    # de-duplicate while preserving order
    for key in ("causes", "prevention", "repair", "consequences"):
        report[key] = list(dict.fromkeys(report[key]))

    return report


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img_bytes = file.read()
        result = run_model_analysis(img_bytes)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "mode": "model" if _model is not None else "demo",
        "model_load_error": _model_load_error,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("AI Based Structural Crack and Defect Detection")
    if _model is not None:
        print("Mode: PRODUCTION (crack_model.pt loaded)")
    else:
        print("Mode: DEMO (OpenCV heuristic fallback — no trained model found)")
        print("Place a trained 'crack_model.pt' (YOLOv8) next to app.py to switch to production mode.")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
