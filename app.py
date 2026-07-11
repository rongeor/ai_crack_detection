"""
AI-Based Structural Crack and Defect Detection System
Flask Web Application — Dual Model Version

Models:
  crack_model.pt  → pothole detection
  crack_model2.pt → Crack-alligator, Crack-long, Crack-trans, pothole

Run: python app.py  →  open http://localhost:5000
"""

import os, io, base64, json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2

# ── Load both models ──────────────────────────────────────────────────
try:
    from ultralytics import YOLO

    # Model 1 — pothole detection (your first model)
    MODEL1_PATH = Path("crack_model.pt")
    model1      = YOLO(str(MODEL1_PATH)) if MODEL1_PATH.exists() else None
    if model1:
        print(f"✅ Model 1 loaded: {MODEL1_PATH}  (pothole)")
    else:
        print(f"⚠️  crack_model.pt not found — skipping Model 1")

    # Model 2 — crack types (your new model best(1).pt)
    # Rename best(1).pt to crack_model2.pt in your folder
    MODEL2_PATH = Path("crack_model2.pt")
    model2      = YOLO(str(MODEL2_PATH)) if MODEL2_PATH.exists() else None
    if model2:
        print(f"✅ Model 2 loaded: {MODEL2_PATH}  (alligator, long, trans, pothole)")
    else:
        print(f"⚠️  crack_model2.pt not found — rename best(1).pt to crack_model2.pt")

    USE_MODEL = model1 is not None or model2 is not None

except ImportError:
    model1 = model2 = None
    USE_MODEL = False
    print("⚠️  ultralytics not installed — demo mode")

# ── Class names for each model ────────────────────────────────────────
MODEL1_CLASSES = ["pothole"]
MODEL2_CLASSES = ["Crack-alligator", "Crack-long", "Crack-trans", "pothole"]

# ── Full analysis knowledge base ──────────────────────────────────────
# Keys must match the class names from both models (case-insensitive lookup)
CRACK_ANALYSIS = {
    "pothole": {
        "full_name"   : "Road Pothole",
        "severity"    : "Critical",
        "color"       : "#FF2222",
        "icon"        : "🔴",
        "causes"      : [
            "Repeated traffic overloading beyond pavement design capacity",
            "Water infiltration through surface cracks eroding the sub-base",
            "Poor drainage leading to prolonged water saturation",
            "Aging and oxidation of bituminous binder material",
            "Freeze-thaw expansion cycles breaking apart asphalt layers",
        ],
        "prevention"  : [
            "Seal surface cracks immediately before water penetrates",
            "Improve road drainage — clear kerb outlets and install edge drains",
            "Enforce axle load limits on commercial vehicles",
            "Schedule preventive resurfacing every 5–7 years",
            "Use polymer-modified bitumen (PMB) for longer service life",
        ],
        "repair"      : [
            "Clear all debris, water, and loose material from the pothole",
            "Saw-cut edges square to create a clean repair boundary",
            "Apply bituminous tack coat to all cut surfaces",
            "Fill with hot-mix asphalt in 50mm compacted layers",
            "Compact firmly with a vibratory roller or plate compactor",
            "Apply surface sealant to prevent future water ingress",
        ],
        "consequences": [
            "Vehicle tyre and suspension damage — avg. ₹15,000–₹40,000 per incident",
            "High risk of motorcycle and bicycle accidents",
            "Size doubles within 3–6 months under continued traffic loading",
            "Progressive structural failure of the full road pavement",
        ],
        "urgency"     : "Emergency — repair within 24–72 hours",
    },
    "crack-alligator": {
        "full_name"   : "Alligator (Fatigue) Cracking",
        "severity"    : "Critical",
        "color"       : "#CC0000",
        "icon"        : "🔴",
        "causes"      : [
            "Fatigue failure from repeated traffic loading beyond design life",
            "Pavement structural thickness insufficient for actual traffic volume",
            "Weak or moisture-saturated subgrade reducing structural capacity",
            "Excessive overloading by heavy commercial and goods vehicles",
            "Absence of structural maintenance during service life",
        ],
        "prevention"  : [
            "Conduct structural pavement design using accurate traffic count data",
            "Ensure proper drainage to protect subgrade bearing capacity",
            "Monitor road deflection with Falling Weight Deflectometer annually",
            "Plan overlay before fatigue cracking reaches the surface",
            "Enforce vehicle overloading regulations strictly",
        ],
        "repair"      : [
            "Perform full-depth reclamation — remove and recycle the failed layer",
            "Investigate and address subgrade moisture if present",
            "Reconstruct with structurally adequate layer thicknesses",
            "Apply geogrid reinforcement between layers to prevent recurrence",
            "Consider recycled asphalt pavement (RAP) for cost and sustainability",
        ],
        "consequences": [
            "Imminent complete structural collapse of road surface",
            "Severe vehicle damage and very high accident risk",
            "Full reconstruction cost is 20–50× higher than preventive maintenance",
            "Road becomes impassable during heavy rainfall",
        ],
        "urgency"     : "Emergency — immediate structural repair required",
    },
    "crack-long": {
        "full_name"   : "Longitudinal Crack",
        "severity"    : "High",
        "color"       : "#FF8800",
        "icon"        : "🟠",
        "causes"      : [
            "Reflective cracking upward from joints in underlying base layers",
            "Differential settlement from uneven subgrade compaction",
            "Asphalt layer shrinkage during thermal cooling overnight",
            "Lane-edge support failure due to weak or eroded road shoulders",
            "Excessive tensile stress from heavy vehicle channelisation",
        ],
        "prevention"  : [
            "Ensure uniform compaction of all sub-layers during construction",
            "Install geosynthetic interlayer fabric to suppress reflective cracking",
            "Maintain adequate road shoulders for lateral structural support",
            "Apply annual fog seal or chip seal for UV and moisture protection",
        ],
        "repair"      : [
            "Route the crack to create uniform walls and remove debris",
            "Clean thoroughly with compressed air",
            "Apply hot-pour rubberised crack filler to full depth",
            "Sand the surface before filler sets to restore skid resistance",
            "For cracks wider than 25mm — saw-cut and patch with cold-mix asphalt",
        ],
        "consequences": [
            "Water infiltration causing pothole formation in 1–2 monsoon seasons",
            "Progressive crack widening under traffic and thermal cycling",
            "Structural weakening of base and subgrade layers",
        ],
        "urgency"     : "High priority — repair within 2 weeks",
    },
    "crack-trans": {
        "full_name"   : "Transverse Crack",
        "severity"    : "Medium",
        "color"       : "#FFCC00",
        "icon"        : "🟡",
        "causes"      : [
            "Thermal expansion and contraction stress across the road width",
            "Low-temperature hardening (embrittlement) of aged asphalt binder",
            "Reflection cracking from transverse joints in concrete base",
            "Asphalt layer shrinkage during initial curing period",
        ],
        "prevention"  : [
            "Specify polymer-modified binder with wider temperature performance range",
            "Use PG binder grade appropriate for local climate extremes",
            "Include transverse crack sealing in routine maintenance programme",
            "Monitor Pavement Condition Index (PCI) annually",
        ],
        "repair"      : [
            "Clean crack with compressed air to remove debris and moisture",
            "Apply rubberised crack sealant for cracks narrower than 19mm",
            "For wider cracks — clean, tack coat, and surface patch",
            "Inspect adjacent area for hidden sub-surface damage",
        ],
        "consequences": [
            "Water penetration causing sub-base saturation and softening",
            "Rapid deterioration into potholes during monsoon season",
            "Reduced riding quality increasing vehicle operating costs by 10–20%",
        ],
        "urgency"     : "Moderate — repair within 1 month",
    },
}

SEV_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def get_analysis(class_name):
    """Look up analysis by class name — case-insensitive."""
    key = class_name.lower().replace("_", "-")
    # direct match
    if key in CRACK_ANALYSIS:
        return CRACK_ANALYSIS[key]
    # partial match fallback
    for k in CRACK_ANALYSIS:
        if k in key or key in k:
            return CRACK_ANALYSIS[k]
    # default
    return {
        "full_name"   : class_name.replace("-", " ").title(),
        "severity"    : "Medium",
        "color"       : "#888888",
        "icon"        : "⚪",
        "causes"      : ["Road surface deterioration detected"],
        "prevention"  : ["Schedule inspection and maintenance"],
        "repair"      : ["Consult a road engineer for repair assessment"],
        "consequences": ["May worsen if left untreated"],
        "urgency"     : "Schedule inspection",
    }


def run_inference(img_array):
    """
    Run both models on the image.
    Merge results, remove duplicate detections of the same area.
    """
    all_detections = []

    # ── Model 1 — pothole ──────────────────────────────────────────────
    if model1:
        res = model1.predict(img_array, conf=0.25, verbose=False)[0]
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = MODEL1_CLASSES[cls_id] if cls_id < len(MODEL1_CLASSES) else "pothole"
                info   = get_analysis(name)
                all_detections.append({
                    "class"       : name,
                    "full_name"   : info["full_name"],
                    "confidence"  : round(conf * 100, 1),
                    "bbox"        : [int(v) for v in box.xyxy[0].tolist()],
                    "severity"    : info["severity"],
                    "urgency"     : info["urgency"],
                    "color"       : info["color"],
                    "icon"        : info["icon"],
                    "causes"      : info["causes"],
                    "prevention"  : info["prevention"],
                    "repair"      : info["repair"],
                    "consequences": info["consequences"],
                    "source"      : "model1",
                })

    # ── Model 2 — crack types ──────────────────────────────────────────
    if model2:
        res = model2.predict(img_array, conf=0.25, verbose=False)[0]
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = MODEL2_CLASSES[cls_id] if cls_id < len(MODEL2_CLASSES) else "unknown"
                info   = get_analysis(name)
                all_detections.append({
                    "class"       : name,
                    "full_name"   : info["full_name"],
                    "confidence"  : round(conf * 100, 1),
                    "bbox"        : [int(v) for v in box.xyxy[0].tolist()],
                    "severity"    : info["severity"],
                    "urgency"     : info["urgency"],
                    "color"       : info["color"],
                    "icon"        : info["icon"],
                    "causes"      : info["causes"],
                    "prevention"  : info["prevention"],
                    "repair"      : info["repair"],
                    "consequences": info["consequences"],
                    "source"      : "model2",
                })

    # ── Remove duplicate pothole detections (keep highest confidence) ──
    # Both models detect potholes — keep only the one with higher confidence
    potholes = [d for d in all_detections if "pothole" in d["class"].lower()]
    others   = [d for d in all_detections if "pothole" not in d["class"].lower()]

    if len(potholes) > 1:
        # Check IoU — if two pothole boxes overlap a lot, keep highest conf
        def iou(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            if inter == 0:
                return 0
            area_a = (a[2]-a[0]) * (a[3]-a[1])
            area_b = (b[2]-b[0]) * (b[3]-b[1])
            return inter / (area_a + area_b - inter)

        kept = []
        for p in sorted(potholes, key=lambda x: x["confidence"], reverse=True):
            if not any(iou(p["bbox"], k["bbox"]) > 0.5 for k in kept):
                kept.append(p)
        potholes = kept

    all_detections = others + potholes

    # Sort by severity
    all_detections.sort(
        key=lambda d: SEV_ORDER.get(d["severity"], 0),
        reverse=True
    )

    return all_detections


def demo_analysis(img_array):
    """Fallback when no models are loaded — basic image heuristics."""
    gray   = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 50, 150)
    edge_d = np.sum(edges > 0) / edges.size
    dark_r = np.sum(gray < 60)  / gray.size
    h, w   = img_array.shape[:2]

    if dark_r > 0.12:
        cls = "pothole"
    elif edge_d > 0.12:
        cls = "Crack-alligator"
    elif edge_d > 0.07:
        cls = "Crack-long"
    else:
        cls = "Crack-trans"

    info = get_analysis(cls)
    return [{
        "class"       : cls,
        "full_name"   : info["full_name"],
        "confidence"  : round(min(55 + edge_d * 200, 88), 1),
        "bbox"        : [int(w*.15), int(h*.15), int(w*.85), int(h*.85)],
        "severity"    : info["severity"],
        "urgency"     : info["urgency"],
        "color"       : info["color"],
        "icon"        : info["icon"],
        "causes"      : info["causes"],
        "prevention"  : info["prevention"],
        "repair"      : info["repair"],
        "consequences": info["consequences"],
        "source"      : "demo",
    }]


def annotate_image(img_array, detections):
    """Draw bounding boxes and labels on the image."""
    annotated = img_array.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Convert hex colour to BGR
        hex_c  = det["color"].lstrip("#")
        r,g,b  = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        bgr    = (b, g, r)
        # Box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), bgr, 3)
        # Label background
        label  = f"{det['icon']} {det['full_name']}  {det['confidence']}%"
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1-th-12), (x1+tw+10, y1), bgr, -1)
        cv2.putText(annotated, label, (x1+5, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return annotated


# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    models_status = {
        "model1": model1 is not None,
        "model2": model2 is not None,
    }
    return render_template("index.html",
                           use_model=USE_MODEL,
                           models=models_status)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read image
    img_bytes = file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Run inference
    if USE_MODEL:
        detections = run_inference(img)
    else:
        detections = demo_analysis(img)

    # Annotate image
    annotated = annotate_image(img, detections)
    _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64   = base64.b64encode(buf).decode("utf-8")

    # Overall severity
    overall = (
        max(detections, key=lambda d: SEV_ORDER.get(d["severity"], 0))["severity"]
        if detections else "No defects detected"
    )

    return jsonify({
        "status"           : "success",
        "num_detections"   : len(detections),
        "overall_severity" : overall,
        "detections"       : detections,
        "annotated_image"  : img_b64,
        "models_used"      : {
            "model1": model1 is not None,
            "model2": model2 is not None,
        }
    })



@app.route("/status")
def status():
    return jsonify({
        "model1": model1 is not None,
        "model2": model2 is not None,
    })

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI Crack Detection System — Dual Model")
    print(f"  Model 1 (pothole)      : {'✅ loaded' if model1 else '❌ not found'}")
    print(f"  Model 2 (crack types)  : {'✅ loaded' if model2 else '❌ not found'}")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
