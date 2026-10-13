"""
AI-Based Structural Crack and Defect Detection System
Flask Web Application — Dual Model Version (ONNX RUNTIME)

Models:
  crack_model.onnx  → pothole detection
  crack_model2.onnx → Crack-alligator, Crack-long, Crack-trans, pothole
"""

import os, io, base64, json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2

# ── Load both ONNX models ─────────────────────────────────────────────
try:
    import onnxruntime as ort

    MODEL1_PATH = Path("crack_model.onnx")
    if MODEL1_PATH.exists():
        model1 = ort.InferenceSession(str(MODEL1_PATH))
        print(f"✅ Model 1 loaded: {MODEL1_PATH} (pothole)")
    else:
        model1 = None
        print(f"⚠️ {MODEL1_PATH} not found — skipping Model 1")

    MODEL2_PATH = Path("crack_model2.onnx")
    if MODEL2_PATH.exists():
        model2 = ort.InferenceSession(str(MODEL2_PATH))
        print(f"✅ Model 2 loaded: {MODEL2_PATH} (alligator, long, trans, pothole)")
    else:
        model2 = None
        print(f"⚠️ {MODEL2_PATH} not found")

    USE_MODEL = model1 is not None or model2 is not None

except ImportError:
    model1 = model2 = None
    USE_MODEL = False
    print("⚠️ onnxruntime not installed — demo mode")

# ── Class names for each model ────────────────────────────────────────
MODEL1_CLASSES = ["pothole"]
MODEL2_CLASSES = ["Crack-alligator", "Crack-long", "Crack-trans", "pothole"]

# ── Full analysis knowledge base ──────────────────────────────────────
CRACK_ANALYSIS = {
    "pothole": {
        "full_name"   : "Road Pothole",
        "severity"    : "Critical",
        "color"       : "#FF2222",
        "icon"        : "🔴",
        "causes"      : ["Repeated traffic overloading", "Water infiltration", "Poor drainage"],
        "prevention"  : ["Seal surface cracks immediately", "Improve road drainage"],
        "repair"      : ["Clear debris", "Saw-cut edges", "Apply hot-mix asphalt"],
        "consequences": ["Vehicle damage", "High risk of accidents"],
        "urgency"     : "Emergency — repair within 24–72 hours",
    },
    "crack-alligator": {
        "full_name"   : "Alligator (Fatigue) Cracking",
        "severity"    : "Critical",
        "color"       : "#CC0000",
        "icon"        : "🔴",
        "causes"      : ["Fatigue failure from traffic", "Weak subgrade"],
        "prevention"  : ["Proper drainage", "Enforce overloading regulations"],
        "repair"      : ["Full-depth reclamation", "Address subgrade moisture"],
        "consequences": ["Complete structural collapse", "Severe vehicle damage"],
        "urgency"     : "Emergency — immediate structural repair required",
    },
    "crack-long": {
        "full_name"   : "Longitudinal Crack",
        "severity"    : "High",
        "color"       : "#FF8800",
        "icon"        : "🟠",
        "causes"      : ["Differential settlement", "Asphalt layer shrinkage"],
        "prevention"  : ["Uniform compaction", "Maintain road shoulders"],
        "repair"      : ["Apply hot-pour rubberised crack filler"],
        "consequences": ["Water infiltration", "Structural weakening"],
        "urgency"     : "High priority — repair within 2 weeks",
    },
    "crack-trans": {
        "full_name"   : "Transverse Crack",
        "severity"    : "Medium",
        "color"       : "#FFCC00",
        "icon"        : "🟡",
        "causes"      : ["Thermal expansion/contraction", "Low-temperature hardening"],
        "prevention"  : ["Use appropriate PG binder grade"],
        "repair"      : ["Clean crack", "Apply rubberised crack sealant"],
        "consequences": ["Sub-base saturation", "Rapid deterioration into potholes"],
        "urgency"     : "Moderate — repair within 1 month",
    },
}

SEV_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

def get_analysis(class_name):
    key = class_name.lower().replace("_", "-")
    if key in CRACK_ANALYSIS:
        return CRACK_ANALYSIS[key]
    for k in CRACK_ANALYSIS:
        if k in key or key in k:
            return CRACK_ANALYSIS[k]
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

# ── ONNX Helper Functions ──────────────────────────────────────────────
def preprocess_image(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_data = img_resized.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

def process_yolo_outputs(output, img_w, img_h, classes_list, conf_thresh=0.25):
    predictions = np.squeeze(output[0]).T
    boxes, scores, class_ids = [], [], []
    
    x_factor = img_w / 640.0
    y_factor = img_h / 640.0

    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        max_score = classes_scores[class_id]

        if max_score >= conf_thresh:
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            x1 = int((xc - w/2) * x_factor)
            y1 = int((yc - h/2) * y_factor)
            boxes.append([x1, y1, int(w * x_factor), int(h * y_factor)])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45)
    results = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            name = classes_list[class_ids[i]] if class_ids[i] < len(classes_list) else "unknown"
            results.append({
                "class": name,
                "confidence": round(scores[i] * 100, 1),
                "bbox": [x, y, x + w, y + h]
            })
    return results

def run_inference(img_array):
    img_h, img_w = img_array.shape[:2]
    input_tensor = preprocess_image(img_array)
    all_detections = []

    # Model 1
    if model1:
        inputs = {model1.get_inputs()[0].name: input_tensor}
        outputs = model1.run(None, inputs)
        results = process_yolo_outputs(outputs, img_w, img_h, MODEL1_CLASSES)
        for res in results:
            info = get_analysis(res["class"])
            res.update({
                "full_name": info["full_name"], "severity": info["severity"],
                "urgency": info["urgency"], "color": info["color"],
                "icon": info["icon"], "causes": info["causes"],
                "prevention": info["prevention"], "repair": info["repair"],
                "consequences": info["consequences"], "source": "model1"
            })
            all_detections.append(res)

    # Model 2
    if model2:
        inputs = {model2.get_inputs()[0].name: input_tensor}
        outputs = model2.run(None, inputs)
        results = process_yolo_outputs(outputs, img_w, img_h, MODEL2_CLASSES)
        for res in results:
            info = get_analysis(res["class"])
            res.update({
                "full_name": info["full_name"], "severity": info["severity"],
                "urgency": info["urgency"], "color": info["color"],
                "icon": info["icon"], "causes": info["causes"],
                "prevention": info["prevention"], "repair": info["repair"],
                "consequences": info["consequences"], "source": "model2"
            })
            all_detections.append(res)

    # Deduplicate Potholes
    potholes = [d for d in all_detections if "pothole" in d["class"].lower()]
    others   = [d for d in all_detections if "pothole" not in d["class"].lower()]

    if len(potholes) > 1:
        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            return inter / (((a[2]-a[0])*(a[3]-a[1])) + ((b[2]-b[0])*(b[3]-b[1])) - inter + 1e-6)

        kept = []
        for p in sorted(potholes, key=lambda x: x["confidence"], reverse=True):
            if not any(iou(p["bbox"], k["bbox"]) > 0.5 for k in kept):
                kept.append(p)
        potholes = kept

    all_detections = others + potholes
    all_detections.sort(key=lambda d: SEV_ORDER.get(d["severity"], 0), reverse=True)
    return all_detections

# ── Flask Routes Below (Untouched) ────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

def demo_analysis(img_array):
    return []  # Simplified for space, keep your original demo if preferred

def annotate_image(img_array, detections):
    annotated = img_array.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        hex_c  = det["color"].lstrip("#")
        bgr    = (int(hex_c[4:6],16), int(hex_c[2:4],16), int(hex_c[0:2],16))
        cv2.rectangle(annotated, (x1,y1), (x2,y2), bgr, 3)
        label  = f"{det['icon']} {det['full_name']} {det['confidence']}%"
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1-th-12), (x1+tw+10, y1), bgr, -1)
        cv2.putText(annotated, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return annotated

@app.route("/")
def index():
    return render_template("index.html", use_model=USE_MODEL,
                           models={"model1": model1 is not None, "model2": model2 is not None})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    detections = run_inference(img) if USE_MODEL else demo_analysis(img)
    annotated = annotate_image(img, detections)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    
    overall = max(detections, key=lambda d: SEV_ORDER.get(d["severity"], 0))["severity"] if detections else "No defects"
    return jsonify({
        "status": "success", "num_detections": len(detections),
        "overall_severity": overall, "detections": detections,
        "annotated_image": base64.b64encode(buf).decode("utf-8"),
        "models_used": {"model1": model1 is not None, "model2": model2 is not None}
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
