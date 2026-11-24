import os
import io
import base64
import torch
from PIL import Image
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="../templates", static_folder="../static")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

predictor = None


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded. Please train or provide model checkpoint."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No image file provided in upload request"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid format. Supported formats: PNG, JPG, JPEG, WEBP"}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    try:
        image = Image.open(upload_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to decode image file: {str(e)}"}), 400

    result, cam_overlay = predictor.predict_with_visualization(image)

    base_name = os.path.splitext(filename)[0]
    result_filename = f"pred_{base_name}.png"
    gradcam_filename = f"gradcam_{base_name}.png"

    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
    gradcam_path = os.path.join(app.config["RESULT_FOLDER"], gradcam_filename)

    # Save diagnostic visualization
    _save_diagnostic_panel(image, result, result_path)

    gradcam_url = None
    if cam_overlay is not None:
        cam_overlay.save(gradcam_path, quality=95)
        gradcam_url = url_for("static", filename=f"results/{gradcam_filename}")

    return jsonify({
        "success": True,
        "filename": filename,
        "upload_url": url_for("static", filename=f"uploads/{filename}"),
        "result_url": url_for("static", filename=f"results/{result_filename}"),
        "gradcam_url": gradcam_url,
        "prediction": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "is_anomaly": result["is_anomaly"],
        "is_reliable": result["is_reliable"],
        "probabilities": result["probabilities_chart"],
    })


def _save_diagnostic_panel(image: Image.Image, result: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].set_title("Input Surveillance Frame", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    color = "#e53e3e" if result["is_anomaly"] else "#38a169"
    status_tag = "THREAT / ANOMALY DETECTED" if result["is_anomaly"] else "NORMAL SURVEILLANCE STATE"

    title_text = f"Class: {result['predicted_class'].upper()}\nConfidence: {result['confidence']:.2%}\n[{status_tag}]"
    axes[1].imshow(image)
    axes[1].set_title(title_text, fontsize=13, fontweight="bold", color=color)
    axes[1].axis("off")

    if result["is_anomaly"]:
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.08",
                              edgecolor=color, linewidth=5, facecolor="none",
                              transform=axes[1].transAxes)
        axes[1].add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def start_app(model, device, class_names, confidence_threshold=0.60, input_size=224, port=5000):
    global predictor
    from inference.predict import Predictor
    predictor = Predictor(
        model=model,
        device=device,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        input_size=input_size,
    )
    print(f"[*] Security Anomaly Detection Server online at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
