"""
Plant Disease Detection — Flask backend.

Loads the trained MobileNetV2 Keras model and serves predictions
for uploaded leaf images. Preprocessing matches training:

- image_dataset_from_directory used RGB images resized to 224x224
- pixel values stayed in 0–255 (float32)
- MobileNetV2 preprocess_input (x / 127.5 - 1) is already inside the
  saved model as TrueDivide + Subtract layers, so it is NOT applied again
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "plant_disease_mobilenetv2.keras"
DEFAULT_LABELS_PATH = BASE_DIR / "models" / "class_names.txt"

# Match training notebook: IMG_SIZE = (224, 224)
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_model = None
_model_error = None
_class_names: list[str] = []
_labels_error = None


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()


def labels_path() -> Path:
    return Path(os.environ.get("CLASS_NAMES_PATH", str(DEFAULT_LABELS_PATH))).expanduser()


def load_class_names() -> list[str]:
    global _class_names, _labels_error
    if _class_names:
        return _class_names
    path = labels_path()
    if not path.is_file():
        _labels_error = f"Class label file not found at {path}"
        raise FileNotFoundError(_labels_error)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 38:
        _labels_error = f"Expected 38 class names, found {len(names)} in {path}"
        raise ValueError(_labels_error)
    _class_names = names
    _labels_error = None
    return _class_names


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    path = model_path()
    if not path.is_file():
        _model_error = (
            f"Trained model file not found at {path}. "
            "Place plant_disease_mobilenetv2.keras in web_app/models/ "
            "or set the MODEL_PATH environment variable."
        )
        raise FileNotFoundError(_model_error)
    try:
        _model = load_model(path)
        _model_error = None
        return _model
    except Exception as exc:  # noqa: BLE001 — report any load failure to the API
        _model_error = f"Failed to load the trained model: {exc}"
        raise RuntimeError(_model_error) from exc


def allowed_file(filename: str, mimetype: str | None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    if mimetype and mimetype.lower() not in ALLOWED_MIMETYPES:
        return False
    return True


def preprocess_image(file_storage) -> np.ndarray:
    """Resize to 224x224 RGB and keep 0–255 float32 values, matching training."""
    try:
        image = Image.open(file_storage.stream)
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise ValueError("Could not read the uploaded image.") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    # image_dataset_from_directory uses bilinear interpolation by default.
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (224, 224, 3):
        raise ValueError("Image preprocessing produced an unexpected shape.")
    return np.expand_dims(array, axis=0)


def format_label(raw: str) -> str:
    plant, _, condition = raw.partition("___")
    plant = plant.replace("_", " ").replace(",", ", ").strip()
    condition = condition.replace("_", " ").strip() if condition else raw.replace("_", " ")
    return f"{plant} — {condition}" if condition else plant


def health_status(raw_class: str) -> str:
    return "Healthy" if raw_class.lower().endswith("healthy") else "Diseased"


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"ok": False, "error": "Image is too large. Please upload a file under 8 MB."}), 413


@app.route("/")
def index():
    labels = []
    labels_available = True
    try:
        labels = load_class_names()
    except (FileNotFoundError, ValueError):
        labels_available = False
    return render_template(
        "index.html",
        class_names=labels,
        labels_available=labels_available,
        num_classes=len(labels) if labels else 38,
    )


@app.route("/health")
def health():
    model_ok = False
    labels_ok = False
    errors = []
    try:
        load_class_names()
        labels_ok = True
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    try:
        get_model()
        model_ok = True
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    status = "ok" if model_ok and labels_ok else "degraded"
    return jsonify(
        {
            "status": status,
            "model_loaded": model_ok,
            "labels_loaded": labels_ok,
            "model_path": str(model_path()),
            "labels_path": str(labels_path()),
            "image_size": list(IMG_SIZE),
            "num_classes": len(_class_names) if _class_names else None,
            "errors": errors,
        }
    ), (200 if model_ok and labels_ok else 503)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        class_names = load_class_names()
        model = get_model()
    except Exception as exc:  # noqa: BLE001
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "hint": "The backend cannot run predictions until the real trained model and class labels are available.",
            }
        ), 503

    if "image" not in request.files and "file" not in request.files:
        return jsonify({"ok": False, "error": "No image file was uploaded. Use form field name 'image'."}), 400

    uploaded = request.files.get("image") or request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"ok": False, "error": "No image file was selected."}), 400

    if not allowed_file(uploaded.filename, uploaded.mimetype):
        return jsonify(
            {
                "ok": False,
                "error": "Unsupported file type. Please upload a JPG, JPEG, or PNG leaf image.",
            }
        ), 400

    try:
        batch = preprocess_image(uploaded)
        probabilities = model.predict(batch, verbose=0)[0]
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Prediction failed: {exc}"}), 500

    if probabilities.shape[0] != len(class_names):
        return jsonify(
            {
                "ok": False,
                "error": (
                    f"Model output size ({probabilities.shape[0]}) does not match "
                    f"the number of class labels ({len(class_names)})."
                ),
            }
        ), 500

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    raw_class = class_names[index]

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "class_name": class_names[int(i)],
            "display_name": format_label(class_names[int(i)]),
            "confidence": round(float(probabilities[int(i)]) * 100, 2),
        }
        for i in top_indices
    ]

    return jsonify(
        {
            "ok": True,
            "predicted_class": raw_class,
            "display_name": format_label(raw_class),
            "confidence": round(confidence * 100, 2),
            "status": health_status(raw_class),
            "confidence_note": (
                "Confidence is the model's softmax output score for the top class. "
                "It is not a guarantee that the prediction is correct."
            ),
            "top_predictions": top_predictions,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
