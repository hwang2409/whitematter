"""Generate server.py content for EC2 inference instance (image models)."""


def get_server_py_content() -> str:
    """Return the full Python source for the inference server (Flask + subprocess infer)."""
    return r'''
import io
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)
BASE = Path("/opt/whitematter-infer")
MODEL_BIN = BASE / "model.bin"
INFER_EXE = BASE / "infer"
CONFIG_PATH = BASE / "config.json"

_config = None

def load_config():
    global _config
    if _config is None and CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
    return _config

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    config = load_config()
    if not config:
        return jsonify({"error": "config.json not found"}), 500

    target_size = tuple(config["target_size"])
    channels = config["channels"]
    mean = np.array(config["mean"]).reshape(-1, 1, 1)
    std = np.array(config["std"]).reshape(-1, 1, 1)
    class_names = config["class_names"]

    image = Image.open(io.BytesIO(file.read()))
    if channels == 3:
        image = image.convert("RGB")
    else:
        image = image.convert("L")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    if channels == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[np.newaxis, ...]
    arr = (arr - mean) / std
    arr = arr[np.newaxis, ...]
    input_tensor = np.ascontiguousarray(arr, dtype=np.float32)

    TENSOR_MAGIC = 0x54454E53
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("I", TENSOR_MAGIC))
            f.write(struct.pack("I", len(input_tensor.shape)))
            for dim in input_tensor.shape:
                f.write(struct.pack("Q", dim))
            f.write(input_tensor.tobytes())

        result = subprocess.run(
            [str(INFER_EXE), str(MODEL_BIN), path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(BASE),
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr or "Inference failed"}), 500

        json_line = None
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                json_line = line
                break
        if not json_line:
            return jsonify({"error": "No JSON from infer"}), 500

        out = json.loads(json_line)
        pred = out["predicted_class"]
        probs = out["probabilities"]
        return jsonify({
            "predicted_class": pred,
            "class_name": class_names[pred],
            "confidence": float(probs[pred]),
            "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        })
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
'''
