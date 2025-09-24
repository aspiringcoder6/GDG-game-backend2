from flask import Flask, request, jsonify
import cv2, torch
import numpy as np
from ModelArchitect import VGG_64
from flask_cors import CORS
import time, base64, re
from io import BytesIO
from PIL import Image
from typing import Tuple
from huggingface_hub import hf_hub_download
import torch
from dotenv import load_dotenv
load_dotenv()
import os
token = os.getenv("HF_TOKEN")


# Load the model
model = None

def get_model():
    global model
    if model is None:
        token = os.getenv("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id="Ninhminhhieu/vgg_64",
            filename="vgg_64.pt",
            token=token
        )
        model = torch.load(model_path, map_location="cpu")
    return model

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES_VN = np.array([
    'Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Bánh cá', 'Mặt nạ',
    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'
])

# --- Load model ---
model = VGG_64(len(CLASSES_VN))
state_dict = torch.load(model_path, map_location=device)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    model.load_state_dict(state_dict["state_dict"])
else:
    model.load_state_dict(state_dict)
model.to(device)
model.eval()


# --- Image preprocessing ---
IMG_SIZE = (64, 64)
MASK_THRESHOLD = 50  # 0-255, for binarization
def torch_process_image(canvas: np.ndarray, size=IMG_SIZE):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    inp = cv2.resize(bw, size).astype(np.float32) / 255.0  # [0,1]

    # apply same normalization as training: (x - 0.5) / 0.5
    inp = (inp - 0.5) / 0.5  # -> [-1,1]

    cv2.imwrite("debug_input.png", ((inp + 1) * 127.5).astype(np.uint8))  # visualize

    return torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def torch_predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch_process_image(image).to(device)
    with torch.no_grad():
        out = model(tensor)
        if out.dim() == 4:
            out = out.view(out.size(0), -1)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return np.argsort(probs)[-3:][::-1]


# --- Flask route ---
@app.route("/predict", methods=["POST"])
def predict():
    model=get_model()
    try:
        if "frame" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["frame"]
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # crop + predict like before...
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        box_range = 225
        x0 = max(0, center_x - box_range)
        y0 = max(0, center_y - box_range)
        x1 = min(w, center_x + box_range)
        y1 = min(h, center_y + box_range)
        cropped = frame[y0:y1, x0:x1]

        top_indices = torch_predict(model, cropped, device)
        top_classes = [CLASSES_VN[i] for i in top_indices]

        return jsonify({
            "prediction": top_classes[0],
            "top5": top_classes,
            "indices": top_indices.tolist()
        })
    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
