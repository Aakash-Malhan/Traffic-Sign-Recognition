import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

DEVICE = "cpu"
IMG_SIZE = 224

def load_label_map(path="label_map.json"):
    with open(path, "r") as f:
        d = json.load(f)
    # keys saved as strings; convert to int
    return {int(k): v for k, v in d.items()}

def build_model(num_classes: int):
    model = models.resnet18(weights=None)  # head will match your trained checkpoint
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_everything(weights_path="model.pth", label_map_path="label_map.json"):
    assert Path(weights_path).exists(), f"Missing {weights_path}. Upload your trained weights."
    assert Path(label_map_path).exists(), f"Missing {label_map_path}. Upload your label map."
    idx_to_class = load_label_map(label_map_path)
    num_classes = len(idx_to_class)
    model = build_model(num_classes).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    preproc = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, idx_to_class, preproc

# ---- Load once at import time (simple + fast on CPU Spaces)
MODEL, IDX_TO_CLASS, PREPROC = load_everything()

@torch.inference_mode()
def predict(image: Image.Image):
    if image is None:
        return "", {}
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = PREPROC(image).unsqueeze(0).to(DEVICE)
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    topk = probs.argsort()[::-1][:5]
    top1_label = IDX_TO_CLASS[int(topk[0])]
    dist = {IDX_TO_CLASS[int(i)]: float(probs[i]) for i in topk}
    return top1_label, dist

with gr.Blocks(title="Traffic Sign Recognition (GTSRB) — ResNet18") as demo:
    gr.Markdown(
        "# 🚦 Traffic Sign Recognition (GTSRB)\n"
        "Upload an image or use your webcam. The model is a ResNet-18 fine-tuned on GTSRB."
    )
    with gr.Row():
        inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Image")
        with gr.Column():
            top = gr.Textbox(label="Top Prediction")
            dist = gr.Label(num_top_classes=5, label="Top-5 Probabilities")
            btn = gr.Button("Predict", variant="primary")
            clear = gr.Button("Clear")

    btn.click(predict, inp, [top, dist])
    clear.click(lambda: (None, "", {}), None, [inp, top, dist], queue=False)

if __name__ == "__main__":
    demo.launch()
