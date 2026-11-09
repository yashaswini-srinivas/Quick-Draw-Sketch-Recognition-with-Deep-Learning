"""Gradio demo: draw a sketch, get the top-3 predicted classes.

    python app.py

Uses the CNN checkpoint (checkpoints/cnn.pt). A browser canvas only gives us the
final picture, so the stroke-sequence models are not used here.
"""
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.models import build_model

CKPT = "checkpoints/cnn.pt"
ckpt = torch.load(CKPT, map_location="cpu")
CLASSES = ckpt["classes"]
model = build_model(ckpt["name"], len(CLASSES))
model.load_state_dict(ckpt["state"])
model.eval()


def preprocess(sketch):
    """Canvas -> 28x28 max-normalised ink image, mimicking the training images."""
    if isinstance(sketch, dict):                       # Gradio 5 ImageEditor format
        sketch = sketch.get("composite")
    if sketch is None:
        return None
    arr = np.asarray(sketch).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:            # composite RGBA onto white
        a = arr[..., 3:4] / 255.0
        arr = arr[..., :3] * a + 255.0 * (1 - a)
    gray = arr.mean(2) if arr.ndim == 3 else arr
    ink = 255.0 - gray                                 # strokes -> bright
    ys, xs = np.where(ink > 30)
    if len(ys) == 0:
        return None
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)  # pad to square, centred
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop
    img = Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return out / out.max() if out.max() > 0 else out


def classify(sketch):
    x = preprocess(sketch)
    if x is None:
        return {}
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None, None].float())
        probs = torch.softmax(logits, dim=1)[0].numpy()
    top = np.argsort(-probs)[:3]
    return {CLASSES[i]: float(probs[i]) for i in top}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Sketchpad(label=f"Draw one of: {', '.join(CLASSES)}"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 predictions"),
    title="Quick, Draw! sketch recogniser",
    description="Draw a simple doodle and the CNN guesses what it is.",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
