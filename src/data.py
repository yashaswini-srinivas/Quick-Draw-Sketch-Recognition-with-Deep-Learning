"""Data pipeline for the Quick, Draw! sketch-recognition project.

Steps:
  1. Download the *simplified* .ndjson file of each chosen category.
  2. Keep only drawings the game recognised (recognized == True).
  3. Take the first N drawings per class  ->  perfectly balanced by construction.
  4. Convert every drawing to
        (a) a stroke sequence  (dx, dy, pen_lift) of fixed length, and
        (b) a 28x28 grayscale image rendered from the SAME strokes,
     so image models and sequence models see exactly the same drawings.
  5. Stratified train / val / test split, cached as one compact .npz file.

Run:  python -m src.data --per-class 10000
"""
import argparse
import json
import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

CLASSES = ["cat", "dog", "car", "bus", "airplane",
           "house", "tree", "flower", "apple", "clock"]

BASE_URL = ("https://storage.googleapis.com/quickdraw_dataset/"
            "full/simplified/{}.ndjson")

MAX_LEN = 100      # max number of points kept per drawing
IMG_SIZE = 28      # image resolution
SEED = 42


# --------------------------------------------------------------------------
# Download + parse
# --------------------------------------------------------------------------
def download_category(name, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.ndjson")
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL.format(name.replace(" ", "%20")), path)
    return path


def read_drawings(path, n_samples):
    """Return the first n_samples recognised drawings (list of stroke lists)."""
    drawings = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            if not item.get("recognized", False):
                continue
            drawings.append(item["drawing"])   # [[xs, ys], [xs, ys], ...]
            if len(drawings) >= n_samples:
                break
    return drawings


# --------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------
def strokes_to_sequence(strokes, max_len=MAX_LEN):
    """Strokes -> array (max_len, 3) of (dx, dy, pen_lift) + true length.

    pen_lift = 1 on the last point of every stroke (pen leaves the paper).
    Coordinates are already scaled to 0-255 in the simplified format, so we
    take point-to-point differences and divide by 255.
    """
    pts = []
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            pts.append([xs[i], ys[i], 1.0 if i == len(xs) - 1 else 0.0])
    pts = np.array(pts, dtype=np.float32)

    deltas = np.zeros_like(pts)
    deltas[1:, :2] = pts[1:, :2] - pts[:-1, :2]
    deltas[:, :2] /= 255.0
    deltas[:, 2] = pts[:, 2]

    length = min(len(deltas), max_len)
    out = np.zeros((max_len, 3), dtype=np.float32)
    out[:length] = deltas[:length]
    return out, length


def strokes_to_image(strokes, size=IMG_SIZE, upto_stroke=None):
    """Render strokes to a size x size grayscale image in [0, 1]."""
    scale = 4                                   # draw large, then downsample
    canvas = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(canvas)
    use = strokes if upto_stroke is None else strokes[:upto_stroke]
    for stroke in use:
        pts = list(zip(stroke[0], stroke[1]))
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=255)
        else:
            draw.line(pts, fill=255, width=6)
    canvas = canvas.resize((size, size), Image.LANCZOS)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    m = arr.max()
    return arr / m if m > 0 else arr        # max-normalise: same step is used in app.py


# --------------------------------------------------------------------------
# Build dataset
# --------------------------------------------------------------------------
def build_dataset(classes=CLASSES, per_class=10000, raw_dir="data/raw",
                  out_path="data/quickdraw_10cls.npz"):
    seqs, lens, imgs, labels, strokes_all = [], [], [], [], []
    for label, name in enumerate(classes):
        path = download_category(name, raw_dir)
        drawings = read_drawings(path, per_class)
        if len(drawings) < per_class:
            raise RuntimeError(f"{name}: only {len(drawings)} recognised drawings")
        for d in drawings:
            s, l = strokes_to_sequence(d)
            seqs.append(s)
            lens.append(l)
            imgs.append(strokes_to_image(d))
            labels.append(label)
            strokes_all.append(json.dumps(d))     # kept for the early-guess experiment
        print(f"{name:10s} -> {len(drawings)} drawings")

    X_seq = np.stack(seqs)
    X_len = np.array(lens, dtype=np.int64)
    X_img = np.stack(imgs)[:, None, :, :]          # (N, 1, 28, 28)
    y = np.array(labels, dtype=np.int64)
    raw = np.array(strokes_all, dtype=object)

    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X_seq=X_seq, X_len=X_len, X_img=X_img, y=y,
                        raw=raw, train_idx=tr, val_idx=va, test_idx=te,
                        classes=np.array(classes))
    print(f"Saved {out_path}: {len(y)} samples "
          f"(train {len(tr)}, val {len(va)}, test {len(te)})")


def load_dataset(path="data/quickdraw_10cls.npz"):
    return np.load(path, allow_pickle=True)


def make_synthetic(n_per_class=60, n_classes=4, seed=0):
    """Tiny fake dataset used only for smoke tests (no download needed)."""
    rng = np.random.default_rng(seed)
    seqs, lens, imgs, labels, raws = [], [], [], [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            strokes = []
            for _ in range(rng.integers(1, 4)):
                n = int(rng.integers(5, 20))
                xs = np.clip(rng.integers(0, 256, n) // (c + 1) * (c + 1), 0, 255).tolist()
                ys = rng.integers(0, 256, n).tolist()
                strokes.append([xs, ys])
            s, l = strokes_to_sequence(strokes)
            seqs.append(s); lens.append(l)
            imgs.append(strokes_to_image(strokes)); labels.append(c)
            raws.append(json.dumps(strokes))
    y = np.array(labels)
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    return {"X_seq": np.stack(seqs), "X_len": np.array(lens),
            "X_img": np.stack(imgs)[:, None], "y": y,
            "raw": np.array(raws, dtype=object),
            "train_idx": tr, "val_idx": va, "test_idx": te,
            "classes": np.array([f"c{i}" for i in range(n_classes)])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10000)
    ap.add_argument("--out", default="data/quickdraw_10cls.npz")
    args = ap.parse_args()
    build_dataset(per_class=args.per_class, out_path=args.out)
