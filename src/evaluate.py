"""Evaluation, plots and the extra experiments.

    python -m src.evaluate --all                      # tables + plots for everything trained
    python -m src.evaluate --early-guess --models cnn bigru fusion --n-eval 3000
    python -m src.evaluate --learning-curve --models cnn bigru
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import load_dataset, strokes_to_image, strokes_to_sequence
from src.models import MODEL_INPUTS, build_model
from src.utils import get_device

RES = "results"
COLORS = {"image": "#2a6f97", "sequence": "#c8553d", "both": "#5b8c5a"}


def _save(fig, name):
    os.makedirs(RES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name), dpi=160)
    plt.close(fig)
    print("saved", os.path.join(RES, name))


def read_metrics(path=f"{RES}/metrics.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("top1", "top3", "macro_f1", "train_time_s", "latency_ms"):
            r[k] = float(r[k])
        r["params"] = int(r["params"])
        r["epochs_run"] = int(r["epochs_run"])
    return rows


def load_model(run, device):
    ckpt = torch.load(f"checkpoints/{run}.pt", map_location=device)
    model = build_model(ckpt["name"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt


# --------------------------------------------------------------------------
# Dataset figures
# --------------------------------------------------------------------------
def plot_balance(data):
    classes = list(data["classes"])
    counts = [int((data["y"] == i).sum()) for i in range(len(classes))]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(classes, counts, color=COLORS["image"])
    ax.set_ylabel("drawings")
    ax.set_title("Class balance (every class has the same number of samples)")
    ax.set_ylim(0, max(counts) * 1.15)
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30)
    _save(fig, "class_balance.png")


def plot_samples(data, per_class=6):
    classes = list(data["classes"])
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(per_class * 1.1, len(classes) * 1.1))
    for c in range(len(classes)):
        idx = np.where(data["y"] == c)[0][:per_class]
        for j, i in enumerate(idx):
            axes[c, j].imshow(data["X_img"][i, 0], cmap="gray_r")
            axes[c, j].axis("off")
        axes[c, 0].set_title(classes[c], fontsize=8, loc="left")
    _save(fig, "sample_drawings.png")


# --------------------------------------------------------------------------
# Per-model outputs
# --------------------------------------------------------------------------
def plot_history(run):
    p = f"{RES}/history_{run}.json"
    if not os.path.exists(p):
        return
    h = json.load(open(p))
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ep, [r["train_loss"] for r in h], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in h], label="validation")
    ax[0].set_title(f"{run}: loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [r["train_acc"] for r in h], label="train")
    ax[1].plot(ep, [r["val_acc"] for r in h], label="validation")
    ax[1].set_title(f"{run}: accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    _save(fig, f"curves_{run}.png")


def plot_confusion(run, data):
    logits = np.load(f"{RES}/test_logits_{run}.npy")
    y = data["y"][data["test_idx"]]
    classes = list(data["classes"])
    cm = confusion_matrix(y, logits.argmax(1))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalised): {run}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    _save(fig, f"confusion_{run}.png")
    with open(f"{RES}/classification_report_{run}.txt", "w") as f:
        f.write(classification_report(y, logits.argmax(1), target_names=classes, digits=4,
                                       zero_division=0))


def error_gallery(run, data, n=12):
    logits = np.load(f"{RES}/test_logits_{run}.npy")
    test_idx = data["test_idx"]
    y = data["y"][test_idx]
    wrong = np.where(logits.argmax(1) != y)[0][:n]
    classes = list(data["classes"])
    cols = 6
    rows = int(np.ceil(len(wrong) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for k, w in enumerate(wrong):
        a = axes.ravel()[k]
        a.imshow(data["X_img"][test_idx[w], 0], cmap="gray_r")
        a.set_title(f"true: {classes[y[w]]}\npred: {classes[logits[w].argmax()]}",
                    fontsize=7)
    _save(fig, f"errors_{run}.png")


def results_table(rows):
    rows = sorted(rows, key=lambda r: -r["top1"])
    lines = ["| Model | Input | Top-1 | Top-3 | Macro-F1 | Params | Train time (s) | Latency (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['input']} | {r['top1']:.4f} | {r['top3']:.4f} | "
                     f"{r['macro_f1']:.4f} | {r['params']:,} | {r['train_time_s']:.0f} | "
                     f"{r['latency_ms']:.2f} |")
    md = "\n".join(lines)
    open(f"{RES}/results_table.md", "w").write(md + "\n")
    print(md)
    return rows


def plot_comparison(rows):
    base = [r for r in rows if r["model"] in MODEL_INPUTS]
    base = sorted(base, key=lambda r: r["top1"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([r["model"] for r in base], [r["top1"] * 100 for r in base],
            color=[COLORS[r["input"]] for r in base])
    for i, r in enumerate(base):
        ax.text(r["top1"] * 100, i, f" {r['top1'] * 100:.1f}", va="center", fontsize=8)
    ax.set_xlabel("test top-1 accuracy (%)")
    lo = min(r["top1"] for r in base) * 100
    ax.set_xlim(max(0, lo - 10), 100)
    ax.set_title("Model comparison (blue = image, red = strokes, green = both)")
    _save(fig, "model_comparison.png")


# --------------------------------------------------------------------------
# Experiment: early-guess curve
# --------------------------------------------------------------------------
def _batch_predict(model, imgs, seqs, lens, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            o = model(torch.from_numpy(imgs[i:i + bs]).float().to(device),
                      torch.from_numpy(seqs[i:i + bs]).float().to(device),
                      torch.from_numpy(lens[i:i + bs]).long())
            out.append(o.cpu())
    return torch.cat(out).numpy()


def early_guess(runs, data, n_eval=3000, max_strokes=6, seed=42):
    """Accuracy when the model only sees the first k strokes of each drawing."""
    device = get_device()
    rng = np.random.default_rng(seed)
    idx = rng.choice(data["test_idx"], size=min(n_eval, len(data["test_idx"])),
                     replace=False)
    y = data["y"][idx]
    raws = data["raw"][idx]
    ks = list(range(1, max_strokes + 1))
    results = {r: [] for r in runs}
    models = {r: load_model(r, device) for r in runs}

    for k in ks:
        imgs, seqs, lens = [], [], []
        for r in raws:
            strokes = json.loads(r)[:k]
            s, l = strokes_to_sequence(strokes)
            seqs.append(s); lens.append(l)
            imgs.append(strokes_to_image(strokes)[None])
        imgs, seqs, lens = np.stack(imgs), np.stack(seqs), np.array(lens)
        for run in runs:
            model, ckpt = models[run]
            s_in = seqs.copy()
            if ckpt.get("no_pen_lift"):
                s_in[:, :, 2] = 0
            acc = float((_batch_predict(model, imgs, s_in, lens, device).argmax(1) == y).mean())
            results[run].append(acc)
        print(f"k={k}: " + ", ".join(f"{r} {results[r][-1]:.3f}" for r in runs))

    fig, ax = plt.subplots(figsize=(6.8, 4))
    for run in runs:
        ax.plot(ks, [a * 100 for a in results[run]], marker="o", label=run,
                color=COLORS[MODEL_INPUTS[models[run][1]["name"]]], alpha=0.9)
    ax.set_xlabel("number of strokes shown"); ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("Early-guess curve: how soon can each model recognise the drawing?")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "early_guess.png")
    json.dump({"strokes": ks, **results}, open(f"{RES}/early_guess.json", "w"), indent=1)


# --------------------------------------------------------------------------
# Experiment: learning curve (data efficiency)
# --------------------------------------------------------------------------
def learning_curve(models, per_class_sizes=(1000, 2000, 5000, 7000)):
    rows = {r["model"]: r for r in read_metrics()}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in models:
        xs, ys = [], []
        for k in per_class_sizes:
            key = m if k == per_class_sizes[-1] else f"{m}_{k // 1000}k"
            if key in rows:
                xs.append(k); ys.append(rows[key]["top1"] * 100)
        if xs:
            ax.plot(xs, ys, marker="o", label=m,
                    color=COLORS[MODEL_INPUTS[m]])
    ax.set_xlabel("training samples per class"); ax.set_ylabel("test top-1 accuracy (%)")
    ax.set_title("Learning curve"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "learning_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quickdraw_10cls.npz")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--early-guess", action="store_true")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--models", nargs="+", default=["cnn", "bigru", "fusion"])
    ap.add_argument("--n-eval", type=int, default=3000)
    args = ap.parse_args()
    data = load_dataset(args.data)

    if args.all:
        rows = read_metrics()
        plot_balance(data)
        plot_samples(data)
        ranked = results_table(rows)
        plot_comparison(rows)
        for r in rows:
            plot_history(r["model"])
        best = ranked[0]["model"]
        print("best model:", best)
        plot_confusion(best, data)
        error_gallery(best, data)
    if args.early_guess:
        early_guess(args.models, data, args.n_eval)
    if args.learning_curve:
        learning_curve(args.models)


if __name__ == "__main__":
    main()
