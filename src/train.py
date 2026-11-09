"""Train one or all models and log metrics to results/metrics.csv.

Examples
    python -m src.train --model cnn --epochs 25
    python -m src.train --model all --epochs 25                 # full benchmark
    python -m src.train --model bigru --no-pen-lift --tag _nopen    # ablation
    python -m src.train --model cnn --train-per-class 1000 --tag _1k  # learning curve
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data import load_dataset
from src.models import MODEL_INPUTS, build_model, count_params
from src.utils import get_device, make_loaders, predict, set_seed

ALL_MODELS = ["ann", "cnn", "lstm", "bilstm", "gru", "bigru",
              "conv1d_bigru", "fusion"]


def topk_accuracy(logits, y, k=3):
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def summarise(logits, y):
    pred = logits.argmax(1)
    return {"top1": float((pred == y).mean()),
            "top3": topk_accuracy(logits, y, 3),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


def measure_latency(model, loader, n=200):
    """Average single-sample CPU inference time in milliseconds."""
    model = copy.deepcopy(model).cpu().eval()
    img, seq, ln, _ = next(iter(loader))
    n = min(n, len(img))
    with torch.no_grad():
        for i in range(5):                                    # warm-up
            model(img[i:i + 1], seq[i:i + 1], ln[i:i + 1])
        t0 = time.perf_counter()
        for i in range(n):
            model(img[i:i + 1], seq[i:i + 1], ln[i:i + 1])
    return (time.perf_counter() - t0) / n * 1000


def train_one(name, data, args, device):
    set_seed(args.seed)
    n_classes = len(data["classes"])
    train_loader, val_loader, test_loader = make_loaders(
        data, args.batch_size, args.train_per_class, args.no_pen_lift, args.seed)

    model = build_model(name, n_classes).to(device)
    params = count_params(model)
    run = f"{name}{args.tag}"
    print(f"\n=== {run}  ({MODEL_INPUTS[name]} input, {params:,} parameters) ===")

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=2)

    best_val, best_state, wait = 0.0, None, 0
    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, tot, correct = 0.0, 0, 0
        for img, seq, ln, y in train_loader:
            img, seq, y = img.to(device), seq.to(device), y.to(device)
            optim.zero_grad()
            out = model(img, seq, ln)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilises RNNs
            optim.step()
            tot_loss += loss.item() * len(y)
            tot += len(y)
            correct += (out.argmax(1) == y).sum().item()

        v_logits, v_y = predict(model, val_loader, device)
        v_loss = float(criterion(torch.from_numpy(v_logits), torch.from_numpy(v_y)))
        v_acc = float((v_logits.argmax(1) == v_y).mean())
        sched.step(v_loss)
        history.append({"epoch": epoch, "train_loss": tot_loss / tot,
                        "train_acc": correct / tot,
                        "val_loss": v_loss, "val_acc": v_acc})
        print(f"epoch {epoch:02d} | train loss {tot_loss / tot:.4f} acc {correct / tot:.4f}"
              f" | val loss {v_loss:.4f} acc {v_acc:.4f}")

        if v_acc > best_val:                                    # early stopping
            best_val, best_state, wait = v_acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    t_logits, t_y = predict(model, test_loader, device)
    res = summarise(t_logits, t_y)
    res.update({"model": run, "input": MODEL_INPUTS[name], "params": params,
                "epochs_run": len(history), "train_time_s": round(train_time, 1),
                "latency_ms": round(measure_latency(model, test_loader), 3)})
    print(f"TEST  top1 {res['top1']:.4f} | top3 {res['top3']:.4f} | "
          f"macro-F1 {res['macro_f1']:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save({"state": best_state, "name": name, "classes": [str(c) for c in data["classes"]],
                "no_pen_lift": args.no_pen_lift}, f"checkpoints/{run}.pt")
    with open(f"results/history_{run}.json", "w") as f:
        json.dump(history, f)
    np.save(f"results/test_logits_{run}.npy", t_logits)
    append_metrics(res)
    return res


def append_metrics(res, path="results/metrics.csv"):
    cols = ["model", "input", "top1", "top3", "macro_f1", "params",
            "epochs_run", "train_time_s", "latency_ms"]
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["model"] != res["model"]]
    rows.append({c: res[c] for c in cols})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="|".join(ALL_MODELS) + "|all")
    ap.add_argument("--data", default="data/quickdraw_10cls.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="suffix for ablation runs")
    ap.add_argument("--train-per-class", type=int, default=None)
    ap.add_argument("--no-pen-lift", action="store_true")
    args = ap.parse_args()

    data = load_dataset(args.data)
    device = get_device()
    print("device:", device)
    names = ALL_MODELS if args.model == "all" else [args.model]
    for name in names:
        train_one(name, data, args, device)


if __name__ == "__main__":
    main()
