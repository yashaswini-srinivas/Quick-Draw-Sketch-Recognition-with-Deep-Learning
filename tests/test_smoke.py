"""Smoke tests: every model runs, and the train/evaluate pipeline works end to end
on a tiny synthetic dataset (no download needed).

    python -m tests.test_smoke        # or:  pytest -q
"""
import argparse
import os
import tempfile

import numpy as np
import torch

from src.data import load_dataset, make_synthetic
from src.models import MODEL_INPUTS, build_model
from src.utils import QDDataset


def test_all_models_forward_backward():
    data = make_synthetic(n_per_class=20, n_classes=4)
    ds = QDDataset(data, data["train_idx"][:16])
    img, seq, ln, y = ds.img, ds.seq, ds.len, ds.y
    for name in MODEL_INPUTS:
        model = build_model(name, 4)
        out = model(img, seq, ln)
        assert out.shape == (16, 4), name
        torch.nn.functional.cross_entropy(out, y).backward()


def test_pipeline_end_to_end():
    from src import evaluate, train
    data = make_synthetic(n_per_class=60, n_classes=4)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            np.savez_compressed("syn.npz", **data)
            data = load_dataset("syn.npz")
            args = argparse.Namespace(epochs=2, batch_size=32, lr=1e-3, patience=3,
                                      seed=0, tag="", train_per_class=None,
                                      no_pen_lift=False)
            for name in ["cnn", "bigru", "fusion"]:
                train.train_one(name, data, args, torch.device("cpu"))
            rows = evaluate.read_metrics()
            assert len(rows) == 3
            evaluate.results_table(rows)
            evaluate.plot_comparison(rows)
            evaluate.plot_balance(data)
            evaluate.plot_samples(data, per_class=3)
            evaluate.plot_history("cnn")
            evaluate.plot_confusion("cnn", data)
            evaluate.error_gallery("cnn", data)
            evaluate.early_guess(["cnn", "bigru", "fusion"], data,
                                 n_eval=60, max_strokes=3)
            assert os.path.exists("results/early_guess.png")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    test_all_models_forward_backward()
    print("forward/backward OK for all models")
    test_pipeline_end_to_end()
    print("pipeline OK")
