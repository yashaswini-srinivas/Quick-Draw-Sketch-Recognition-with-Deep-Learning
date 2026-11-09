"""Shared helpers: seeding, datasets/loaders and batched prediction."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QDDataset(Dataset):
    """Holds image, stroke sequence, true length and label for a set of indices."""

    def __init__(self, data, idx, no_pen_lift=False):
        self.img = torch.from_numpy(data["X_img"][idx]).float()
        self.seq = torch.from_numpy(data["X_seq"][idx]).float()
        self.len = torch.from_numpy(data["X_len"][idx]).long()
        self.y = torch.from_numpy(data["y"][idx]).long()
        if no_pen_lift:                        # ablation: hide stroke boundaries
            self.seq[:, :, 2] = 0.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.img[i], self.seq[i], self.len[i], self.y[i]


def subsample_per_class(idx, y, k, seed=42):
    """Keep k training samples per class (used for the learning-curve study)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        keep.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(keep))


def make_loaders(data, batch_size=256, train_per_class=None,
                 no_pen_lift=False, seed=42, num_workers=0):
    y = data["y"]
    tr_idx = data["train_idx"]
    if train_per_class is not None:
        tr_idx = subsample_per_class(tr_idx, y, train_per_class, seed)
    train = QDDataset(data, tr_idx, no_pen_lift)
    val = QDDataset(data, data["val_idx"], no_pen_lift)
    test = QDDataset(data, data["test_idx"], no_pen_lift)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size, shuffle=True, generator=g,
                       num_workers=num_workers),
            DataLoader(val, batch_size, num_workers=num_workers),
            DataLoader(test, batch_size, num_workers=num_workers))


@torch.no_grad()
def predict(model, loader, device):
    """Return (logits, labels) as numpy arrays."""
    model.eval()
    outs, ys = [], []
    for img, seq, ln, y in loader:
        out = model(img.to(device), seq.to(device), ln)
        outs.append(out.cpu())
        ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()
