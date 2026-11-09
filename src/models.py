"""All model architectures for the Quick, Draw! benchmark.

Every model has the same forward signature  model(img, seq, lengths)  so the
training and evaluation code is identical for all of them. Image-only models
ignore (seq, lengths); sequence-only models ignore img.

  img     : (B, 1, 28, 28) float
  seq     : (B, T, 3)      float   (dx, dy, pen_lift)
  lengths : (B,)           long    true number of points (before padding)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# --------------------------------------------------------------------------
# Track A - image models
# --------------------------------------------------------------------------
class ANN(nn.Module):
    """Fully connected baseline on the flattened 28x28 image."""
    def __init__(self, n_classes, img_size=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, img, seq=None, lengths=None):
        return self.net(img)


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    """3 conv blocks (BatchNorm + MaxPool) followed by a small classifier."""
    def __init__(self, n_classes, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128))       # 28 -> 3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, feat_dim), nn.ReLU())
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(feat_dim, n_classes)

    def encode(self, img):
        return self.proj(self.pool(self.features(img)))

    def forward(self, img, seq=None, lengths=None):
        return self.fc(self.drop(self.encode(img)))


# --------------------------------------------------------------------------
# Track B - stroke-sequence models
# --------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    """LSTM / GRU (optionally bidirectional) over the stroke sequence.

    Padded steps are skipped with pack_padded_sequence, so the final hidden
    state always corresponds to the last REAL point of each drawing.
    """
    def __init__(self, n_classes, cell="gru", bidirectional=False,
                 in_dim=3, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.cell = cell
        self.bidirectional = bidirectional
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes)
        self.out_dim = out_dim

    def encode(self, seq, lengths):
        lengths = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(seq, lengths, batch_first=True,
                                      enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.cell == "lstm":
            h = h[0]                                   # (layers*dirs, B, H)
        if self.bidirectional:
            return torch.cat([h[-2], h[-1]], dim=1)    # last layer fwd + bwd
        return h[-1]

    def forward(self, img, seq, lengths):
        return self.fc(self.drop(self.encode(seq, lengths)))


class Conv1DBiGRU(nn.Module):
    """Conv1D picks up local stroke shapes, BiGRU models their order."""
    def __init__(self, n_classes, in_dim=3, hidden=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.gru = nn.GRU(64, hidden, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)
        self.out_dim = hidden * 2

    def encode(self, seq, lengths):
        x = self.conv(seq.transpose(1, 2)).transpose(1, 2)   # (B, T, 64)
        packed = pack_padded_sequence(x, lengths.clamp(min=1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return torch.cat([h[-2], h[-1]], dim=1)

    def forward(self, img, seq, lengths):
        return self.fc(self.drop(self.encode(seq, lengths)))


# --------------------------------------------------------------------------
# Track C - fusion
# --------------------------------------------------------------------------
class FusionNet(nn.Module):
    """CNN branch (final picture) + BiGRU branch (drawing process)."""
    def __init__(self, n_classes, dropout=0.4):
        super().__init__()
        self.cnn = CNN(n_classes, feat_dim=128)
        self.rnn = RNNClassifier(n_classes, cell="gru", bidirectional=True,
                                 hidden=128, layers=1)
        self.head = nn.Sequential(
            nn.Linear(128 + self.rnn.out_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, img, seq, lengths):
        z = torch.cat([self.cnn.encode(img), self.rnn.encode(seq, lengths)], dim=1)
        return self.head(z)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
MODEL_INPUTS = {          # which representation each model uses
    "ann": "image", "cnn": "image",
    "lstm": "sequence", "bilstm": "sequence", "gru": "sequence",
    "bigru": "sequence", "conv1d_bigru": "sequence",
    "fusion": "both",
}


def build_model(name, n_classes):
    name = name.lower()
    if name == "ann":
        return ANN(n_classes)
    if name == "cnn":
        return CNN(n_classes)
    if name == "lstm":
        return RNNClassifier(n_classes, "lstm", False)
    if name == "bilstm":
        return RNNClassifier(n_classes, "lstm", True)
    if name == "gru":
        return RNNClassifier(n_classes, "gru", False)
    if name == "bigru":
        return RNNClassifier(n_classes, "gru", True)
    if name == "conv1d_bigru":
        return Conv1DBiGRU(n_classes)
    if name == "fusion":
        return FusionNet(n_classes)
    raise ValueError(f"Unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
