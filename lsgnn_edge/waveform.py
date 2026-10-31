"""Matched raw-waveform controls and an external official LiConvFormer adapter."""

import importlib.util
from pathlib import Path
import urllib.request

import torch
from torch import nn

VARIANTS = ("cnn", "tcn", "gru", "liconvformer")
LICONV_COMMIT = "8207cddb4df62921187d608e7209579dbdeecf68"
LICONV_URL = f"https://raw.githubusercontent.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework/{LICONV_COMMIT}/models/Liconvformer.py"


def fetch_liconvformer(path):
    """Fetch the author's pinned file for local use; do not redistribute it."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite external source: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LICONV_URL, timeout=60) as response:
        source = response.read().decode("utf-8")
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8", newline="\n")


class WaveformClassifier(nn.Module):
    """Compact CNN/TCN/GRU controls sharing the exact grouped raw windows."""

    def __init__(self, classes, variant, width=32, external_source=None):
        super().__init__()
        self.variant = variant
        if variant == "liconvformer":
            if not external_source or not Path(external_source).is_file():
                raise ValueError(
                    "Pass --external-source for the downloaded official Liconvformer.py"
                )
            spec = importlib.util.spec_from_file_location("external_liconvformer", external_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.core = module.Liconvformer(None, 1, classes, dim=width)
        elif variant == "gru":
            self.front = nn.Sequential(nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.ReLU())
            self.recurrent = nn.GRU(16, width, batch_first=True)
            self.output = nn.Linear(width, classes)
        else:
            dilations = [1, 2, 4] if variant == "tcn" else [1, 1, 1]
            layers = []
            for index, dilation in enumerate(dilations):
                layers += [
                    nn.Conv1d(
                        1 if index == 0 else width,
                        width,
                        7,
                        stride=2 if index == 0 else 1,
                        dilation=dilation,
                        padding=3 * dilation,
                    ),
                    nn.ReLU(),
                ]
                if index < 2:
                    layers.append(nn.AvgPool1d(2))
            self.core = nn.Sequential(
                *layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, classes)
            )

    def forward(self, x):
        x = x[:, None] if x.ndim == 2 else x
        if self.variant == "gru":
            _, hidden = self.recurrent(self.front(x).transpose(1, 2))
            logits = self.output(hidden[-1])
        else:
            logits = self.core(x)
            if logits.ndim == 1:
                logits = logits[None]  # The author's squeeze() drops batch size one.
        return logits, x.new_zeros(()), x.new_zeros(len(x))


def dense_macs(model, example):
    """Count Conv/Linear/GRU MACs; extra broadcast operations are explicit estimates."""
    total = 0

    def count(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv1d):
            total += output.numel() * module.in_channels * module.kernel_size[0] // module.groups
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
        elif isinstance(module, nn.GRU):
            batch, ticks, _ = inputs[0].shape
            total += (
                batch * ticks * 3 * (module.input_size * module.hidden_size + module.hidden_size**2)
            )
        elif type(module).__name__ == "BroadcastAttention":
            total += 2 * output.numel()
        elif type(module).__name__ == "Add":
            total += 2 * output.numel()

    handles = [module.register_forward_hook(count) for module in model.modules()]
    with torch.no_grad():
        model(example)
    for handle in handles:
        handle.remove()
    return int(total)
