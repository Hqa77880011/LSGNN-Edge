"""Measured CWRU subset check; not a full benchmark or a paper result."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lsgnn_edge.cli import prepare, train, evaluate
from lsgnn_edge.public_data import download
from lsgnn_edge.waveform import fetch_liconvformer

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/cwru-small")
parser.add_argument("--external-source", default="external/Liconvformer.py")
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()
torch.set_num_threads(1)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
manifest = download(output / "data", seconds=2.0)
source = Path(args.external_source)
if not source.exists():
    fetch_liconvformer(source)
data = output / "bearing.npz"
prepare(
    Namespace(manifest=str(manifest), output=str(data), window=2048, steps=8, bands=32, radius=2)
)
bundle = np.load(data)
assert (
    len(
        set(bundle["group"][bundle["split"] == "train"])
        & set(bundle["group"][bundle["split"] == "test"])
    )
    == 0
)
results = []
for variant in ["full", "cnn", "tcn", "gru", "liconvformer"]:
    destination = output / variant
    train(
        Namespace(
            data=str(data),
            output=str(destination),
            variant=variant,
            seed=13,
            width=32,
            threshold=0.25,
            lr=0.001,
            epochs=args.epochs,
            batch=8,
            penalty=0.02,
            external_source=str(source),
        )
    )
    evaluate(
        Namespace(
            data=str(data),
            model=str(destination / "model.pt"),
            output=str(destination / "evaluation"),
            mac_pj=4.6,
            ac_pj=0.9,
            external_source=str(source),
        )
    )
    metric = json.loads((destination / "evaluation/metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metric["accuracy"]) and np.isfinite(metric["macro_f1"])
    assert metric["parameters"] > 0 and metric["estimated_energy_mj"] >= 0
    results.append(
        dict(
            variant=variant,
            accuracy=metric["accuracy"],
            macro_f1=metric["macro_f1"],
            parameters=metric["parameters"],
        )
    )
print(
    json.dumps(
        dict(
            check="passed",
            measured_recordings=6,
            windows=len(bundle["y"]),
            epochs=args.epochs,
            data="CWRU normal and 0.007-inch inner-race, 0/1/2 hp recording-disjoint",
            results=results,
            functional_subset_only=True,
        )
    )
)
