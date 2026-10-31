"""Small synthetic recording-to-metrics check, not benchmark reproduction."""
import csv
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lsgnn_edge.cli import BandGraph, evaluate, normalized, prepare, train

torch.set_num_threads(1)
torch.manual_seed(0)
work = Path(tempfile.mkdtemp(prefix="lsgnn-smoke-"))
rows = []
for split in ["train", "val", "test"]:
    for label in range(2):
        name = f"{split}{label}.npy"
        t = np.arange(1024) / 128
        np.save(work / name, np.sin(2*np.pi*(10+label*18)*t) + .1*np.random.default_rng(label).normal(size=len(t)))
        rows.append(dict(path=name, key="", label=label, split=split, group=f"{split}{label}"))
with (work / "recordings.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
prepare(Namespace(manifest=str(work/"recordings.csv"), output=str(work/"data.npz"), window=256, steps=4, bands=8, radius=2))
data = np.load(work / "data.npz")
assert np.allclose(data["x"][data["split"] == "train"].mean((0, 1)), 0, atol=1e-5)
for variant in ["full", "no_graph", "dense", "adjacent", "rate", "ann", "no_penalty"]:
    model = BandGraph(normalized(data["adjacency"]), 2, variant, width=8)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(.6)
    x = torch.tensor(data["x"][:4])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    logits, activity, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 1, 1])) + .02 * activity
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.layers[0].weight.grad.abs().sum() > 0 and model.layers[1].weight.grad.abs().sum() > 0
    if variant == "dense":
        assert model.graph_logits.grad.abs().sum() > 0
    torch.optim.Adam(model.parameters(), lr=.01).step()
    assert all((p != before[name]).any() for name,p in model.named_parameters()), variant
    model.eval(); torch.manual_seed(123); expected = model(x)[0].detach()
    torch.save(model.state_dict(), work / "roundtrip.pt")
    restored = BandGraph(normalized(data["adjacency"]), 2, variant, width=8)
    restored.load_state_dict(torch.load(work / "roundtrip.pt", weights_only=True)); restored.eval()
    torch.manual_seed(123); torch.testing.assert_close(restored(x)[0], expected, atol=1e-7, rtol=1e-6)
train(Namespace(data=str(work/"data.npz"), output=str(work/"fit"), variant="full", seed=0,
                width=8, threshold=.25, lr=.01, epochs=2, batch=4, penalty=.02))
evaluate(Namespace(data=str(work/"data.npz"), model=str(work/"fit/model.pt"), output=str(work/"evaluation"), mac_pj=4.6, ac_pj=.9))
# Each class has precision=recall=1/2 in this four-point example.
assert f1_score([0,0,1,1],[0,1,0,1],average="macro") == .5
metrics=json.loads((work/"evaluation/metrics.json").read_text())
assert np.isfinite(metrics["estimated_energy_mj"]) and metrics["estimated_energy_mj"] >= 0
print(json.dumps({"check":"passed", "variants":7,"epochs":2,"outputs":str(work)}))

from lsgnn_edge.waveform import WaveformClassifier, dense_macs
native_variants=["cnn","tcn","gru"]
external=Path("external/Liconvformer.py")
if external.exists():native_variants.append("liconvformer")
for variant in native_variants:
    torch.manual_seed(13)
    network=WaveformClassifier(2,variant,width=32,external_source=str(external))
    sample=torch.randn(4,256);labels=torch.tensor([0,1,0,1])
    first=next(module for module in network.modules() if isinstance(module,torch.nn.Conv1d))
    last=[module for module in network.modules() if isinstance(module,torch.nn.Linear)][-1]
    before=first.weight.detach().clone()
    logits,_,_=network(sample)
    loss=torch.nn.functional.cross_entropy(logits,labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert first.weight.grad.abs().sum()>0 and last.weight.grad.abs().sum()>0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    torch.optim.Adam(network.parameters(),lr=.001).step()
    assert not torch.equal(first.weight,before)
    network.eval();expected=network(sample)[0].detach()
    torch.save(network.state_dict(),work/f"native_{variant}.pt")
    restored=WaveformClassifier(2,variant,width=32,external_source=str(external))
    restored.load_state_dict(torch.load(work/f"native_{variant}.pt",weights_only=True));restored.eval()
    torch.testing.assert_close(restored(sample)[0],expected,atol=1e-7,rtol=1e-6)
    assert dense_macs(network,sample[:1])>0
print('PASS: raw-waveform gradient/update, save/load and operation-count checks',native_variants)
