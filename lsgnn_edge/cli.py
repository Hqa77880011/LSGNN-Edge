import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from torch import nn

from .waveform import WaveformClassifier, VARIANTS as WAVEFORM_VARIANTS, dense_macs


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return gradient / (1 + 5 * value.abs()).square()


class BandGraph(nn.Module):
    def __init__(self, graph, classes, variant='full', width=32, threshold=.25):
        super().__init__()
        self.register_buffer('graph', torch.as_tensor(graph, dtype=torch.float32))
        self.layers = nn.ModuleList([nn.Linear(1, width, bias=False), nn.Linear(width, width, bias=False)])
        self.output = nn.Linear(width, classes)
        self.variant = variant
        self.threshold = threshold
        self.graph_logits = nn.Parameter(torch.zeros_like(self.graph)) if variant == 'dense' else None

    def effective_graph(self):
        if self.graph_logits is None:
            return self.graph
        weight = torch.sigmoid((self.graph_logits + self.graph_logits.T) / 2)
        eye = torch.eye(len(weight), device=weight.device)
        weight = weight * (1 - eye) + eye
        degree = weight.sum(1).clamp_min(1e-8).rsqrt()
        return weight * degree[:, None] * degree[None, :]

    def forward(self, x):
        graph = self.effective_graph()
        if self.variant == 'rate':
            events = (torch.rand_like(x) < x.sigmoid()).float()
        elif self.variant == 'ann':
            events = x
        else:
            previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
            events = ((x - previous).abs() >= self.threshold).float()
        x = events.unsqueeze(-1)
        activities = []
        ac = torch.zeros(len(x), device=x.device)
        degree = (graph != 0).sum(0).float()
        for layer in self.layers:
            ac += x.sum((1, 3)).mul(degree).sum(1) * layer.out_features
            u = torch.zeros(x.shape[0], x.shape[2], layer.out_features, device=x.device)
            last = torch.zeros_like(u)
            output = []
            for tick in range(x.shape[1]):
                drive = torch.einsum('ij,bjd->bid', graph, layer(x[:, tick]))
                if self.variant == 'ann':
                    last = drive.relu()
                else:
                    u = .9 * u * (1 - last) + drive
                    last = Spike.apply(u - 1.)
                output.append(last)
            x = torch.stack(output, 1)
            activities.append(x)
        activity = sum(v.sum() for v in activities) / sum(v.numel() for v in activities)
        return self.output(x.mean((1, 2))), activity, ac


def prepare(args):
    if args.steps < 1 or args.bands < 1 or args.window // args.steps // 2 + 1 < args.bands:
        raise ValueError('Each frame must contain at least one FFT bin per band')
    records = list(csv.DictReader(open(args.manifest, encoding='utf-8')))
    groups = {}
    features, waveforms, labels, splits, identities = [], [], [], [], []
    for row in records:
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if row['group'] in groups and groups[row['group']] != split:
            raise ValueError('A bearing/recording group crosses splits')
        groups[row['group']] = split
        source = Path(args.manifest).parent / row['path']
        raw = loadmat(source)[row['key']] if source.suffix.lower() == '.mat' else np.load(source)
        raw = np.asarray(raw).squeeze()
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError('Each source must contain a finite one-channel vector')
        for start in range(0, len(raw) - args.window + 1, args.window):
            window = raw[start:start + args.window]
            waveforms.append(window)
            frame_length = args.window // args.steps
            frames = window[:frame_length * args.steps].reshape(args.steps, frame_length)
            power = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=-1)) ** 2
            bands = np.array_split(np.arange(power.shape[-1]), args.bands)
            features.append(np.stack([np.log1p(power[:, b].mean(-1)) for b in bands], -1))
            labels.append(int(row['label']))
            splits.append(split)
            identities.append(row['group'])
    x = np.asarray(features, dtype=np.float32)
    split = np.asarray(splits)
    if set(split) != {'train', 'val', 'test'}:
        raise ValueError('All three partitions must have windows')
    mean = x[split == 'train'].mean((0, 1), keepdims=True)
    scale = x[split == 'train'].std((0, 1), keepdims=True).clip(1e-6)
    z = (x - mean) / scale
    correlation = np.nan_to_num(np.corrcoef(z[split == 'train'].mean(1).T))
    local = np.abs(np.arange(args.bands)[:, None] - np.arange(args.bands)) <= args.radius
    adjacency = np.abs(correlation) * local
    np.fill_diagonal(adjacency, 0)
    waveform = np.asarray(waveforms, dtype=np.float32)
    waveform_mean = float(waveform[split == "train"].mean())
    waveform_scale = float(waveform[split == "train"].std().clip(1e-6))
    np.savez_compressed(args.output, waveform=(waveform-waveform_mean)/waveform_scale, waveform_mean=waveform_mean, waveform_scale=waveform_scale, x=z, y=labels, split=split, group=identities, adjacency=adjacency,
                        mean=mean, scale=scale, window=args.window, steps=args.steps, radius=args.radius)


def normalized(adjacency):
    adjacency = adjacency + np.eye(len(adjacency))
    degree = adjacency.sum(1).clip(1e-8) ** -.5
    return adjacency * degree[:, None] * degree[None, :]


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    d = np.load(args.data)
    input_key = 'waveform' if args.variant in WAVEFORM_VARIANTS else 'x'
    if input_key not in d:
        raise ValueError('Re-run prepare to include raw waveforms for this baseline')
    x, y = torch.tensor(d[input_key]), torch.tensor(d['y'], dtype=torch.long)
    adjacency = d['adjacency']
    if args.variant == 'no_graph':
        adjacency = np.zeros_like(adjacency)
    if args.variant == 'adjacent':
        radius = int(d['radius']) if 'radius' in d else 2
        index = np.arange(len(adjacency))
        adjacency = (abs(index[:, None] - index[None, :]) <= radius).astype(float)
        np.fill_diagonal(adjacency, 0)
    if args.variant == 'dense':
        adjacency = np.ones_like(adjacency) - np.eye(len(adjacency))
    model = (WaveformClassifier(int(y.max())+1, args.variant, args.width, getattr(args, 'external_source', None))
             if args.variant in WAVEFORM_VARIANTS else
             BandGraph(normalized(adjacency), int(y.max())+1, args.variant, args.width, args.threshold))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ids = np.flatnonzero(d['split'] == 'train')
    val_ids = np.flatnonzero(d['split'] == 'val')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in torch.tensor(train_ids)[torch.randperm(len(train_ids))].split(args.batch):
            logits, activity, _ = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch])
            if args.variant not in ('no_penalty', 'ann') and args.variant not in WAVEFORM_VARIANTS:
                loss = loss + args.penalty * activity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = sum(nn.functional.cross_entropy(model(x[b])[0], y[b], reduction='sum').item()
                      for b in torch.tensor(val_ids).split(args.batch)) / len(val_ids)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_loss': val})
        if val < best:
            best = val
            torch.save({'state': model.state_dict(), 'variant': args.variant, 'width': args.width,
                        'threshold': args.threshold, 'classes': int(y.max()) + 1,
                        'graph': normalized(adjacency), 'external_source': getattr(args, 'external_source', None)}, output / 'model.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2))


def evaluate(args):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    d = np.load(args.data)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    native = checkpoint['variant'] in WAVEFORM_VARIANTS
    model = (WaveformClassifier(checkpoint['classes'], checkpoint['variant'], checkpoint['width'], getattr(args,'external_source',None) or checkpoint.get('external_source')) if native else
             BandGraph(checkpoint['graph'],checkpoint['classes'],checkpoint['variant'],checkpoint['width'],checkpoint['threshold']))
    input_key = 'waveform' if native else 'x'
    model.load_state_dict(checkpoint['state'])
    model.eval()
    ids = np.flatnonzero(d['split'] == 'test')
    truth, prediction, rates, accumulations, latencies = [], [], [], [], []
    with torch.no_grad():
        for i in ids:
            start = time.perf_counter()
            logits, rate, ac = model(torch.tensor(d[input_key][i:i + 1]))
            latencies.append((time.perf_counter() - start) * 1000)
            truth.append(int(d['y'][i]))
            prediction.append(int(logits.argmax(1)))
            rates.append(float(rate))
            accumulations.append(float(ac[0]))
    bands, ticks = d['x'].shape[2], d['x'].shape[1]
    width = checkpoint['width']
    mac = dense_macs(model, torch.tensor(d[input_key][ids[:1]])) if native else width * checkpoint['classes']
    if checkpoint['variant'] == 'ann':
        mac += ticks * bands * (width + width * width)
        mac += ticks * np.count_nonzero(checkpoint['graph']) * width * 2
        accumulations = [0.] * len(ids)
    result = {'accuracy': accuracy_score(truth, prediction), 'macro_f1': f1_score(truth, prediction, average='macro'),
              'parameters': sum(p.numel() for p in model.parameters()),
              'spike_rate': float(np.mean(rates)) if checkpoint['variant'] != 'ann' and not native else None,
              'latency_median_ms': float(np.median(latencies)), 'latency_p95_ms': float(np.quantile(latencies, .95)),
              'mean_spike_accumulations': float(np.mean(accumulations)), 'dense_macs': mac,
              'estimated_energy_mj': float((mac * args.mac_pj + np.mean(accumulations) * args.ac_pj) * 1e-9)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / 'predictions.npz', truth=truth, prediction=prediction, group=d['group'][ids])
    ConfusionMatrixDisplay.from_predictions(truth, prediction)
    plt.tight_layout()
    plt.savefig(output / 'confusion.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--bands', type=int, default=32)
    p.add_argument('--radius', type=int, default=2)
    p = commands.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['full', 'no_graph', 'dense', 'adjacent', 'rate', 'ann', 'no_penalty', *WAVEFORM_VARIANTS], default='full')
    p.add_argument('--external-source')
    for name, default in [('seed', 0), ('epochs', 100), ('batch', 64), ('width', 32)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', .001), ('threshold', .25), ('penalty', .02)]:
        p.add_argument('--' + name, type=float, default=default)
    p = commands.add_parser('evaluate')
    for name in ['data', 'model', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--external-source')
    p.add_argument('--mac-pj', type=float, default=4.6)
    p.add_argument('--ac-pj', type=float, default=.9)
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
