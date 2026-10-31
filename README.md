# LSGNN-Edge

Implementation of **Sparse Spiking Graph Learning for Resource-Constrained Bearing Fault Diagnosis**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Obtain the original CWRU, Paderborn, MFPT, or XJTU-SY recordings from their dataset maintainers. CWRU is available at https://engineering.case.edu/bearingdatacenter; Paderborn at https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter. Keep each dataset separate because sampling rates and fault taxonomies differ. For XJTU-SY, assign health stages from a declared run-to-failure protocol before preparing windows; stage labels are not inferred from filenames.

Create `recordings.csv`:

```csv
path,key,label,split,group
data/normal.mat,X097_DE_time,0,train,record097
data/fault_train.npy,,1,train,bearing_a
data/normal_val.npy,,0,val,bearing_b
data/fault_val.npy,,1,val,bearing_c
data/normal_test.npy,,0,test,bearing_d
data/fault_test.npy,,1,test,bearing_e
```

These filenames illustrate the schema. Supply actual recordings. Each MAT key must contain one finite vector; NPY inputs contain one channel. Labels are consecutive integers starting at zero and retain the same meaning in all partitions. Assign original recordings or physical bearings to splits first; a group crossing splits is rejected. Select working-condition disjoint groups in this manifest for cross-condition experiments.

Preparation creates nonoverlapping windows, divides each into eight equal Hann frames, averages FFT power in 32 equal index bands, and applies `log1p`. Bands cover the recording's full Nyquist range. The training partition supplies band means, standard deviations, and absolute correlations. Radius 2 retains only nearby frequency bands. Use a consistent acquisition rate within one prepared dataset, or resample recordings before preparation.

```bash
lsgnn-edge prepare --manifest recordings.csv --output bearing.npz --window 2048 --steps 8 --bands 32 --radius 2
```

## Training and inference

```bash
lsgnn-edge train --data bearing.npz --output outputs/full --epochs 100 --seed 0
```

## Evaluation

```bash
lsgnn-edge evaluate --data bearing.npz --model outputs/full/model.pt --output outputs/full/evaluation --mac-pj 4.6 --ac-pj 0.9
```

## Ablations

Supported controls: `no_graph`, `dense`, `adjacent`, `rate`, `ann`, `no_penalty`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
lsgnn-edge train --data bearing.npz --output outputs/no_graph --epochs 100 --seed 0 --variant no_graph
```

## Outputs

`bearing.npz` stores prepared windows/graph. Training writes `model.pt` and `history.json`; evaluation writes `metrics.json`, `predictions.npz` and `confusion.png`. Energy figures use the declared MAC/AC pJ model.


## Method, metrics and implementation choices

| Paper item | Code | Entry |
| --- | --- | --- |
| Training-only local correlation, delta code, two LIF graph layers | `prepare`, `BandGraph.forward` | `prepare`, `train` |
| Trainable dense-graph / adjacency-only control | `BandGraph.effective_graph` / `train` | `--variant dense` / `--variant adjacent` |
| Cross-entropy + normalized hidden spike activity | `train` | `--penalty 0.02 --threshold 0.25` |
| Accuracy, macro-F1, parameters, spike rate, latency, MAC/AC energy | `evaluate` | `evaluate` |

The dense control learns symmetric nonnegative edge weights with a fixed self-loop and degree normalization. The adjacent control connects all frequency bands within the prepared radius and removes correlation magnitudes. Older prepared files without a saved radius use radius 2. Rate coding uses Bernoulli input events; evaluation fixes Torch seed 0, so use the same sample order for comparison. Two width-32 layers are the default implementation configuration; parameter counts are computed from the instantiated model and are not copied from the paper. The energy estimate excludes preprocessing, memory traffic and device power. Native-waveform CNN, TCN and GRU controls plus an adapter to the author's LiConvFormer code are supplied below. The existing implementation and this functional update are not the original paper's experiment records.

## Functional validation

```bash
python tests/smoke.py
```

Python 3.13.5, PyTorch 2.7.1+cu118, CPU, one Torch thread, seed 0: synthetic recording CSV loading, grouped window preparation, training-only normalization, all seven variants' finite losses/backward and effective parameter updates (including dense graph logits), save/load agreement within atol 1e-7 / rtol 1e-6, two training epochs, evaluation and confusion plot passed. A four-point macro-F1 example is checked against 0.5. Controlled positive hidden weights are used in the gradient check to exercise firing and readout. These checks do not establish benchmark accuracy or hardware energy savings.

## Native-waveform controls and a measured CWRU subset

`prepare` now retains raw windows alongside graph features. Raw-window mean and scale are fitted only on training recordings. Variants `cnn`, `tcn`, `gru` and `liconvformer` use those identical raw windows; existing graph/event variants retain the original spectral path. CNN/TCN/GRU are explicitly configured matched controls (width 32), not a claim to reproduce an external paper's training recipe.

[LiConvFormer author code](https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework) is fetched at commit `8207cddb4df62921187d608e7209579dbdeecf68` and instantiated as the original `Liconvformer(None,1,classes,dim=32)`. It remains under ignored `external/` and is not redistributed here; upstream does not currently include a license file. The adapter restores the batch dimension dropped by the author's final `squeeze()` for batch size one and leaves the upstream file unchanged.

```bash
python -m lsgnn_edge.public_data --output data/cwru-small --seconds 2 --liconvformer external/Liconvformer.py
python -m lsgnn_edge.cli prepare --manifest data/cwru-small/recordings.csv --output data/cwru-small.npz
python -m lsgnn_edge.cli train --data data/cwru-small.npz --output outputs/cnn --variant cnn --epochs 1 --seed 13
python -m lsgnn_edge.cli train --data data/cwru-small.npz --output outputs/liconvformer --variant liconvformer --external-source external/Liconvformer.py --epochs 1 --seed 13
python -m lsgnn_edge.cli evaluate --data data/cwru-small.npz --model outputs/liconvformer/model.pt --output outputs/liconvformer/evaluation --external-source external/Liconvformer.py
python tests/cwru_small.py
```

The downloader uses the [CWRU university host](https://engineering.case.edu/bearingdatacenter): normal recordings 97/98/99 and 0.007-inch inner-race recordings 105/106/107. Whole recordings at 0/1/2 hp define train/validation/test respectively. Only the first 2 seconds per recording enter this small check. Normal drive-end signals at 48 kHz are anti-alias resampled to 12 kHz to match fault recordings. This is a binary subset, not the full fault taxonomy or the paper's four-dataset evaluation.

Actual CPU run `python tests/cwru_small.py`: 6 measured recordings, 66 nonoverlapping windows, and one epoch for `full/cnn/tcn/gru/liconvformer` completed training, saved checkpoints, test metrics and plots. The separate smoke checks verify nonzero first-convolution/readout gradients, parameter updates, save/load equality and positive operation counts for native controls. Native energy estimates count Conv/Linear/GRU MACs and explicit broadcast-operation estimates; they exclude activation/normalization, preprocessing and memory costs. No board-level energy measurement is claimed.
