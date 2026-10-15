# LSGNN-Edge

Implementation of **Sparse Spiking Graph Learning for Resource-Constrained Bearing Fault Diagnosis**.

LSGNN-Edge converts eight short-time frequency-band energies into delta events, propagates them through two leaky integrate-and-fire graph layers, and classifies their mean spike activity. Graph correlations and feature normalization are fitted only on training recordings. Training combines cross-entropy and normalized hidden spike activity.

## Install

Python 3.11 or newer, with CPU or CUDA PyTorch:

```bash
python -m venv .venv
python -m pip install -e .
```

Activate the environment before using the commands. Select a CUDA-enabled PyTorch build separately when required.

## Prepare recordings

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

```bash
lsgnn-edge prepare --manifest recordings.csv --output bearing.npz --window 2048 --steps 8 --bands 32 --radius 2
```

Preparation creates nonoverlapping windows, divides each into eight equal Hann frames, averages FFT power in 32 equal index bands, and applies `log1p`. Bands cover the recording's full Nyquist range. The training partition supplies band means, standard deviations, and absolute correlations. Radius 2 retains only nearby frequency bands. Use a consistent acquisition rate within one prepared dataset, or resample recordings before preparation.

## Train and compare

```bash
lsgnn-edge train --data bearing.npz --output outputs/full --epochs 100 --seed 0
lsgnn-edge train --data bearing.npz --output outputs/ann --variant ann
lsgnn-edge train --data bearing.npz --output outputs/no_graph --variant no_graph
lsgnn-edge train --data bearing.npz --output outputs/dense --variant dense
lsgnn-edge train --data bearing.npz --output outputs/rate --variant rate
lsgnn-edge train --data bearing.npz --output outputs/no_penalty --variant no_penalty
```

`ann` is a real-valued two-layer graph baseline on the same band features; `no_graph` removes neighbor messages, `dense` uses uniform full connectivity, `rate` replaces delta events with Bernoulli sigmoid-rate events, and `no_penalty` removes hidden activity regularization. The full model uses leak 0.9, firing threshold 1, delta threshold 0.25, width 32, and activity weight 0.02. `--lr`, `--batch`, `--width`, `--threshold`, and `--penalty` control optimization and the quality/activity trade-off. A fast-sigmoid surrogate supplies spike gradients. Each run saves the lowest validation cross-entropy checkpoint and an epoch history; test labels never select a checkpoint.

## Evaluate

```bash
lsgnn-edge evaluate --data bearing.npz --model outputs/full/model.pt --output outputs/full/evaluation --mac-pj 4.6 --ac-pj 0.9
```

`metrics.json` reports accuracy, Macro-F1, parameter count, hidden spike activity, CPU median/p95 forward latency, active accumulation counts, and estimated operation energy. `predictions.npz` retains per-window labels and group identities; `confusion.png` visualizes errors. Repeat for every model and each dataset rather than averaging incompatible label spaces. Repeat training seeds and aggregate per-dataset metrics before reporting a four-dataset mean.

Energy is `MAC × E_MAC + AC × E_AC`; constants are configurable technology assumptions in picojoules. The count includes spike-triggered transformations/messages and dense readout, while ANN counts dense transformations/messages. It excludes FFT, memory traffic, membrane bookkeeping, and hardware static power. The tensor implementation executes dense kernels; activity estimates do not imply measured device energy or event-driven acceleration. Latency measures CPU model forward calls after prepared feature extraction, including the first evaluated call. Rate-coding evaluation is stochastic; set a process seed when a fixed realization is needed.

The manuscript does not specify all signal-transform and optimizer settings. Equal FFT bands, nonoverlapping windows, hidden width, surrogate shape, and Adam at 0.001 are explicit implementation choices. No experimental scores are bundled.
