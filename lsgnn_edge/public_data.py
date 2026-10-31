"""Download a small CWRU normal/inner-race subset from its university host."""

import argparse
import csv
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

# Recording-disjoint load conditions; normal baseline DE is 48 kHz, fault DE 12 kHz.
RECORDS = [
    (97, 0, "train", 48000),
    (105, 1, "train", 12000),
    (98, 0, "val", 48000),
    (106, 1, "val", 12000),
    (99, 0, "test", 48000),
    (107, 1, "test", 12000),
]


def download(output, seconds=2.0):
    """Create measured-signal NPYs and the prepare manifest without redistributing data."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, label, split, rate in RECORDS:
        path = output / f"{number}.mat"
        if not path.exists():
            url = f"https://engineering.case.edu/sites/default/files/{number}.mat"
            with urllib.request.urlopen(url, timeout=90) as response:
                path.write_bytes(response.read())
        raw = loadmat(path)
        key = f"X{number:03d}_DE_time"
        signal = np.asarray(raw[key]).reshape(-1)[: int(seconds * rate)]
        if len(signal) < int(seconds * rate) or not np.isfinite(signal).all():
            raise ValueError(f"Incomplete/nonfinite recording {number}")
        if rate == 48000:
            signal = resample_poly(signal, 1, 4)
        np.save(output / f"record{number}.npy", signal.astype(np.float32))
        rows.append(
            dict(
                path=f"record{number}.npy", key="", label=label, split=split, group=f"CWRU_{number}"
            )
        )
    with (output / "recordings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output / "recordings.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--liconvformer")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    print(download(args.output, args.seconds))
    if args.liconvformer:
        from .waveform import fetch_liconvformer

        fetch_liconvformer(args.liconvformer)


if __name__ == "__main__":
    main()
