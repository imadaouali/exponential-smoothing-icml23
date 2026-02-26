#!/usr/bin/env python
"""Run repeated experiment launches and report reproducibility diagnostics."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_once(root: Path, run_idx: int, args) -> Path:
    out_dir = root / args.output_root / f"run_{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(root / "experiments" / "run_vision.py"),
        "--dataset",
        args.dataset,
        "--output_dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--logging_epochs",
        str(args.logging_epochs),
        "--train_epochs",
        str(args.train_epochs),
        "--eta_steps",
        str(args.eta_steps),
        "--replay_count",
        str(args.replay_count),
    ]

    if args.deterministic:
        cmd.append("--deterministic")
    if args.cpu:
        cmd.append("--cpu")

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=root)
    return out_dir / f"results_{args.dataset}.csv"


def compare_frames(paths):
    frames = [pd.read_csv(p) for p in paths]
    base = frames[0].sort_index(axis=1)

    max_abs = 0.0
    for i in range(1, len(frames)):
        current = frames[i].sort_index(axis=1)
        if list(base.columns) != list(current.columns) or len(base) != len(current):
            return False, np.inf
        delta = (current - base).abs().to_numpy()
        max_abs = max(max_abs, float(np.nanmax(delta)))

    return True, max_abs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FashionMNIST", "EMNIST", "CIFAR"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default="results_repro")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--logging_epochs", type=int, default=2)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--eta_steps", type=int, default=3)
    parser.add_argument("--replay_count", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    csv_paths = [run_once(root, i, args) for i in range(args.runs)]

    print("\nper-run file hashes:")
    hashes = [sha256_file(p) for p in csv_paths]
    for p, h in zip(csv_paths, hashes):
        print(f"  {p}: {h}")

    same_shape, max_abs = compare_frames(csv_paths)
    exact_hash_match = len(set(hashes)) == 1
    numeric_match = same_shape and max_abs <= args.atol

    print("\nreproducibility summary:")
    print("  exact_csv_match:", exact_hash_match)
    print("  numeric_match (atol=%g): %s" % (args.atol, numeric_match))
    print("  max_abs_diff:", max_abs)

    if not numeric_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
