#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

for dataset in MNIST FashionMNIST EMNIST CIFAR; do
  echo "Running ${dataset}..."
  python experiments/run_vision.py --dataset "$dataset" "$@"
done

python experiments/make_vision_figure.py

echo "All experiments complete."
