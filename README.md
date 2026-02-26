# Exponential Smoothing for Off-Policy Learning (ICML 2023)

Code for the paper:
[Exponential Smoothing for Off-Policy Learning](https://proceedings.mlr.press/v202/aouali23a/aouali23a.pdf)
(Oral, ICML 2023).

Authors: Imad Aouali, Victor-Emmanuel Brunel, David Rohde, Anna Korba.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

## Repository Structure

- `exponential_smoothing_opl/`
  - `policies.py` — policy classes (logging, Gaussian, mixed-logit, softmax)
  - `models.py` — OPL objective/bound implementations (Ours, London, Sakhi)
  - `utils.py` — utilities for dataset generation and risk evaluation
- `experiments/`
  - `run_vision.py` — run experiments on one dataset and write `results/results_<dataset>.csv`
  - `make_vision_figure.py` — generate the main paper figure from CSVs
  - `check_reproducibility.py` — rerun experiments and compare generated CSV files
- `results/` — saved experiment outputs
- `figures/` — generated figures
- `run_all_experiments.sh` — run all datasets + figure generation

The repository is package/script-first. Notebook workflows have been replaced by CLI entrypoints.

## Usage

Run one dataset:

```bash
python experiments/run_vision.py --dataset MNIST --deterministic
python experiments/run_vision.py --dataset FashionMNIST --deterministic
python experiments/run_vision.py --dataset EMNIST --deterministic
python experiments/run_vision.py --dataset CIFAR --deterministic
```

Important flags:

```bash
python experiments/run_vision.py \
  --dataset MNIST \
  --logging_epochs 10 \
  --train_epochs 20 \
  --eta_steps 10 \
  --eval_samples 2048 \
  --eval_batch_size 64 \
  --batch_size 128 \
  --output_dir results \
  --deterministic
```

Generate figure from CSV results:

```bash
python experiments/make_vision_figure.py --results_dir results --output figures/vision_datasets.pdf
```

Run reproducibility validation (same seed, repeated runs):

```bash
python experiments/check_reproducibility.py \
  --dataset MNIST \
  --runs 2 \
  --seed 0 \
  --deterministic \
  --cpu
```

Run everything:

```bash
./run_all_experiments.sh
```

## Notes

- CIFAR uses ImageNet-pretrained ResNet50 feature extraction through `torchvision`.
- Import code from `exponential_smoothing_opl.*` modules.

## Acknowledgement

The baseline code was provided by Otmane Sakhi, Pierre Alquier, Nicolas Chopin:
[PAC-Bayesian Offline Contextual Bandits With Guarantees](https://proceedings.mlr.press/v202/sakhi23a/sakhi23a.pdf)
