#!/usr/bin/env python
"""Generate the paper figure from experiment CSV files."""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def configure_plot_style():
    mpl.style.use("classic")
    mpl.rcParams["figure.figsize"] = [5, 3]
    mpl.rcParams["axes.linewidth"] = 0.75
    mpl.rcParams["figure.facecolor"] = "w"
    mpl.rcParams["grid.linewidth"] = 0.75
    mpl.rcParams["lines.linewidth"] = 0.75
    mpl.rcParams["patch.linewidth"] = 0.75
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 7
    mpl.rcParams["axes.titlesize"] = "medium"
    mpl.rcParams["legend.fontsize"] = "medium"


def make_figure(results_dir: str, output_path: str):
    vision_datasets = ["MNIST", "FashionMNIST", "EMNIST", "CIFAR"]
    methods = [
        "ours, gaussian",
        "ours, mixed-logit",
        "london, gaussian",
        "london, mixed-logit",
        "sakhi1, gaussian",
        "sakhi1, mixed-logit",
        "sakhi2, gaussian",
        "sakhi2, mixed-logit",
        "logging_reward",
    ]
    colors = {
        "ours, gaussian": "red",
        "ours, mixed-logit": "brown",
        "london, gaussian": "cyan",
        "london, mixed-logit": "blue",
        "sakhi1, gaussian": "gray",
        "sakhi1, mixed-logit": "purple",
        "sakhi2, gaussian": "orange",
        "sakhi2, mixed-logit": "green",
        "logging_reward": "k",
    }
    labels = [
        "Ours, Gaussian",
        "Ours, Mixed-Logit",
        "London et al., Gaussian",
        "London et al., Mixed-Logit",
        "Sakhi et al. 1, Gaussian",
        "Sakhi et al. 1, Mixed-Logit",
        "Sakhi et al. 2, Gaussian",
        "Sakhi et al. 2, Mixed-Logit",
        "Logging",
    ]
    line_styles = {key: "-." for key in methods}
    markers = {key: "." for key in methods}
    markers["logging_reward"] = ""

    fig, axs = plt.subplots(1, 4, sharex="col", figsize=(2.26 * 4, 2.2))

    for i, dataset_name in enumerate(vision_datasets):
        ax = axs[i]
        path = os.path.join(results_dir, f"results_{dataset_name}.csv")
        df = pd.read_csv(path)

        for k, method in enumerate(methods):
            ax.plot(
                df["eta"],
                df[method],
                linestyle=line_styles[method],
                marker=markers[method],
                color=colors[method],
                label=labels[k],
            )

        if i < 2:
            ax.set_title(f"{dataset_name}, K=10, d=784")
        elif i == 2:
            ax.set_title(f"{dataset_name}, K=47, d=784")
        else:
            ax.set_title(f"{dataset_name}, K=100, d=2048")

        if i == 0:
            ax.set_ylabel("reward of the learned policy")
        ax.set_xlabel(r"inverse-temperature parameter $\eta_0$")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    fig.legend(labels=labels, loc="lower center", ncol=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=1200, bbox_inches=0)
    print("saved:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="figures/vision_datasets.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    configure_plot_style()
    args = parse_args()
    make_figure(args.results_dir, args.output)
