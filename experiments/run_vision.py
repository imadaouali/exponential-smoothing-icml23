#!/usr/bin/env python
"""Run Exponential Smoothing OPL experiments on vision datasets."""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.datasets import CIFAR100, EMNIST, FashionMNIST, MNIST

# Package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exponential_smoothing_opl.models import (
    BernsteinGaussian,
    BernsteinMixedLogit,
    CatoniGaussian,
    CatoniMixedLogit,
    LondonGaussian,
    LondonMixedLogit,
    OurGaussian,
    OurMixedLogit,
)
from exponential_smoothing_opl.policies import SupervisedPolicy
from exponential_smoothing_opl.utils import build_bandit_dataset, test_risk_exact_probit


DATASETS = {
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
    "EMNIST": EMNIST,
    "CIFAR": CIFAR100,
}


def make_trainer(max_epochs: int, device: torch.device, deterministic: bool = False) -> Trainer:
    use_gpu = device.type == "cuda"
    try:
        return Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=deterministic,
        )
    except TypeError:
        return Trainer(
            max_epochs=max_epochs,
            gpus=1 if use_gpu else 0,
            checkpoint_callback=False,
            weights_summary=None,
            logger=None,
            deterministic=deterministic,
        )


def load_features(name: str, root: str, cifar_feature_batch_size: int = 128):
    dataset = DATASETS[name]

    if name != "CIFAR":
        if name == "EMNIST":
            train_ds = dataset(root, train=True, split="balanced", download=True, transform=transforms.ToTensor())
            test_ds = dataset(root, train=False, split="balanced", download=True, transform=transforms.ToTensor())
        else:
            train_ds = dataset(root, train=True, download=True, transform=transforms.ToTensor())
            test_ds = dataset(root, train=False, download=True, transform=transforms.ToTensor())

        x_train = torch.tensor(train_ds.data).float().reshape(len(train_ds.data), -1)
        y_train = torch.tensor(train_ds.targets)
        x_test = torch.tensor(test_ds.data).float().reshape(len(test_ds.data), -1)
        y_test = torch.tensor(test_ds.targets)
        return x_train, y_train, x_test, y_test

    train_ds = dataset(root, train=True, download=True, transform=transforms.ToTensor())
    test_ds = dataset(root, train=False, download=True, transform=transforms.ToTensor())

    feature_device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            if free_mem >= 2 * 1024**3:
                feature_device = torch.device("cuda:0")
            else:
                print("Low free GPU memory detected; extracting CIFAR features on CPU.")
        except Exception:
            feature_device = torch.device("cuda:0")
    try:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(feature_device).eval()

    def extract_features(images_np: np.ndarray, batch_size: int = 128) -> torch.Tensor:
        feats = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=feature_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=feature_device).view(1, 3, 1, 1)
        with torch.no_grad():
            for start in range(0, len(images_np), batch_size):
                end = min(start + batch_size, len(images_np))
                batch = torch.from_numpy(images_np[start:end]).permute(0, 3, 1, 2).float().to(feature_device)
                batch = batch / 255.0
                batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
                batch = (batch - mean) / std
                batch_feats = feature_extractor(batch).flatten(1).cpu()
                feats.append(batch_feats)
        return torch.cat(feats, dim=0)

    x_train = extract_features(train_ds.data, batch_size=cifar_feature_batch_size)
    x_test = extract_features(test_ds.data, batch_size=cifar_feature_batch_size)
    del feature_extractor, resnet
    if feature_device.type == "cuda":
        torch.cuda.empty_cache()

    x_train = x_train.float()
    y_train = torch.tensor(train_ds.targets)
    x_test = x_test.float()
    y_test = torch.tensor(test_ds.targets)
    return x_train, y_train, x_test, y_test


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    return x / norm


def fit_and_eval(
    model,
    train_loader,
    x_test,
    y_test,
    device,
    epochs,
    deterministic,
    eval_samples,
    eval_batch_size,
):
    trainer = make_trainer(max_epochs=epochs, device=device, deterministic=deterministic)
    trainer.fit(model, train_loader)
    model = model.to(device)
    with torch.no_grad():
        risk = test_risk_exact_probit(
            x_test,
            y_test,
            model,
            n_samples=eval_samples,
            bsize=eval_batch_size,
        )
    return -risk


def run_experiment(args):
    pl.seed_everything(args.seed, workers=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("device:", device)

    x_train, y_train, x_test, y_test = load_features(
        args.dataset,
        args.data_dir,
        cifar_feature_batch_size=args.cifar_feature_batch_size,
    )
    x_train = normalize_rows(x_train)
    x_test = normalize_rows(x_test)

    x0, x_log, y0, y_log = train_test_split(
        x_train, y_train, train_size=args.logging_ratio, random_state=args.seed
    )

    n = len(x_log)
    context_dim = x_log.shape[1]
    num_actions = len(np.unique(y_log))
    print(f"dataset={args.dataset}, n={n}, context_dim={context_dim}, num_actions={num_actions}")

    subsample_loader = DataLoader(TensorDataset(x0, y0), batch_size=args.batch_size, shuffle=True)
    logging_loader = DataLoader(TensorDataset(x_log, y_log), batch_size=args.batch_size, shuffle=True)

    logging_policy = SupervisedPolicy(
        n_actions=num_actions,
        context_dim=context_dim,
        softmax=True,
        reg=args.reg,
        device=device,
    )
    make_trainer(max_epochs=args.logging_epochs, device=device, deterministic=args.deterministic).fit(
        logging_policy, subsample_loader
    )

    etas = np.round(np.linspace(args.eta_min, args.eta_max, args.eta_steps), 2)
    results = defaultdict(list)

    model_builders = [
        ("ours, gaussian", lambda mu0: OurGaussian(n_actions=num_actions, context_dim=context_dim, beta=1 - 1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("ours, mixed-logit", lambda mu0: OurMixedLogit(n_actions=num_actions, context_dim=context_dim, beta=1 - 1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("london, gaussian", lambda mu0: LondonGaussian(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("london, mixed-logit", lambda mu0: LondonMixedLogit(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("sakhi1, gaussian", lambda mu0: CatoniGaussian(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("sakhi1, mixed-logit", lambda mu0: CatoniMixedLogit(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, device=device)),
        ("sakhi2, gaussian", lambda mu0: BernsteinGaussian(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, num_p=100, rc=1, xi=0, device=device)),
        ("sakhi2, mixed-logit", lambda mu0: BernsteinMixedLogit(n_actions=num_actions, context_dim=context_dim, tau=1 / (n ** 0.25), N=n, loc_weight=mu0, num_p=100, rc=1, xi=0, device=device)),
    ]

    for eta in etas:
        logging_policy = logging_policy.to(device)
        logging_policy.alpha = eta

        logging_reward = -test_risk_exact_probit(
            x_test,
            y_test,
            logging_policy,
            n_samples=args.eval_samples,
            bsize=args.eval_batch_size,
        )
        print(f"eta={eta:.2f}, logging_reward={logging_reward:.6f}")

        results["eta"].append(eta)
        results["logging_reward"].append(logging_reward)

        contexts, actions, propensities, costs = build_bandit_dataset(
            logging_loader, logging_policy, replay_count=args.replay_count
        )
        bandit_loader = DataLoader(
            TensorDataset(contexts, actions, propensities, costs),
            batch_size=args.batch_size,
            shuffle=True,
        )

        mu0 = eta * logging_policy.linear.weight.data

        for name, builder in model_builders:
            reward = fit_and_eval(
                builder(mu0),
                bandit_loader,
                x_test,
                y_test,
                device,
                args.train_epochs,
                args.deterministic,
                args.eval_samples,
                args.eval_batch_size,
            )
            print(f"  {name}: {reward:.6f}")
            results[name].append(reward)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_{args.dataset}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print("saved:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="MNIST")
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--logging_ratio", type=float, default=0.05)
    parser.add_argument("--replay_count", type=int, default=1)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--logging_epochs", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--eta_max", type=float, default=1.0)
    parser.add_argument("--eta_steps", type=int, default=10)
    parser.add_argument("--cifar_feature_batch_size", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
