from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config as base_config, get_device
from data_loader import generate_test
from model import build_model
from util import gaussian_log_density, integrate_log_probability, set_seed


DEFAULT_NFE: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024)


def merge_config(base_cfg: Dict, checkpoint_cfg: Dict | None, checkpoint_path: Path) -> Dict:
    merged = base_cfg.copy()
    if checkpoint_cfg:
        merged.update(checkpoint_cfg)
    merged["config_path"] = str(checkpoint_path)
    # Align data_dim with checkpoint or mu length to avoid shape mismatches.
    data_dim = None
    if checkpoint_cfg and "data_dim" in checkpoint_cfg:
        data_dim = int(checkpoint_cfg["data_dim"])
    elif "mu" in merged:
        data_dim = len(merged["mu"])
    elif "sigma" in merged:
        data_dim = len(merged["sigma"])
    if data_dim is not None:
        merged["data_dim"] = int(data_dim)
    return merged


def compute_analytic_nll(test_np: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, float]:
    mu = np.asarray(cfg["mu"], dtype=np.float64)
    sigma = np.asarray(cfg["sigma"], dtype=np.float64)
    logp_true = gaussian_log_density(test_np, mean=mu, cov=sigma)
    nll_true = -logp_true
    return nll_true, float(nll_true.mean())


def estimate_model_nll(
    model,
    cfg: Dict,
    device: torch.device,
    test_tensor: torch.Tensor,
    steps: int,
) -> Tuple[np.ndarray, float]:
    """Estimate NLL with PF-ODE (Heun) at a given NFE."""
    eps_eval = float(cfg.get("nll_eval_eps", cfg.get("eps", 1e-3)))
    params = tuple(model.parameters())
    flags = tuple(p.requires_grad for p in params)
    was_training = model.training

    try:
        for p in params:
            p.requires_grad_(False)
        model.eval()
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                test_tensor,
                cfg,
                steps=int(steps),
                eps=eps_eval,
                method="Heun",
            )
        nll = (-logp).detach().cpu().numpy()
        return nll, float(nll.mean())
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)
        if was_training:
            model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFE vs NLL error panel for 5D Gaussian.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(base_config.get("config_path", "vp_sde_5d_gaussian.pt")),
        help="Checkpoint path (default: config_path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nfe_error_5d.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=list(DEFAULT_NFE),
        help="List of NFE values to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config).")
    return parser.parse_args()


def plot_single(
    nfes: np.ndarray,
    abs_delta: np.ndarray,
    abs_logp_err: np.ndarray,
    *,
    output_path: Path,
) -> None:
    """Plot |ΔNLL_true| and E|log pθ - log p| vs NFE with dual y-axes."""
    with plt.rc_context(
        {
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "axes.titlesize": 16.0,
            "legend.fontsize": 12.0,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "0.2",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.color": "0.0",
            "ytick.color": "0.0",
        }
    ):
        # Slightly squatter canvas for easier paper placement.
        fig, ax_left = plt.subplots(figsize=(4.6, 4.1))
        ax_right = ax_left.twinx()

        palette = {
            "bias": "#204b57",  # deep teal
            "abs_err": "#d000b7",  # magenta
        }

        l1 = ax_left.plot(
            nfes,
            abs_delta,
            color=palette["bias"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="-",
            linewidth=2.4,
            markersize=7.0,
            label=r"|$\Delta\mathrm{NLL}_\text{true}$|",
        )
        l2 = ax_right.plot(
            nfes,
            abs_logp_err,
            color=palette["abs_err"],
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="--",
            linewidth=2.2,
            markersize=7.0,
            label=r"$\mathbb{E}[|\log p_\theta - \log p|]$",
        )

        ax_left.set_xscale("log")
        ax_right.set_xscale("log")
        ax_left.set_xlabel("NFE")
        ax_left.set_ylabel(r"|$\Delta\mathrm{NLL}_\text{true}$|", color=palette["bias"])
        ax_right.set_ylabel(r"$\mathbb{E}[|\log p_\theta - \log p|]$", color=palette["abs_err"])
        ax_left.tick_params(axis="y", colors="black")
        ax_right.tick_params(axis="y", colors="black")

        ax_left.set_title("5D Gaussian", pad=10)

        # Styled grid for a clean, publication-like look.
        ax_left.grid(axis="y", which="major", linestyle="--", alpha=0.25, linewidth=0.9)
        ax_left.grid(axis="x", which="major", linestyle=":", alpha=0.2, linewidth=0.9)

        ax_left.set_xticks(nfes)
        ax_left.set_xticklabels([str(int(n)) for n in nfes])

        for spine in ax_left.spines.values():
            spine.set_linewidth(1.0)
        for spine in ax_right.spines.values():
            spine.set_linewidth(1.0)

        handles = l1 + l2
        labels = [h.get_label() for h in handles]
        ax_left.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            borderpad=0.5,
            handletextpad=0.5,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    seed = int(base_config.get("seed", 42) if args.seed is None else args.seed)
    set_seed(seed)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = merge_config(base_config, checkpoint.get("config", {}), args.checkpoint)

    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model"])

    test_np = generate_test(cfg, seed=seed)
    test_tensor = torch.from_numpy(test_np).to(device=device, dtype=torch.float32)

    true_nll, true_mean = compute_analytic_nll(test_np, cfg)
    logp_true = -true_nll

    nfes = np.asarray(sorted(set(int(n) for n in args.nfe)), dtype=np.int64)
    abs_delta_list: list[float] = []
    abs_logp_err_list: list[float] = []

    for nfe in nfes:
        est_nll, est_mean = estimate_model_nll(model, cfg, device, test_tensor, nfe)
        logp_model = -est_nll
        delta_true = est_mean - true_mean
        abs_delta_list.append(abs(delta_true))
        abs_logp_err_list.append(float(np.abs(logp_model - logp_true).mean()))
        print(
            f"NFE={nfe:4d} | mean NLL={est_mean:.6f} "
            f"| Delta_true={delta_true:.6e} | Abs logp err={abs_logp_err_list[-1]:.6e}"
        )

    abs_delta_arr = np.asarray(abs_delta_list, dtype=np.float64)
    abs_logp_err_arr = np.asarray(abs_logp_err_list, dtype=np.float64)

    plot_single(nfes, abs_delta_arr, abs_logp_err_arr, output_path=args.output)


if __name__ == "__main__":
    main()
