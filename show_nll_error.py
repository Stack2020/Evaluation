"""
Evaluate NLL bias vs NFE for PF-ODE (Heun) and plot convergence.

- |ΔNLL_true|: bias to analytic Gaussian log-density.
- Expectation: |ΔNLL_true| decreases with larger NFE and flattens at a plateau.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config as base_config, get_device
from data_loader import generate_test
from grid_error_metrics import gaussian_log_density
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability, set_seed

# Evaluation settings
NFE_LIST: Sequence[int] = (16, 32, 64, 128, 256, 512)
CHECKPOINT_PATH = Path(base_config.get("config_path", "vp_sde_2d_best.pt"))
OUTPUT_NPZ = Path("2d_gaussian.npz")
OUTPUT_PNG = Path("nll_error.png")


def merge_config(checkpoint_cfg: Dict | None) -> Dict:
    merged = base_config.copy()
    if checkpoint_cfg:
        merged.update(checkpoint_cfg)
    merged.setdefault("config_path", str(CHECKPOINT_PATH))
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


def plot_bias(
    nfes: Sequence[int],
    abs_delta_true: Sequence[float],
    kl_values: Sequence[float],
    *,
    output_path: Path,
) -> None:
    """Plot |ΔNLL_true| vs NFE on a log-log axis with KL(p||q_model) on a twin y-axis."""
    nfes = np.asarray(nfes, dtype=np.int64)
    order = np.argsort(nfes)
    nfes = nfes[order]
    abs_delta_true = np.asarray(abs_delta_true, dtype=np.float64)[order]
    kl_values = np.asarray(kl_values, dtype=np.float64)[order]

    eps_floor = 1e-14
    abs_delta_true = np.clip(abs_delta_true, eps_floor, None)
    kl_values = np.clip(np.abs(kl_values), eps_floor, None)

    palette = {"true": "#204b57", "kl": "#e76f51"}

    with plt.rc_context(
        {
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "0.2",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.color": "0.25",
            "ytick.color": "0.25",
        }
    ):
        fig, ax_left = plt.subplots(figsize=(4.8, 3.4))
        ax_right = ax_left.twinx()
        ax_left.set_xscale("log")
        ax_left.set_yscale("log")
        ax_right.set_xscale("log")
        ax_right.set_yscale("log")

        l1 = ax_left.plot(
            nfes,
            abs_delta_true,
            color=palette["true"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=r"|$\Delta\mathrm{NLL}_\text{true}$|",
        )
        l2 = ax_right.plot(
            nfes,
            kl_values,
            color=palette["kl"],
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="--",
            linewidth=2.0,
            markersize=6.0,
            label=r"$\mathrm{KL}(p\|\;p_\theta)$",
        )

        ax_left.set_xlabel("NFE")
        ax_left.set_ylabel("Absolute NLL bias")
        ax_right.set_ylabel("KL divergence")

        ax_left.grid(axis="y", which="major", linestyle="--", alpha=0.25, linewidth=0.8)
        ax_left.grid(axis="x", which="major", linestyle=":", alpha=0.18, linewidth=0.8)
        ax_left.set_xlim(nfes[0] * 0.9, nfes[-1] * 1.1)
        ax_right.set_xlim(nfes[0] * 0.9, nfes[-1] * 1.1)
        ax_left.set_xticks(nfes)
        ax_left.set_xticklabels([str(int(n)) for n in nfes])

        for ax in (ax_left, ax_right):
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
            ax.tick_params(axis="y", colors=("0.25" if ax is ax_left else "0.35"))

        handles = l1 + l2
        labels = [h.get_label() for h in handles]
        ax_left.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.82,
            borderpad=0.35,
            handletextpad=0.45,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def main() -> None:
    seed = int(base_config.get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(base_config, device)
    checkpoint = load_checkpoint(model, str(CHECKPOINT_PATH), device)
    cfg = merge_config(checkpoint.get("config", {}))

    # Prepare test set
    test_np = generate_test(cfg, seed=seed)
    test_tensor = torch.from_numpy(test_np).to(device=device, dtype=torch.float32)

    # Analytic NLL (ground truth)
    true_nll, true_mean = compute_analytic_nll(test_np, cfg)
    print(f"Analytic mean NLL: {true_mean:.6f}")

    nfes = list(NFE_LIST)
    delta_true_list: list[float] = []
    kl_list: list[float] = []
    mean_nll_list: list[float] = []

    logp_true = -true_nll

    for nfe in nfes:
        est_nll, est_mean = estimate_model_nll(model, cfg, device, test_tensor, nfe)
        logp_model = -est_nll
        delta_true = est_mean - true_mean
        delta_true_list.append(delta_true)
        kl_est = float((logp_true - logp_model).mean())
        kl_list.append(kl_est)
        mean_nll_list.append(est_mean)
        print(
            f"NFE={nfe:4d} | mean NLL={est_mean:.6f} "
            f"| ΔNLL_true={delta_true:.6e} | KL={kl_est:.6e}"
        )

    # Save metrics
    np.savez(
        OUTPUT_NPZ,
        nfes=np.asarray(nfes, dtype=np.int64),
        delta_nll_true=np.asarray(delta_true_list, dtype=np.float64),
        abs_delta_nll_true=np.abs(np.asarray(delta_true_list, dtype=np.float64)),
        kl_div=np.asarray(kl_list, dtype=np.float64),
        mean_nll=np.asarray(mean_nll_list, dtype=np.float64),
        true_mean_nll=true_mean,
    )
    print(f"Saved metrics to {OUTPUT_NPZ}")

    plot_bias(nfes, np.abs(delta_true_list), kl_list, output_path=OUTPUT_PNG)


if __name__ == "__main__":
    main()
