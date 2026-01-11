from __future__ import annotations

import argparse
from typing import Iterable, List

import numpy as np
import torch

from config import config as base_config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint, pc_sample, pf_sample
from util import set_seed


def energy_statistic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Empirical Energy Distance D_E^2 between two samples."""
    dist_xy = torch.cdist(x, y).mean()
    dist_xx = torch.cdist(x, x).mean()
    dist_yy = torch.cdist(y, y).mean()
    return 2.0 * dist_xy - dist_xx - dist_yy


def median_bandwidths(
    data: torch.Tensor,
    *,
    max_samples: int = 2000,
    scales: Iterable[float] = (0.5, 1.0, 2.0),
    seed: int | None = None,
) -> List[float]:
    """Median heuristic for RBF bandwidths with optional scaling."""
    total = data.shape[0]
    if total > max_samples:
        rng = torch.Generator(device="cpu")
        if seed is not None:
            rng.manual_seed(int(seed))
        idx = torch.randperm(total, generator=rng)[:max_samples]
        subset = data[idx]
    else:
        subset = data

    dist_sq = torch.cdist(subset, subset) ** 2
    triu = dist_sq.triu(diagonal=1)
    non_zero = triu[triu > 0]
    if non_zero.numel() == 0:
        median_sq = torch.tensor(1.0, device=data.device, dtype=data.dtype)
    else:
        median_sq = torch.median(non_zero)
    median_sq = torch.clamp(median_sq, min=1e-12)
    return [float(median_sq * float(s)) for s in scales]


def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, bandwidths: List[float]) -> torch.Tensor:
    """Gaussian RBF kernel matrix with optional bandwidth mixture."""
    dist_sq = torch.cdist(x, y) ** 2
    kernels = [torch.exp(-dist_sq / (2.0 * bw)) for bw in bandwidths]
    return sum(kernels) / float(len(kernels))


def mmd_statistic(x: torch.Tensor, y: torch.Tensor, bandwidths: List[float]) -> torch.Tensor:
    """Unbiased MMD^2 estimator with Gaussian RBF kernel mixture."""
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Both samples must contain at least 2 points for MMD.")

    k_xx = gaussian_kernel_matrix(x, x, bandwidths)
    k_yy = gaussian_kernel_matrix(y, y, bandwidths)
    k_xy = gaussian_kernel_matrix(x, y, bandwidths)

    # Remove diagonal terms for unbiased estimation
    k_xx_sum = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    k_yy_sum = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    k_xy_mean = k_xy.mean()
    return k_xx_sum + k_yy_sum - 2.0 * k_xy_mean


def permutation_test(
    stat_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_permutations: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute observed statistic and permutation p-value for a two-sample test."""
    obs = float(stat_fn(x, y))

    n = x.shape[0]
    combined = torch.cat([x, y], dim=0)
    total = combined.shape[0]

    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(int(seed))

    hits = 0
    for _ in range(num_permutations):
        perm = torch.randperm(total, generator=rng)
        xa = combined[perm[:n]]
        yb = combined[perm[n:]]
        stat_b = float(stat_fn(xa, yb))
        if stat_b >= obs:
            hits += 1

    p_val = (hits + 1.0) / (num_permutations + 1.0)  # add-one smoothing
    return obs, p_val


def summarize_moments(gen_np: np.ndarray, target_mu: np.ndarray, target_sigma: np.ndarray) -> dict:
    """Compute moment calibration metrics for generated samples."""
    hat_mu = np.mean(gen_np, axis=0)
    hat_sigma = np.cov(gen_np, rowvar=False)

    mu_diff = float(np.linalg.norm(hat_mu - target_mu))
    sigma_diff = float(np.linalg.norm(hat_sigma - target_sigma, ord="fro"))
    if mu_diff <= 0.0 or sigma_diff <= 0.0:
        raise ValueError("Moment distance metrics must be positive; check generated samples.")

    return {
        "mu_diff": mu_diff,
        "sigma_diff": sigma_diff,
    }


def run_tests(
    test_np: np.ndarray,
    gen_np: np.ndarray,
    *,
    num_permutations: int,
    bandwidth_scales: Iterable[float],
    seed: int | None,
) -> dict:
    device = torch.device("cpu")
    dtype = torch.float32
    test_t = torch.as_tensor(test_np, device=device, dtype=dtype)
    gen_t = torch.as_tensor(gen_np, device=device, dtype=dtype)

    energy_stat, energy_p = permutation_test(
        energy_statistic,
        test_t,
        gen_t,
        num_permutations=num_permutations,
        seed=seed,
    )

    bw = median_bandwidths(
        torch.cat([test_t, gen_t], dim=0),
        seed=seed,
        scales=bandwidth_scales,
    )
    mmd_fn = lambda a, b: mmd_statistic(a, b, bw)  
    mmd_stat, mmd_p = permutation_test(
        mmd_fn,
        test_t,
        gen_t,
        num_permutations=num_permutations,
        seed=seed,
    )

    return {
        "energy_stat": energy_stat,
        "energy_p": energy_p,
        "mmd_stat": mmd_stat,
        "mmd_p": mmd_p,
        "bandwidths": bw,
    }

#程序入口
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy test + Kernel MMD two-sample checks.")
    
    parser.add_argument("--method", choices=["pc", "pf"], default="pc", help="Sampling method to generate new data.")
    parser.add_argument("--permutations", type=int, default=200, help="Number of permutations for p-value estimation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config seed).")
    parser.add_argument("--num-gen", type=int, default=None, help="Number of generated samples (default: config n_gen).")
    parser.add_argument(
        "--bandwidth-scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Multipliers applied to median heuristic bandwidth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(base_config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    set_seed(seed)
    
    device = get_device()
    model = build_model(cfg, device)

    checkpoint = load_checkpoint(model, cfg["config_path"], device)
    ckpt_cfg = checkpoint.get("config", cfg)

    n_gen = int(ckpt_cfg.get("n_gen", 5000) if args.num_gen is None else args.num_gen)

    test_np = generate_test(ckpt_cfg, seed=seed)

    with torch.no_grad():
        if args.method == "pf":
            gen = pf_sample(model, ckpt_cfg, device, n_gen).cpu().numpy()
        else:
            gen = pc_sample(model, ckpt_cfg, device, n_gen).cpu().numpy()

    results = run_tests(
        test_np,
        gen,
        num_permutations=args.permutations,
        bandwidth_scales=args.bandwidth_scales,
        seed=seed,
    )

    n_test = test_np.shape[0]
    target_mu = np.asarray(ckpt_cfg.get("mu", cfg["mu"]), dtype=np.float64)
    target_sigma = np.asarray(ckpt_cfg.get("sigma", cfg["sigma"]), dtype=np.float64)
    moment_stats = summarize_moments(gen, target_mu, target_sigma)

    energy_stat = max(results["energy_stat"], 0.0)
    mmd_stat = max(results["mmd_stat"], 0.0)

    header = [
        "mu_diff_l2",
        "sigma_diff_fro",
        "D_E2",
        "K",
        "p_E",
        "p_K",
    ]

    print("|".join(header))
    print(
        " ".join(
            [
                f"{moment_stats['mu_diff']:.16f}",
                f"{moment_stats['sigma_diff']:.16f}",
                f"{energy_stat:.16f}",
                f"{mmd_stat:.16f}",
                f"{results['energy_p']:.16f}",
                f"{results['mmd_p']:.16f}",
            ]
        )
    )
   
if __name__ == "__main__":
    main()
