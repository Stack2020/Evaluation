import math
from typing import List, Tuple

import numpy as np
import torch

config = {
    "seed": 52,
    "n_train": 100000,
    "n_val": 10000,
    "n_test": 10000,
    "n_gen": 10000,
    # involute dataset parameters
    "r_min": 0.0,
    "r_max": 2.0 * math.pi,
    "noise_std": 0.4,
    # noise scheduler
    "beta_min": 0.1,
    "beta_max": 20.0,
    "eps": 1e-3,
    "density_eval_points": 2000,
    "density_eval_steps": 1200,
    "nll_time_schedule": "power",
    "nll_time_gamma": 2.5,
    "loss_time_gamma": 2.2,
    "loss_time_reweight": False,
    "loss_time_weight_max": 0.0,
    "hidden": [64, 128, 256, 512],
    "layers": 4,
    "t_embed_dim": 64,
    "batch_size": 128,
    "epochs": 3500,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    "lambda_gamma": 1.0,

    "curl": True,
    "lambda_max": 0.1,
    "t_cutoff": 0.1,
    "k": 120,
    "num_steps": 5000,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
    "density_method": "heun",
    "config_path": "vp_sde_involute_regularization_high.pt",
}


def estimate_involute_moments(
    r_min: float,
    r_max: float,
    noise_std: float,
    *,
    n_grid: int = 4000,
) -> Tuple[List[float], List[List[float]]]:
    """Estimate moments for the involute distribution (uniform r in [r_min, r_max] with Gaussian noise)."""
    r_vals = np.linspace(r_min, r_max, int(max(10, n_grid)), dtype=np.float64)
    base_x = r_vals * np.sin(2.0 * r_vals)
    base_y = r_vals * np.cos(2.0 * r_vals)

    mean_x = float(base_x.mean())
    mean_y = float(base_y.mean())
    cx = base_x - mean_x
    cy = base_y - mean_y
    var_x = float((cx * cx).mean() + noise_std * noise_std)
    var_y = float((cy * cy).mean() + noise_std * noise_std)
    cov_xy = float((cx * cy).mean())
    mu = [mean_x, mean_y]
    sigma = [[var_x, cov_xy], [cov_xy, var_y]]
    return mu, sigma


def ensure_moments(cfg: dict) -> dict:
    """Return a copy of cfg with involute moments populated if absent."""
    if "mu" in cfg and "sigma" in cfg:
        return cfg
    r_min = float(cfg.get("r_min", 0.0))
    r_max = float(cfg.get("r_max", 2.0 * math.pi))
    noise_std = float(cfg.get("noise_std", 0.4))
    mu, sigma = estimate_involute_moments(r_min, r_max, noise_std)
    cfg = dict(cfg)
    cfg["mu"] = mu
    cfg["sigma"] = sigma
    return cfg


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
