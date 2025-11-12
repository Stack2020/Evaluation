from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import torch

from config import config, get_device
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability

# === User-editable settings ===================================================
CHECKPOINT_PATH: Path = Path(config["config_path"])
RESOLUTION: int = 200
SPAN_SIGMA: float = 4.0
BATCH_SIZE: int = 1024
METHOD: str | None = "Heun"
STEPS_OVERRIDE: int | None = None
EPS_OVERRIDE: float | None = None


def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov_inv = np.linalg.inv(cov)
    diff = points - mean
    exponent = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
    dim = mean.shape[0]
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + np.log(np.linalg.det(cov)))
    return log_norm - 0.5 * exponent


def evaluate_model_log_density(
    model,
    cfg: dict,
    device: torch.device,
    points: np.ndarray,
    *,
    steps: int,
    eps: float,
    method: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    total = points.shape[0]
    outputs: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = torch.from_numpy(points[start:end]).to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                xb,
                cfg,
                steps=steps,
                eps=eps,
                method=method,
            )
        outputs.append(logp.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_grid(cfg: dict, *, resolution: int, span_sigma: float) -> np.ndarray:
    mean = np.asarray(cfg["mu"], dtype=np.float64)
    cov = np.asarray(cfg["sigma"], dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    span = span_sigma * std
    x_vals = np.linspace(mean[0] - span[0], mean[0] + span[0], resolution)
    y_vals = np.linspace(mean[1] - span[1], mean[1] + span[1], resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


@dataclass
class Metrics:
    checkpoint: Path
    mse: float
    mae: float
    max_abs: float


def compute_metrics_for_checkpoint(
    checkpoint_path: Path,
    *,
    grid_points: np.ndarray,
    analytic_log_density: np.ndarray,
    device: torch.device,
) -> Metrics:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_cfg = checkpoint.get("config", {}).copy()
    merged_cfg = config.copy()
    merged_cfg.update(checkpoint_cfg)
    merged_cfg["config_path"] = str(checkpoint_path)

    steps_eval = int(STEPS_OVERRIDE or merged_cfg.get("density_eval_steps", 600))
    eps_eval = float(EPS_OVERRIDE or merged_cfg.get("nll_eval_eps", merged_cfg.get("eps", 1e-3)))
    method = str(METHOD or merged_cfg.get("density_method", "Heun"))

    model_log_density = evaluate_model_log_density(
        model,
        merged_cfg,
        device,
        grid_points,
        steps=steps_eval,
        eps=eps_eval,
        method=method,
        batch_size=max(1, BATCH_SIZE),
    )

    diff = model_log_density - analytic_log_density
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))

    return Metrics(checkpoint=checkpoint_path, mse=mse, mae=mae, max_abs=max_abs)


def main() -> None:
    device = get_device()

    grid_points = build_grid(
        config,
        resolution=RESOLUTION,
        span_sigma=SPAN_SIGMA,
    )
    mean = np.asarray(config["mu"], dtype=np.float64)
    cov = np.asarray(config["sigma"], dtype=np.float64)
    analytic_log_density = gaussian_log_density(grid_points, mean, cov)

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"Evaluating checkpoint {CHECKPOINT_PATH} ...")
    result = compute_metrics_for_checkpoint(
        CHECKPOINT_PATH,
        grid_points=grid_points,
        analytic_log_density=analytic_log_density,
        device=device,
    )

    print("\nMetrics:")
    print(
        f" {result.checkpoint}: MSE={result.mse:.4e}, "
        f"MAE={result.mae:.4e}, MaxAbs={result.max_abs:.4e}"
    )

    print("\nTable row (single checkpoint):")
    latex_row = (
        "2D Gaussian (aniso) & "
        f"{result.mse:.4e} & "
        f"{result.mae:.4e} & "
        f"{result.max_abs:.4e} \\\\"
    )
    print(latex_row)


if __name__ == "__main__":
    main()
