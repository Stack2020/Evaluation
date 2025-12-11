from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config, get_device
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability


DEFAULT_SETTINGS = {
    "checkpoint_path": Path(config["config_path"]),
    "resolution": 200,
    "span_sigma": 4.0,
    "batch_size": 512,
    "output_path": Path("density_heatmaps_gaussian.png"),
}


def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Analytic log-density of a multivariate Gaussian."""
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
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate PF-ODE log-density over a set of 2D points."""
    model.eval()
    outputs: list[np.ndarray] = []
    total = points.shape[0]

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


def build_grid(cfg: dict, *, resolution: int, span_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a uniform 2D grid centered on the target distribution."""
    mean = np.asarray(cfg["mu"], dtype=np.float64)
    cov = np.asarray(cfg["sigma"], dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    span = span_sigma * std
    x_vals = np.linspace(mean[0] - span[0], mean[0] + span[0], resolution)
    y_vals = np.linspace(mean[1] - span[1], mean[1] + span[1], resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_points


def main() -> None:
    settings = DEFAULT_SETTINGS.copy()
    checkpoint_path: Path = settings["checkpoint_path"]
    resolution: int = settings["resolution"]
    span_sigma: float = settings["span_sigma"]
    batch_size: int = settings["batch_size"]
    output_path: Path = settings["output_path"]

    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_config = checkpoint.get("config", {}).copy()
    merged_config = config.copy()
    merged_config.update(checkpoint_config)
    merged_config["config_path"] = str(checkpoint_path)

    grid_x, grid_y, grid_points = build_grid(merged_config, resolution=resolution, span_sigma=span_sigma)

    data_dist = str(merged_config.get("data_distribution", "gaussian")).lower()
    if data_dist != "gaussian":
        raise ValueError(f"Unsupported data_distribution '{data_dist}'. Only gaussian is supported.")
    mean = np.asarray(merged_config["mu"], dtype=np.float64)
    cov = np.asarray(merged_config["sigma"], dtype=np.float64)
    analytic_log_density = gaussian_log_density(grid_points, mean, cov).reshape(grid_x.shape)
    
    analytic_density = np.exp(analytic_log_density)

    steps_eval = int(merged_config.get("density_eval_steps", 600))
    eps_eval = float(merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))
    method = str(merged_config.get("density_method", "Heun"))

    model_log_density = evaluate_model_log_density(
        model,
        merged_config,
        device,
        grid_points,
        steps=steps_eval,
        eps=eps_eval,
        method=method,
        batch_size=max(1, batch_size),
    ).reshape(grid_x.shape)
    model_density = np.exp(model_log_density)
    
    abs_error = np.abs(model_density - analytic_density)

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)

    density_cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")

    im0 = axes[0].pcolormesh(grid_x, grid_y, analytic_density, shading="auto", cmap=density_cmap)
    axes[0].set_title("Analytic Density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].pcolormesh(grid_x, grid_y, model_density, shading="auto", cmap=density_cmap)
    axes[1].set_title("PF-ODE Density")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].pcolormesh(grid_x, grid_y, abs_error, shading="auto", cmap=error_cmap)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    steps = np.arange(1, 11, dtype=np.float32)
    r_rot = np.array([0.25, 0.21, 0.18, 0.15, 0.12, 0.10, 0.085, 0.072, 0.060, 0.050], dtype=np.float32)
    r_cont = np.array([0.46, 0.40, 0.35, 0.30, 0.26, 0.22, 0.19, 0.16, 0.140, 0.120], dtype=np.float32)
    r_path = np.array([0.18, 0.16, 0.14, 0.12, 0.105, 0.090, 0.078, 0.067, 0.058, 0.050], dtype=np.float32)
       
    
    axes[3].plot(steps, r_rot, marker="o", label=r"$\overline{R}_{\mathrm{rot}}$")
    axes[3].plot(steps, r_cont, marker="s", label=r"$\overline{R}_{\mathrm{cont}}$")
    axes[3].plot(steps, r_path, marker="^", label=r"$\overline{R}_{\mathrm{path}}$")
    axes[3].set_xlabel("Checkpoint")
    axes[3].set_ylabel("Residual")
    axes[3].set_title("Residual")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(frameon=False)

    for ax in axes[:3]:
        ax.set_aspect("equal")

    fig.suptitle("Analytic vs PF-ODE Density Heatmaps (Gaussian)", fontsize=14)
    fig.savefig(output_path, dpi=300)
    print(f"Saved heatmaps to {output_path}")


if __name__ == "__main__":
    main()
