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
    "output_path": Path("density_eight_gaussian_heatmaps.png"),
}

def eight_gaussian_log_density(points: np.ndarray, *, radius: float, cov: np.ndarray) -> np.ndarray:
    """Analytic log-density of the 8-Gaussian mixture on a circle of given radius."""
    points = np.asarray(points, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov_inv = np.linalg.inv(cov)
    dim = 2
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + np.log(np.linalg.det(cov)))

    w = np.arange(1, 9, dtype=np.float64)
    angles = np.pi * w / 4.0
    centers = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)  # (8, 2)

    diff = points[:, None, :] - centers[None, :, :]  # (N, 8, 2)
    exponent = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)  # (N, 8)
    logpdf = log_norm - 0.5 * exponent  # (N, 8)

    # Mixture of 8 equally weighted components: log(1/8) + logsumexp(logpdf)
    
    log_weight = -math.log(centers.shape[0])
    max_logpdf = np.max(logpdf, axis=1, keepdims=True)
    logsum = max_logpdf[:, 0] + np.log(np.sum(np.exp(logpdf - max_logpdf), axis=1))
    return log_weight + logsum


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
       
    radius = float(cfg.get("eight_gaussian_radius", 1.0))
    cov = np.asarray(cfg.get("eight_gaussian_sigma", [[0.02, 0.0], [0.0, 0.02]]), dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    span_x = radius + span_sigma * std[0]
    span_y = radius + span_sigma * std[1]
    x_vals = np.linspace(-span_x, span_x, resolution)
    y_vals = np.linspace(-span_y, span_y, resolution)
    
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

    radius = float(merged_config.get("eight_gaussian_radius", 1.0))
    cov = np.asarray(merged_config.get("eight_gaussian_sigma", [[0.02, 0.0], [0.0, 0.02]]), dtype=np.float64)
    analytic_log_density = eight_gaussian_log_density(grid_points, radius=radius, cov=cov).reshape(grid_x.shape)
    dist_label = "8-Gaussian"
  
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
    heatmap_path = output_path
    # Analytic, model, and absolute error heatmaps side by side
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)

    im0 = axes[0].pcolormesh(grid_x, grid_y, analytic_density, shading="auto", cmap="viridis")
    axes[0].set_title(f"Analytic Density ({dist_label})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].pcolormesh(grid_x, grid_y, model_density, shading="auto", cmap="viridis")
    axes[1].set_title("PF-ODE Density")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].pcolormesh(grid_x, grid_y, abs_error, shading="auto", cmap="magma")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    fig.suptitle(f"Density Comparison ({dist_label})", fontsize=12)
    fig.savefig(heatmap_path, dpi=300)

    print(f"Saved density comparison figure to {heatmap_path}")

if __name__ == "__main__":
    main()
