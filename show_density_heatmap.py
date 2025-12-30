from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config, get_device
from gmm_data_loader import GMM_DATA
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability

DEFAULT_SETTINGS = {
    "checkpoint_path": Path(config["config_path"]),
    "resolution": 200,
    "span_sigma": 4.0,
    "batch_size": 512,
    "output_path": Path("density_heatmaps_gmm_gaussian.png"),
    "method": "Heun",
    "steps_override": None,
    "eps_override": None,
}

def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Analytic log-density of a multivariate Gaussian."""
    cov_inv = np.linalg.inv(cov)
    diff = points - mean
    exponent = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
    dim = mean.shape[0]
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + np.log(np.linalg.det(cov)))
    return log_norm - 0.5 * exponent


def gmm_gaussian_log_density(
    points: np.ndarray,
    mixture: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Analytic log-density of a Gaussian mixture using GMM_DATA defaults."""
    mixture = GMM_DATA if mixture is None else mixture

    means = np.asarray(mixture["means"], dtype=np.float64)

    covariances = np.asarray(mixture["covariances"], dtype=np.float64)
    weights = mixture.get("weights")
    num_components = means.shape[0]

    if weights is None:
        weights_arr = np.full(num_components, 1.0 / num_components, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()

    log_components = []
    for mean, cov, weight in zip(means, covariances, weights_arr):
        component_log = gaussian_log_density(points, mean, cov) + math.log(weight)
        log_components.append(component_log)

    stacked = np.stack(log_components, axis=-1)
    return np.logaddexp.reduce(stacked, axis=-1)


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


def build_grid(
    cfg: dict,
    *,
    resolution: int,
    span_sigma: float,
    mixture: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a uniform 2D grid covering either a single Gaussian or a mixture."""
    if mixture is not None:
        means = np.asarray(mixture["means"], dtype=np.float64)
        covariances = np.asarray(mixture["covariances"], dtype=np.float64)
        diag_vars = np.diagonal(covariances, axis1=1, axis2=2)
        stds = np.sqrt(np.clip(diag_vars, a_min=0.0, a_max=None))
        span = span_sigma * stds
        min_bounds = np.min(means - span, axis=0)
        max_bounds = np.max(means + span, axis=0)
        x_vals = np.linspace(min_bounds[0], max_bounds[0], resolution)
        y_vals = np.linspace(min_bounds[1], max_bounds[1], resolution)
    else:
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
    method_pref: str | None = settings["method"]
    steps_override: int | None = settings["steps_override"]
    eps_override: float | None = settings["eps_override"]

    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_config = checkpoint.get("config", {}).copy()
    merged_config = config.copy()
    merged_config.update(checkpoint_config)
    merged_config["config_path"] = str(checkpoint_path)

    reference_mixture: dict[str, np.ndarray] | None = GMM_DATA

    grid_x, grid_y, grid_points = build_grid(
        merged_config,
        resolution=resolution,
        span_sigma=span_sigma,
        mixture=reference_mixture,
    )

    if reference_mixture is not None:
        # 针对GMM数据
        analytic_log_density = gmm_gaussian_log_density(grid_points, reference_mixture).reshape(grid_x.shape)
    else:
        mean = np.asarray(merged_config["mu"], dtype=np.float64)
        cov = np.asarray(merged_config["sigma"], dtype=np.float64)
        # 标准各向异性二元高斯
        analytic_log_density = gaussian_log_density(grid_points, mean, cov).reshape(grid_x.shape)

    analytic_density = np.exp(analytic_log_density)

    steps_eval = int(steps_override or merged_config.get("density_eval_steps", 1000))
    eps_eval = float(eps_override or merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))
    method = str(method_pref or merged_config.get("density_method", "Heun"))

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    density_cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")

    im0 = axes[0].pcolormesh(grid_x, grid_y, analytic_density, shading="auto", cmap=density_cmap)
    axes[0].set_title("GMM Density")
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

    for ax in axes:
        ax.set_aspect("equal")

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)

    fig.savefig(output_path, dpi=300)
    print(f"Saved heatmaps to {output_path}")


if __name__ == "__main__":
    main()
