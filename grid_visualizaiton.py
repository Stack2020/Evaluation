from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config as gmm_config
from config_banana import config as banana_config
from config_involute import config as involute_config
from data_loader_banana import banana_log_density
from gmm_data_loader import GMM_DATA
from model import build_model
from show_density_heatmap import evaluate_model_log_density, gmm_gaussian_log_density


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: dict
    checkpoint_path: Path
    reference_log_density: Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true density, PF-ODE density and absolute error for three test distributions."
    )
    parser.add_argument("--output", type=Path, default=Path("test_density_grid.png"))
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    parser.add_argument("--resolution", type=int, default=200, help="Grid points per axis.")
    parser.add_argument("--steps", type=int, default=None, help="Override PF-ODE integration steps.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--method", choices=("heun", "euler", "rk45"), default=None)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=(-6.0, 6.0, -6.0, 6.0),
    )
    parser.add_argument(
        "--involute-quadrature-points",
        type=int,
        default=4000,
        help="Number of midpoint quadrature points for the noisy involute reference density.",
    )
    parser.add_argument("--gmm-checkpoint", type=Path, default=Path("vp_sde_best_gmm.pt"))
    parser.add_argument("--involute-checkpoint", type=Path, default=Path("vp_sde_best_involute.pt"))
    parser.add_argument("--banana-checkpoint", type=Path, default=Path("vp_sde_best_wrap.pt"))
    return parser.parse_args()


def involute_log_density(
    points: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    noise_std: float,
    quadrature_points: int,
    point_batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate the noisy involute density by midpoint quadrature over its uniform latent radius."""
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive to evaluate the involute density.")
    if quadrature_points < 2:
        raise ValueError("involute quadrature requires at least two points.")

    dr = (r_max - r_min) / quadrature_points
    radius = r_min + (np.arange(quadrature_points, dtype=np.float64) + 0.5) * dr
    centers = np.column_stack((radius * np.sin(2.0 * radius), radius * np.cos(2.0 * radius)))
    log_norm = -math.log(2.0 * math.pi * noise_std * noise_std)
    outputs: list[np.ndarray] = []

    for start in range(0, points.shape[0], point_batch_size):
        point_batch = np.asarray(points[start : start + point_batch_size], dtype=np.float64)
        diff = point_batch[:, None, :] - centers[None, :, :]
        log_components = log_norm - 0.5 * np.sum(diff * diff, axis=-1) / (noise_std * noise_std)
        max_log = np.max(log_components, axis=1)
        log_mean = max_log + np.log(np.mean(np.exp(log_components - max_log[:, None]), axis=1))
        outputs.append(log_mean)

    return np.concatenate(outputs)


def build_specs(args: argparse.Namespace) -> tuple[DatasetSpec, ...]:
    involute_cfg = involute_config.copy()
    r_min = float(involute_cfg.get("r_min", 0.0))
    r_max = float(involute_cfg.get("r_max", 2.0 * math.pi))
    noise_std = float(involute_cfg.get("noise_std", 0.4))
    banana_a = float(banana_config.get("banana_a", 1.0))

    return (
        DatasetSpec(
            name="GMM",
            config=gmm_config,
            checkpoint_path=args.gmm_checkpoint,
            reference_log_density=lambda points: gmm_gaussian_log_density(points, GMM_DATA),
        ),
        DatasetSpec(
            name="Involute",
            config=involute_cfg,
            checkpoint_path=args.involute_checkpoint,
            reference_log_density=lambda points: involute_log_density(
                points,
                r_min=r_min,
                r_max=r_max,
                noise_std=noise_std,
                quadrature_points=args.involute_quadrature_points,
            ),
        ),
        DatasetSpec(
            name="Banana",
            config=banana_config,
            checkpoint_path=args.banana_checkpoint,
            reference_log_density=lambda points: banana_log_density(points, banana_a),
        ),
    )


def build_grid(
    resolution: int, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if resolution < 2:
        raise ValueError("resolution must be at least two.")
    x_min, x_max, y_min, y_max = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
        indexing="xy",
    )
    return grid_x, grid_y, np.column_stack((grid_x.ravel(), grid_y.ravel()))


def evaluate_dataset(
    spec: DatasetSpec,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    device: torch.device,
    steps_override: int | None,
    method_override: str | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not spec.checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {spec.checkpoint_path}")

    checkpoint = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    merged_config = spec.config.copy()
    merged_config.update(checkpoint.get("config", {}))
    merged_config["config_path"] = str(spec.checkpoint_path)

    model = build_model(merged_config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    steps = int(steps_override or merged_config.get("density_eval_steps", 1200))
    method = str(method_override or merged_config.get("density_method", "heun"))
    eps = float(merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))
    print(f"Evaluating {spec.name}: checkpoint={spec.checkpoint_path}, steps={steps}, method={method}")

    true_density = np.exp(spec.reference_log_density(grid_points)).reshape(grid_shape)
    model_log_density = evaluate_model_log_density(
        model,
        merged_config,
        device,
        grid_points,
        steps=steps,
        eps=eps,
        method=method,
        batch_size=batch_size,
    ).reshape(grid_shape)
    model_density = np.exp(model_log_density)
    true_density = np.nan_to_num(true_density, nan=0.0, posinf=0.0, neginf=0.0)
    model_density = np.nan_to_num(model_density, nan=0.0, posinf=0.0, neginf=0.0)
    return true_density, model_density, np.abs(model_density - true_density)


def draw_grid(
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bounds: tuple[float, float, float, float],
    output_path: Path,
    dpi: int,
) -> None:
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 10.2), constrained_layout=True)
    titles = (r"$p_{\mathrm{data}}(x)$", r"$p_{\theta^{*}}(x)$", "Absolute Error")
    density_cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")
    x_min, x_max, y_min, y_max = bounds

    for row_idx, (name, true_density, model_density, abs_error) in enumerate(rows):
        density_max = float(max(np.max(true_density), np.max(model_density)))
        for col_idx, data in enumerate((true_density, model_density, abs_error)):
            ax = axes[row_idx, col_idx]
            is_error = col_idx == 2
            image = ax.pcolormesh(
                grid_x,
                grid_y,
                data,
                shading="auto",
                cmap=error_cmap if is_error else density_cmap,
                vmin=0.0,
                vmax=None if is_error else density_max,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if row_idx == 0:
                ax.set_title(titles[col_idx])
            if col_idx == 0:
                ax.text(
                    -0.30,
                    0.5,
                    f"({chr(ord('a') + row_idx)}) {name}",
                    rotation=90,
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    fontsize=12,
                )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bounds = tuple(float(value) for value in args.bounds)
    grid_x, grid_y, grid_points = build_grid(args.resolution, bounds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    for spec in build_specs(args):
        densities = evaluate_dataset(
            spec,
            grid_points,
            grid_x.shape,
            device=device,
            steps_override=args.steps,
            method_override=args.method,
            batch_size=max(1, args.batch_size),
        )
        rows.append((spec.name, *densities))

    draw_grid(rows, grid_x, grid_y, bounds, args.output, args.dpi)
    print(f"Saved 3x3 density grid to {args.output.resolve()}")


if __name__ == "__main__":
    main()
