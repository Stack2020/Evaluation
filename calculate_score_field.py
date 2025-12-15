from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from config import config as base_config, get_device
from model import build_model
from sample import load_checkpoint
from sde import v_to_score

__all__ = [
    "merge_config",
    "build_grid",
    "build_time_dependent_grid",
    "validate_times",
    "compute_fields",
    "calculate_score_field",
]


def merge_config(checkpoint: dict, checkpoint_path: Path) -> dict:
    merged = base_config.copy()
    ckpt_cfg = checkpoint.get("config", {})
    merged.update(ckpt_cfg)
    merged["config_path"] = str(checkpoint_path)
    return merged


def build_grid(cfg: dict, resolution: int, span_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(cfg["mu"], dtype=np.float64)
    cov = np.asarray(cfg["sigma"], dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    span = span_sigma * std
    x_vals = np.linspace(mean[0] - span[0], mean[0] + span[0], resolution)
    y_vals = np.linspace(mean[1] - span[1], mean[1] + span[1], resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_points


def validate_times(times: Iterable[float], eps: float) -> list[float]:
    result: list[float] = []
    for t in times:
        t_val = float(t)
        if not (eps <= t_val <= 1.0):
            raise ValueError(f"Time {t_val:.4f} outside valid range [{eps}, 1.0].")
        result.append(t_val)
    return result


def build_time_dependent_grid(
    cfg: dict,
    resolution: int,
    span_sigma: float,
    t_val: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a grid aligned with the forward VP dynamics at time t.

    For VP-SDE: x_t = sqrt(alpha_bar(t)) x0 + sqrt(1-alpha_bar(t)) * z.
    The marginal at time t is Gaussian with mean sqrt(alpha_bar) * mu and
    covariance alpha_bar * Sigma + (1-alpha_bar) * I. We match the grid to
    that marginal so spatial probes track the score's definition w.r.t. x_t.
    """
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])
    lambda_t = beta_min * t_val + 0.5 * (beta_max - beta_min) * t_val * t_val
    alpha_bar_t = math.exp(-lambda_t)
    mean0 = np.asarray(cfg["mu"], dtype=np.float64)
    cov0 = np.asarray(cfg["sigma"], dtype=np.float64)
    eye = np.eye(mean0.shape[0], dtype=np.float64)

    mean_t = math.sqrt(alpha_bar_t) * mean0
    cov_t = alpha_bar_t * cov0 + (1.0 - alpha_bar_t) * eye
    std_t = np.sqrt(np.diag(cov_t))
    span = span_sigma * std_t

    x_vals = np.linspace(mean_t[0] - span[0], mean_t[0] + span[0], resolution)
    y_vals = np.linspace(mean_t[1] - span[1], mean_t[1] + span[1], resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_points


def compute_fields(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    grid_builder,
    times: list[float],
    batch_size: int,
    *,
    time_varying_grid: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute score and rotational residual, supporting time-varying grids."""
    scores: np.ndarray | None = None
    residuals: np.ndarray | None = None
    grid_x_all: list[np.ndarray] = []
    grid_y_all: list[np.ndarray] = []

    model.eval()
    eps = float(cfg.get("eps", 1e-3))

    for time_idx, t_val in enumerate(times):
        grid_x, grid_y, grid_points = grid_builder(t_val)
        num_points = grid_points.shape[0]
        grid_shape = grid_x.shape

        if scores is None or residuals is None:
            scores = np.empty((len(times), *grid_shape, 2), dtype=np.float32)
            residuals = np.empty((len(times), *grid_shape), dtype=np.float32)
        elif grid_shape != scores.shape[1:3]:
            raise ValueError(
                f"Grid shape {grid_shape} at t={t_val:.4f} differs from prior steps {scores.shape[1:3]}."
            )

        score_chunks: list[np.ndarray] = []
        residual_chunks: list[np.ndarray] = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            x_batch = torch.from_numpy(grid_points[start:end]).to(device=device, dtype=torch.float32)
            x_batch = x_batch.requires_grad_(True)
            t_batch = torch.full((x_batch.shape[0],), float(t_val), device=device, dtype=x_batch.dtype)
            t_batch = torch.clamp(t_batch, min=eps, max=1.0)

            with torch.enable_grad():
                score = v_to_score(model(x_batch, t_batch), x_batch, t_batch, cfg)

                jac_rows = []
                for comp in range(score.shape[1]):
                    grad = torch.autograd.grad(
                        outputs=score[:, comp].sum(),
                        inputs=x_batch,
                        retain_graph=comp + 1 < score.shape[1],
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    jac_rows.append(grad.unsqueeze(1))
                jacobian = torch.cat(jac_rows, dim=1)

            diff = jacobian - jacobian.transpose(1, 2)
            fro_sq = (diff * diff).sum(dim=(1, 2))
            rot_residual = torch.sqrt(torch.clamp(fro_sq, min=0.0)) / math.sqrt(2.0)

            score_chunks.append(score.detach().cpu().numpy())
            residual_chunks.append(rot_residual.detach().cpu().numpy())

        score_array = np.concatenate(score_chunks, axis=0).reshape(*grid_shape, 2)
        residual_array = np.concatenate(residual_chunks, axis=0).reshape(*grid_shape)
        scores[time_idx] = score_array.astype(np.float32, copy=False)
        residuals[time_idx] = residual_array.astype(np.float32, copy=False)
        print(f"Computed fields for t={t_val:.4f}")

        if time_varying_grid:
            grid_x_all.append(grid_x.astype(np.float32, copy=False))
            grid_y_all.append(grid_y.astype(np.float32, copy=False))

    if scores is None or residuals is None:
        raise RuntimeError("No scores/residuals computed; ensure times is non-empty.")

    if time_varying_grid:
        grid_x_out = np.stack(grid_x_all, axis=0)
        grid_y_out = np.stack(grid_y_all, axis=0)
    else:
        grid_x_out = grid_x.astype(np.float32, copy=False)
        grid_y_out = grid_y.astype(np.float32, copy=False)

    return scores, residuals, grid_x_out, grid_y_out


def calculate_score_field(
    *,
    checkpoint_path: Path,
    resolution: int,
    span_sigma: float,
    times: Sequence[float],
    batch_size: int,
    output_path: Path | None = None,
    time_varying_grid: bool = True,
) -> dict[str, np.ndarray]:
    device = get_device()
    model = build_model(base_config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    merged_config = merge_config(checkpoint, checkpoint_path)
    eps = float(merged_config.get("eps", 1e-3))
    validated_times = validate_times(times, eps)

    if time_varying_grid:
        grid_builder = lambda t: build_time_dependent_grid(merged_config, resolution, span_sigma, float(t))
    else:
        grid_x, grid_y, grid_points = build_grid(merged_config, resolution, span_sigma)
        grid_builder = lambda _t: (grid_x, grid_y, grid_points)

    scores, residuals, grid_x_out, grid_y_out = compute_fields(
        model,
        merged_config,
        device,
        grid_builder,
        validated_times,
        max(1, int(batch_size)),
        time_varying_grid=time_varying_grid,
    )

    result = {
        "grid_x": grid_x_out,
        "grid_y": grid_y_out,
        "times": np.asarray(validated_times, dtype=np.float32),
        "score": scores,
        "rot_residual": residuals,
        "mu": np.asarray(merged_config["mu"], dtype=np.float32),
        "sigma": np.asarray(merged_config["sigma"], dtype=np.float32),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

    return result
