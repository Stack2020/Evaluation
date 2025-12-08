from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from config import config as base_config, get_device
from model import build_model
from sample import load_checkpoint
from sde import drift_pf, divergence_exact
from util import integrate_log_probability

__all__ = [
    "calculate_continuity_residual",
    "merge_config",
    "build_grid",
    "validate_times",
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


def validate_times(times: Sequence[float], eps: float) -> list[float]:
    if not times:
        raise ValueError("At least two time snapshots are required.")
    ordered = sorted(float(t) for t in times)
    result: list[float] = []
    prev = None
    for t_val in ordered:
        if not (eps <= t_val <= 1.0):
            raise ValueError(f"Time {t_val:.4f} outside valid range [{eps}, 1.0].")
        if prev is not None and abs(t_val - prev) < 1e-8:
            raise ValueError("Time snapshots must be strictly increasing.")
        result.append(t_val)
        prev = t_val
    if len(result) < 2:
        raise ValueError("At least two distinct time snapshots are required.")
    return result


def _choose_steps(cfg: dict, steps_override: int | None) -> int:
    if steps_override is not None:
        return max(2, int(steps_override))
    if "continuity_steps" in cfg:
        return max(2, int(cfg["continuity_steps"]))
    if "density_eval_steps" in cfg:
        return max(2, int(cfg["density_eval_steps"]))
    return max(2, int(cfg.get("nll_eval_steps", cfg.get("num_steps", 1000))))


def _evaluate_log_density_snapshots(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    times: list[float],
    *,
    steps: int,
    method: str,
    batch_size: int,
) -> np.ndarray:
    num_points = grid_points.shape[0]
    log_values = np.empty((len(times), *grid_shape), dtype=np.float32)
    for idx, t_val in enumerate(times):
        print(f"[continuity] Evaluating log-density at t={t_val:.4f}")
        batches: list[np.ndarray] = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            xb = torch.from_numpy(grid_points[start:end]).to(device=device, dtype=torch.float32)
            with torch.enable_grad():
                logp = integrate_log_probability(
                    model,
                    xb,
                    cfg,
                    steps=steps,
                    eps=t_val,
                    method=method,
                )
            batches.append(logp.detach().cpu().numpy())
        stacked = np.concatenate(batches, axis=0)
        log_values[idx] = stacked.reshape(grid_shape).astype(np.float32, copy=False)
    return log_values


def _compute_velocity_and_divergence(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    times: list[float],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_points = grid_points.shape[0]
    dim = grid_points.shape[1]
    velocities = np.empty((len(times), *grid_shape, dim), dtype=np.float32)
    divergences = np.empty((len(times), *grid_shape), dtype=np.float32)
    eps = float(cfg.get("eps", 1e-3))
    for idx, t_val in enumerate(times):
        print(f"[continuity] Evaluating velocity/divergence at t={t_val:.4f}")
        vel_batches: list[np.ndarray] = []
        div_batches: list[np.ndarray] = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            xb = torch.from_numpy(grid_points[start:end]).to(device=device, dtype=torch.float32)
            xb = xb.requires_grad_(True)
            t_batch = torch.full((xb.shape[0],), t_val, device=device, dtype=xb.dtype)
            t_batch = torch.clamp(t_batch, min=eps, max=1.0)
            with torch.enable_grad():
                drift = drift_pf(model, xb, t_batch, cfg)
                div = divergence_exact(drift, xb)
            vel_batches.append(drift.detach().cpu().numpy())
            div_batches.append(div.detach().cpu().numpy())
        velocities[idx] = np.concatenate(vel_batches, axis=0).reshape(*grid_shape, dim).astype(np.float32, copy=False)
        divergences[idx] = np.concatenate(div_batches, axis=0).reshape(grid_shape).astype(np.float32, copy=False)
    return velocities, divergences


def _finite_difference(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    num_times = times.shape[0]
    if num_times < 2:
        raise ValueError("Need at least two snapshots to compute temporal derivatives.")
    deriv = np.empty_like(values, dtype=np.float64)
    for idx in range(num_times):
        if idx == 0:
            dt = times[1] - times[0]
            deriv[idx] = (values[1] - values[0]) / dt
        elif idx == num_times - 1:
            dt = times[-1] - times[-2]
            deriv[idx] = (values[-1] - values[-2]) / dt
        else:
            dt = times[idx + 1] - times[idx - 1]
            deriv[idx] = (values[idx + 1] - values[idx - 1]) / dt
    return deriv.astype(np.float32)


def _spatial_gradients(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("values should have shape (T, H, W).")
    num_times, height, width = values.shape
    edge_order = 2 if height >= 3 and width >= 3 else 1
    x_coords = grid_x[0]
    y_coords = grid_y[:, 0]
    gradients = np.empty((num_times, height, width, 2), dtype=np.float32)
    for idx in range(num_times):
        grad_y, grad_x = np.gradient(values[idx], y_coords, x_coords, edge_order=edge_order)
        gradients[idx, ..., 0] = grad_x.astype(np.float32, copy=False)
        gradients[idx, ..., 1] = grad_y.astype(np.float32, copy=False)
    return gradients


def calculate_continuity_residual(
    *,
    checkpoint_path: Path,
    resolution: int,
    span_sigma: float,
    times: Sequence[float],
    batch_size: int,
    method: str | None = None,
    steps: int | None = None,
    output_path: Path | None = None,
) -> dict[str, np.ndarray]:
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")
    batch_size = max(1, int(batch_size))

    device = get_device()
    model = build_model(base_config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    merged_config = merge_config(checkpoint, checkpoint_path)
    eps = float(merged_config.get("eps", 1e-3))
    validated_times = validate_times(times, eps)
    steps_eval = _choose_steps(merged_config, steps)
    method_name = str(method or merged_config.get("density_method", "Heun"))

    grid_x, grid_y, grid_points = build_grid(merged_config, resolution, span_sigma)
    grid_shape = grid_x.shape

    model.eval()

    log_density = _evaluate_log_density_snapshots(
        model,
        merged_config,
        device,
        grid_points,
        grid_shape,
        validated_times,
        steps=steps_eval,
        method=method_name,
        batch_size=batch_size,
    )
    velocity, divergence = _compute_velocity_and_divergence(
        model,
        merged_config,
        device,
        grid_points,
        grid_shape,
        validated_times,
        batch_size=batch_size,
    )

    time_arr = np.asarray(validated_times, dtype=np.float64)
    dlog_dt = _finite_difference(time_arr, log_density.astype(np.float64))
    grad_log = _spatial_gradients(log_density.astype(np.float64), grid_x, grid_y)
    conv_term = velocity[..., 0] * grad_log[..., 0] + velocity[..., 1] * grad_log[..., 1]
    cont_residual = dlog_dt + conv_term + divergence

    result = {
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
        "times": np.asarray(validated_times, dtype=np.float32),
        "log_density": log_density.astype(np.float32),
        "dlog_dt": dlog_dt.astype(np.float32),
        "divergence": divergence.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "grad_log_density": grad_log.astype(np.float32),
        "convective_term": conv_term.astype(np.float32),
        "cont_residual": cont_residual.astype(np.float32),
        "mu": np.asarray(merged_config["mu"], dtype=np.float32),
        "sigma": np.asarray(merged_config["sigma"], dtype=np.float32),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

    return result
