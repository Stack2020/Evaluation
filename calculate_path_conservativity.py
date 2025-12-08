from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from config import config as base_config, get_device
from model import build_model
from sample import load_checkpoint
from sde import drift_pf, v_to_score

__all__ = [
    "merge_config",
    "build_grid",
    "validate_times",
    "build_rectangular_loops",
    "evaluate_vector_field",
    "compute_loop_integrals",
    "calculate_path_conservativity",
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
    ordered = sorted(float(t) for t in times)
    if not ordered:
        raise ValueError("At least one time snapshot is required.")
    result: list[float] = []
    prev: float | None = None
    for t_val in ordered:
        if not (eps <= t_val <= 1.0):
            raise ValueError(f"Time {t_val:.4f} outside valid range [{eps}, 1.0].")
        if prev is not None and abs(t_val - prev) < 1e-8:
            raise ValueError("Time snapshots must be strictly increasing.")
        result.append(t_val)
        prev = t_val
    return result


def build_rectangular_loops(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must share the same shape.")
    height, width = grid_x.shape
    if height < 2 or width < 2:
        raise ValueError("Need at least a 2x2 grid to define polygonal loops.")
    num_loops = (height - 1) * (width - 1)
    loop_indices = np.zeros((num_loops, 5, 2), dtype=np.int32)
    idx = 0
    for iy in range(height - 1):
        for ix in range(width - 1):
            loop_indices[idx] = np.array(
                [
                    [iy, ix],
                    [iy, ix + 1],
                    [iy + 1, ix + 1],
                    [iy + 1, ix],
                    [iy, ix],
                ],
                dtype=np.int32,
            )
            idx += 1
    coords = np.stack((grid_x, grid_y), axis=-1)
    rows = loop_indices[..., 0]
    cols = loop_indices[..., 1]
    vertices = coords[rows, cols]
    edges = vertices[:, 1:, :] - vertices[:, :-1, :]
    lengths = np.linalg.norm(edges, axis=2).sum(axis=1).astype(np.float32)
    return loop_indices, lengths, (height - 1, width - 1)


def evaluate_vector_field(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    times: Sequence[float],
    *,
    batch_size: int,
    field_type: str,
) -> np.ndarray:
    num_points = grid_points.shape[0]
    outputs = np.empty((len(times), *grid_shape, 2), dtype=np.float32)
    field_name = field_type.lower()
    if field_name not in {"score", "drift"}:
        raise ValueError("field_type must be either 'score' or 'drift'.")

    model.eval()
    eps = float(cfg.get("eps", 1e-3))

    for idx, t_val in enumerate(times):
        chunks: list[np.ndarray] = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            xb = torch.from_numpy(grid_points[start:end]).to(device=device, dtype=torch.float32)
            t_batch = torch.full((xb.shape[0],), float(t_val), device=device, dtype=xb.dtype)
            t_batch = torch.clamp(t_batch, min=eps, max=1.0)
            with torch.no_grad():
                if field_name == "score":
                    vec = v_to_score(model(xb, t_batch), xb, t_batch, cfg)
                else:
                    vec = drift_pf(model, xb, t_batch, cfg)
            chunks.append(vec.detach().cpu().numpy())
        outputs[idx] = np.concatenate(chunks, axis=0).reshape(*grid_shape, 2).astype(np.float32, copy=False)
        print(f"[path] Evaluated vector field at t={t_val:.4f}")
    return outputs


def compute_loop_integrals(
    field_values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    loop_indices: np.ndarray,
    loop_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.stack((grid_x, grid_y), axis=-1)
    rows = loop_indices[..., 0]
    cols = loop_indices[..., 1]
    vertices = coords[rows, cols]
    edges = vertices[:, 1:, :] - vertices[:, :-1, :]
    safe_lengths = np.maximum(loop_lengths, 1e-12).astype(np.float32, copy=False)

    num_times = field_values.shape[0]
    num_loops = loop_indices.shape[0]
    integrals = np.empty((num_times, num_loops), dtype=np.float32)
    normalized = np.empty_like(integrals)

    # Precompute edge vectors for all loops (geometry-only).
    coords = np.stack((grid_x, grid_y), axis=-1)
    rows = loop_indices[..., 0]
    cols = loop_indices[..., 1]
    vertices = coords[rows, cols]
    edges = vertices[:, 1:, :] - vertices[:, :-1, :]

    for t_idx in range(num_times):
        slice_field = field_values[t_idx]
        vec_at_vertices = slice_field[rows, cols]
        # Use a trapezoidal rule along each edge: average the field at edge endpoints.
        vectors = 0.5 * (vec_at_vertices[:, :-1, :] + vec_at_vertices[:, 1:, :])
        dots = (vectors * edges).sum(axis=2)
        loop_vals = dots.sum(axis=1)
        integrals[t_idx] = loop_vals.astype(np.float32, copy=False)
        normalized[t_idx] = np.abs(loop_vals) / safe_lengths
    return integrals, normalized


def calculate_path_conservativity(
    *,
    checkpoint_path: Path,
    resolution: int,
    span_sigma: float,
    times: Sequence[float],
    batch_size: int,
    field_type: str = "drift",
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

    grid_x, grid_y, grid_points = build_grid(merged_config, resolution, span_sigma)
    loop_indices, loop_lengths, loop_shape = build_rectangular_loops(grid_x, grid_y)

    field = evaluate_vector_field(
        model,
        merged_config,
        device,
        grid_points,
        grid_x.shape,
        validated_times,
        batch_size=batch_size,
        field_type=field_type,
    )
    loop_integrals, loop_metric = compute_loop_integrals(field, grid_x, grid_y, loop_indices, loop_lengths)
    metric_map = loop_metric.reshape((len(validated_times), *loop_shape))
    path_metric = loop_metric.mean(axis=1)

    result = {
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
        "times": np.asarray(validated_times, dtype=np.float32),
        "vector_field": field.astype(np.float32),
        "loop_indices": loop_indices.astype(np.int32),
        "loop_lengths": loop_lengths.astype(np.float32),
        "loop_integrals": loop_integrals.astype(np.float32),
        "loop_metric": loop_metric.astype(np.float32),
        "loop_metric_map": metric_map.astype(np.float32),
        "path_metric": path_metric.astype(np.float32),
        "mu": np.asarray(merged_config["mu"], dtype=np.float32),
        "sigma": np.asarray(merged_config["sigma"], dtype=np.float32),
        "loop_shape": np.asarray(loop_shape, dtype=np.int32),
        "field_type": np.asarray(str(field_type), dtype="<U16"),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

    return result
