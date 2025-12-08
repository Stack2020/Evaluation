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


def compute_fields(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    grid_points: np.ndarray,
    times: list[float],
    batch_size: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    num_points = grid_points.shape[0]
    scores = np.empty((len(times), *grid_shape, 2), dtype=np.float32)
    residuals = np.empty((len(times), *grid_shape), dtype=np.float32)

    model.eval()
    eps = float(cfg.get("eps", 1e-3))

    for time_idx, t_val in enumerate(times):
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

    return scores, residuals


def calculate_score_field(
    *,
    checkpoint_path: Path,
    resolution: int,
    span_sigma: float,
    times: Sequence[float],
    batch_size: int,
    output_path: Path | None = None,
) -> dict[str, np.ndarray]:
    device = get_device()
    model = build_model(base_config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    merged_config = merge_config(checkpoint, checkpoint_path)
    eps = float(merged_config.get("eps", 1e-3))
    validated_times = validate_times(times, eps)

    grid_x, grid_y, grid_points = build_grid(merged_config, resolution, span_sigma)
    scores, residuals = compute_fields(
        model,
        merged_config,
        device,
        grid_points,
        validated_times,
        max(1, int(batch_size)),
        grid_x.shape,
    )

    result = {
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
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
