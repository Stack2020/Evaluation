from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from config import config as base_config, get_device
from data_loader import involute_params
from model import build_model
from sample import load_checkpoint
from show_density_heatmap import (
    DEFAULT_SETTINGS as HEATMAP_SETTINGS,
    build_grid,
    evaluate_model_log_density,
    involute_log_density,
)

NFE_VALUES: Sequence[int] = (40,80,160,320,400)
EVALS_PER_STEP = {"heun": 2, "euler": 1, "rk45": 4}

DEFAULT_SETTINGS = {
    "checkpoint_path": Path(base_config["config_path"]),
    "resolution": int(HEATMAP_SETTINGS.get("resolution", 200)),
    "span_sigma": float(HEATMAP_SETTINGS.get("span_sigma", 4.0)),
    "batch_size": int(HEATMAP_SETTINGS.get("batch_size", 512)),
    "plot_bounds": HEATMAP_SETTINGS.get("plot_bounds", None),
    "output_path": Path("involute.npz"),
}


def nfe_to_steps(nfe: int, method: str) -> int:
    """Convert a target NFE budget to integration steps for the chosen solver."""
    method_name = method.lower()
    evals_per_step = max(1, EVALS_PER_STEP.get(method_name, 1))
    return max(2, math.ceil(nfe / evals_per_step) + 1)


def main() -> None:
    settings = DEFAULT_SETTINGS.copy()
    checkpoint_path: Path = settings["checkpoint_path"]
    resolution: int = settings["resolution"]
    span_sigma: float = settings["span_sigma"]
    batch_size: int = settings["batch_size"]
    plot_bounds = settings["plot_bounds"]
    output_path: Path = settings["output_path"]

    device = get_device()
    model = build_model(base_config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_config = checkpoint.get("config", {}).copy()
    merged_config = base_config.copy()
    merged_config.update(checkpoint_config)
    merged_config["config_path"] = str(checkpoint_path)

    grid_x, grid_y, grid_points = build_grid(
        merged_config,
        resolution=resolution,
        span_sigma=span_sigma,
        fixed_bounds=plot_bounds,
    )

    r_min, r_max, noise_std = involute_params(merged_config)
    
    analytic_log_density = involute_log_density(
        grid_points,
        r_min=r_min,
        r_max=r_max,
        noise_std=noise_std,
    ).reshape(grid_x.shape)
    analytic_density = np.exp(analytic_log_density)
    analytic_density = np.nan_to_num(analytic_density, nan=0.0, posinf=0.0, neginf=0.0)

    method = str(merged_config.get("density_method", "Heun"))
    method_lower = method.lower()
    evals_per_step = max(1, EVALS_PER_STEP.get(method_lower, 1))
    eps_eval = float(merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))

    nfe_list: list[int] = []
    mae_list: list[float] = []
    rmse_list: list[float] = []
    max_err_list: list[float] = []

    for nfe in NFE_VALUES:
        steps = nfe_to_steps(int(nfe), method)
        model_log_density = evaluate_model_log_density(
            model,
            merged_config,
            device,
            grid_points,
            steps=steps,
            eps=eps_eval,
            method=method,
            batch_size=max(1, batch_size),
        ).reshape(grid_x.shape)
        model_density = np.exp(model_log_density)
        model_density = np.nan_to_num(model_density, nan=0.0, posinf=0.0, neginf=0.0)

        diff = model_density - analytic_density
        mae = round(float(np.mean(np.abs(diff))), 8)
        rmse = round(float(np.sqrt(np.mean(diff * diff))), 8)
        max_err = round(float(np.max(np.abs(diff))), 8)

        nfe_list.append(int(nfe))
        mae_list.append(mae)
        rmse_list.append(rmse)
        max_err_list.append(max_err)

        actual_nfe = evals_per_step * (steps - 1)
        print(
            f"NFE={nfe} (steps={steps}, actual NFE≈{actual_nfe}): "
            f"MAE={mae:.8f}, RMSE={rmse:.8f}, MAX_ERR={max_err:.8f}"
        )

    np.savez(
        output_path,
        NFE=np.asarray(nfe_list, dtype=np.int64),
        MAE=np.asarray(mae_list, dtype=np.float64),
        RMSE=np.asarray(rmse_list, dtype=np.float64),
        MAX_ERR=np.asarray(max_err_list, dtype=np.float64),
    )

    print(f"Saved NFE/MAE/RMSE arrays to {output_path.resolve()}")


if __name__ == "__main__":
    main()
