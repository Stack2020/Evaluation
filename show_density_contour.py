from __future__ import annotations
import math
import numpy as np
import torch
from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability, set_seed

def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute log-density of a multivariate Gaussian at given points."""
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
    batch_size: int = 128,
) -> np.ndarray:
    """Evaluate model log-density on a set of 2D points."""
    model.eval()
    log_values = []
    total = points.shape[0]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_np = points[start:end]
        xb = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                xb,
                cfg,
                steps=steps,
                eps=eps,
                method=method,
            )
        log_values.append(logp.detach().cpu().numpy())

    return np.concatenate(log_values, axis=0)


def main() -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, config["config_path"], device)
    checkpoint_config: dict = checkpoint.get("config", {}).copy()
    for key, value in config.items():
        checkpoint_config.setdefault(key, value)
    print("Using configuration:", checkpoint_config)
    test_np = generate_test(checkpoint_config, seed=seed)
    max_points = int(checkpoint_config.get("density_eval_points", test_np.shape[0]))
    eval_np = test_np[:max(1, min(max_points, test_np.shape[0]))]

    mean = np.asarray(checkpoint_config["mu"], dtype=np.float64)
    cov = np.asarray(checkpoint_config["sigma"], dtype=np.float64)
    
    true_log_density = gaussian_log_density(eval_np, mean, cov)

    steps_eval = int(checkpoint_config.get("density_eval_steps", 1000))
    print(f"Density eval steps: {steps_eval}, points: {eval_np.shape[0]}")

    eps_eval = float(
        checkpoint_config.get(
            "nll_eval_eps",
            checkpoint_config.get("eps", 1e-3),
        )
    )
    method = checkpoint_config.get("density_method", "Heun")

    model_log_density = evaluate_model_log_density(
        model,
        checkpoint_config,
        device,
        eval_np,
        steps=steps_eval,
        eps=eps_eval,
        method=method,
    )

    model_density = np.exp(model_log_density)
    
    true_density = np.exp(true_log_density)

    abs_err = np.abs(model_density - true_density)
    rel_err = abs_err / np.maximum(true_density, 1e-12)
    print(
        "Density error | abs(mean/median/max) = "
        f"{abs_err.mean():.4e}/{np.median(abs_err):.4e}/{abs_err.max():.4e}"
    )
    print(
        "Density error | rel(mean/median/p95) = "
        f"{rel_err.mean():.4e}/{np.median(rel_err):.4e}/{np.percentile(rel_err, 95):.4e}"
    )

    density_data = np.column_stack(
        (eval_np[:, 0], eval_np[:, 1], model_density, true_density, abs_err, rel_err)
    )
    np.savetxt(
        "density.csv",
        density_data,
        delimiter=",",
        header="x,y,model_density,true_density,abs_err,rel_err",
        comments="",
    )

if __name__ == "__main__":
    main()
