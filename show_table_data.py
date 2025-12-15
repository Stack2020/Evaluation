from __future__ import annotations
import math
from typing import Iterable
import numpy as np
import torch

from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability

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
    """Evaluate model log-density on arbitrary points (dimension inferred from cfg)."""
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


def main() -> None:
    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(config["config_path"]), device)
    ckpt_cfg = checkpoint.get("config", {})
    cfg = config.copy()
    cfg.update(ckpt_cfg)
    cfg["config_path"] = str(config["config_path"])

    # Fixed test set for all evaluations
    test_np = generate_test(cfg, seed=int(cfg.get("eval_seed_offset", 0))).astype(np.float32)
    batch_size = int(cfg.get("score_batch_size", 512))
    
    steps_list: Iterable[int] = (50, 100, 150, 200, 250, 300)

    mean = np.asarray(cfg["mu"], dtype=np.float64)
    cov = np.asarray(cfg["sigma"], dtype=np.float64)
    analytic_logp = gaussian_log_density(test_np.astype(np.float64), mean, cov)
    analytic_nll = float(-analytic_logp.mean())

    method = str(cfg.get("density_method", "Heun"))
    eps_val = float(cfg.get("nll_eval_eps", cfg.get("eps", 1e-3)))

    data_dim = test_np.shape[1] if test_np.ndim > 1 else 1
    print(f"Test samples: {test_np.shape[0]} (dim={data_dim})")
    print("NFE\tPF-ODE NLL\tAnalytic NLL\t|Δ|\tKL(p_data||p_theta)")

    for steps in steps_list:
        model_logp = evaluate_model_log_density(
            model,
            cfg,
            device,
            test_np,
            steps=steps,
            eps=eps_val,
            method=method,
            batch_size=batch_size,
        )
        # PF-ODE NLL value (expected -log p under the data distribution)
        pf_nll_value = float(-model_logp.mean())
        delta = abs(pf_nll_value - analytic_nll)
        kl = float((analytic_logp - model_logp).mean())

        print(
            f"{steps}\t"
            f"{pf_nll_value:.4f}\t"
            f"{analytic_nll:.4f}\t"
            f"{delta:.4f}\t"
            f"{kl:.4f}"
        )


if __name__ == "__main__":
    main()
