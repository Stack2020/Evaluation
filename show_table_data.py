from __future__ import annotations
import math
from typing import Iterable
import numpy as np
import torch

from config import config, get_device
from data_loader import generate_test, involute_params
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability


def involute_log_density(
    points: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    noise_std: float,
    n_quad: int = 1024,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Numerically evaluate the true log-density of the involute dataset:
    x | r ~ N([r*sin(2r), r*cos(2r)], noise_std^2 I), r ~ U[r_min, r_max].
    Approximated via uniform quadrature over r.
    """
    pts = np.asarray(points, dtype=np.float64)
    r = np.linspace(r_min, r_max, n_quad, dtype=np.float64)
    base_x = r * np.sin(2.0 * r)
    base_y = r * np.cos(2.0 * r)

    r_range = max(r_max - r_min, 1e-12)
    dr = r_range / max(n_quad - 1, 1)
    var = noise_std * noise_std
    log_const = math.log(dr) - math.log(r_range) - math.log(2.0 * math.pi * var)

    logp = np.empty(pts.shape[0], dtype=np.float64)

    for start in range(0, pts.shape[0], chunk_size):
        end = min(start + chunk_size, pts.shape[0])
        chunk = pts[start:end]
        dx = chunk[:, 0:1] - base_x[None, :]
        dy = chunk[:, 1:2] - base_y[None, :]
        exponent = -(dx * dx + dy * dy) / (2.0 * var)
        max_exp = np.max(exponent, axis=1, keepdims=True)
        lse = max_exp + np.log(np.exp(exponent - max_exp).sum(axis=1, keepdims=True))
        logp[start:end] = (log_const + lse.squeeze(1))
    return logp


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
    """Evaluate model log-density on arbitrary 2D points."""
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

    # Fixed test set (2D) for all evaluations
    test_np = generate_test(cfg, seed=int(cfg.get("eval_seed_offset", 0))).astype(np.float32)
    batch_size = int(cfg.get("score_batch_size", 512))

    steps_list: Iterable[int] = (100,200,250,500)

    r_min, r_max, noise_std = involute_params(cfg)
    analytic_logp = involute_log_density(
        test_np.astype(np.float64),
        r_min=r_min,
        r_max=r_max,
        noise_std=noise_std,
    )
    analytic_nll = float(-analytic_logp.mean())

    method = str(cfg.get("density_method", "Heun"))
    eps_val = float(cfg.get("nll_eval_eps", cfg.get("eps", 1e-3)))

    print(f"Test samples: {test_np.shape[0]} (2D)")
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
            f"{pf_nll_value:.6f}\t"
            f"{analytic_nll:.6f}\t"
            f"{delta:.6f}\t"
            f"{kl:.6f}"
        )


if __name__ == "__main__":
    main()
