from typing import Dict
import matplotlib.pyplot as plt

import torch
from numpy.linalg import norm

from sde import beta_t, drift_pf, diffusion_coeff, v_to_score, v_to_x0
import numpy as np


def fmt(array) -> str:
    """Nicely format arrays for logging statistics."""
    arr = np.asarray(array, dtype=np.float64)
    return np.array2string(arr, precision=4, suppress_small=True)


def alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:

    return torch.exp(-(beta_min * t + 0.5 * (beta_max - beta_min) * t * t))


@torch.no_grad()
def pc_sample(
    model,
    config: Dict,
    device: torch.device,
    n_samples: int,
    steps: int | None = None,
    snr: float | None = None,
    corrector_steps: int | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    steps = config["num_steps"] if steps is None else steps
    snr = config["snr"] if snr is None else snr
    corrector_steps = config["corrector_steps"] if corrector_steps is None else corrector_steps
    eps = config["eps"] if eps is None else eps

    x = torch.randn(n_samples, 2, device=device)
    ts = torch.linspace(1.0, eps, steps, device=device)

    for i in range(steps - 1):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t
        beta = beta_t(t, config["beta_min"], config["beta_max"]).clamp_min(1e-8)
        diffusion = diffusion_coeff(t, config)
        t_batch = t.expand(n_samples)
        score = v_to_score(model(x, t_batch), x, t_batch, config)
        drift = (-0.5 * beta) * x - (diffusion**2) * score

        noise_scale = diffusion * torch.sqrt(torch.clamp(abs(dt), min=1e-12))
        x = x + drift * dt + noise_scale * torch.randn_like(x)
        #标准差
        sigma = torch.sqrt(1.0 - alpha_bar(t_next, config["beta_min"], config["beta_max"]).clamp(min=1e-12))

        for _ in range(corrector_steps):
            t_next_batch = t_next.expand(n_samples)
            score = v_to_score(model(x, t_next_batch), x, t_next_batch, config)
            avg_norm = score.reshape(n_samples, -1).norm(dim=1).mean().clamp_min(1e-6)
            step_eps = 2.0 * (snr * sigma / avg_norm) ** 2
            x = x + step_eps * score + torch.sqrt(2.0 * step_eps) * torch.randn_like(x)

    t_final = ts[-1]
    t_final_batch = t_final.expand(n_samples)
    v_final = model(x, t_final_batch)
    return v_to_x0(v_final, x, t_final_batch, config)


@torch.no_grad()
def pf_sample(
    model,
    config: Dict,
    device: torch.device,
    n_samples: int,
    steps: int | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    steps = config["num_steps"] if steps is None else steps
    eps = config["eps"] if eps is None else eps

    x = torch.randn(n_samples, 2, device=device)
    ts = torch.linspace(1.0, eps, steps, device=device)

    for i in range(steps - 1):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t
        t_curr = t.expand(n_samples)

        drift_t = drift_pf(model, x, t_curr, config)
        x_pred = x + dt * drift_t

        t_next_expand = t_next.expand(n_samples)
        drift_next = drift_pf(model, x_pred, t_next_expand, config)
        x = x + 0.5 * dt * (drift_t + drift_next)

    t_final = ts[-1]
    t_final_batch = t_final.expand(n_samples)
    v_final = model(x, t_final_batch)
    return v_to_x0(v_final, x, t_final_batch, config)


def pf_drift_true(
    x_state: torch.Tensor,
    t_scalar: torch.Tensor,
    config: Dict,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eye: torch.Tensor,
) -> torch.Tensor:
    ab = alpha_bar(t_scalar, config["beta_min"], config["beta_max"])
    mean_t = torch.sqrt(ab) * mu
    cov_t = ab * sigma + (1.0 - ab) * eye
    score = -torch.linalg.solve(cov_t, (x_state - mean_t).T).T
    t_expanded = t_scalar.expand(x_state.shape[0])
    beta = beta_t(
        t_expanded,
        config["beta_min"],
        config["beta_max"],
    ).clamp_min(1e-8).reshape(-1, 1)
    diffusion = diffusion_coeff(t_expanded, config).reshape(-1, 1)
    diffusion_sq = diffusion * diffusion
    return (-0.5 * beta) * x_state - 0.5 * diffusion_sq * score


@torch.no_grad()
def oracle_sample(
    config: Dict,
    device: torch.device,
    n_samples: int,
    steps: int | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    steps = config["num_steps"] if steps is None else steps
    eps = config["eps"] if eps is None else eps

    x = torch.randn(n_samples, 2, device=device)
    dtype = x.dtype
    mu = torch.as_tensor(config["mu"], device=device, dtype=dtype)
    sigma = torch.as_tensor(config["sigma"], device=device, dtype=dtype)
    eye = torch.eye(mu.shape[0], device=device, dtype=dtype)
    ts = torch.linspace(1.0, eps, steps, device=device, dtype=dtype)

    for i in range(steps - 1):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t

        drift_t = pf_drift_true(x, t, config, mu, sigma, eye)
        x_pred = x + dt * drift_t
        drift_next = pf_drift_true(x_pred, t_next, config, mu, sigma, eye)
        x = x + 0.5 * dt * (drift_t + drift_next)

    ab_final = alpha_bar(ts[-1], config["beta_min"], config["beta_max"])
    
    mean_final = torch.sqrt(ab_final) * mu
    cov_final = ab_final * sigma + (1.0 - ab_final) * eye

    score_final = -torch.linalg.solve(cov_final, (x - mean_final).T).T

    return (x + (1.0 - ab_final) * score_final) / torch.sqrt(ab_final + 1e-12)


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def generate_samples(
    model,
    config: Dict,
    device: torch.device,
    checkpoint_path: str,
    method: str = "pc",
    **kwargs,
) -> np.ndarray:
    
    model.eval()
    load_checkpoint(model, checkpoint_path, device)
    with torch.no_grad():
        if method.lower() == "pf":
            samples = pf_sample(model, config, device, config["n_gen"], **kwargs)
        else:
            samples = pc_sample(model, config, device, config["n_gen"], **kwargs)
    return samples.cpu().numpy()

