from typing import Dict
import matplotlib.pyplot as plt

import torch
from numpy.linalg import norm
from config import config_path
from sde import beta_t, drift_pf
import numpy as np
from util import (
    fmt,
    scatter_side_by_side_shared,
)

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
        score = model(x, t.expand(n_samples))
        drift = (-0.5 * beta) * x - beta * score

        x = x + drift * dt + torch.sqrt(beta * abs(dt)) * torch.randn_like(x)
        sigma = torch.sqrt(1.0 - alpha_bar(t_next, config["beta_min"], config["beta_max"]).clamp(min=1e-12))

        for _ in range(corrector_steps):
            score = model(x, t_next.expand(n_samples))
            avg_norm = score.reshape(n_samples, -1).norm(dim=1).mean().clamp_min(1e-6)
            step_eps = 2.0 * (snr * sigma / avg_norm) ** 2
            x = x + step_eps * score + torch.sqrt(2.0 * step_eps) * torch.randn_like(x)

    t_final = ts[-1]
    ab = alpha_bar(t_final, config["beta_min"], config["beta_max"])
    score = model(x, t_final.expand(n_samples))
    return (x + (1.0 - ab) * score) / torch.sqrt(ab + 1e-12)


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

    ab = alpha_bar(ts[-1], config["beta_min"], config["beta_max"])
    score = model(x, ts[-1].expand(n_samples))
    return (x + (1.0 - ab) * score) / torch.sqrt(ab + 1e-12)


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
    beta = beta_t(
        t_scalar.expand(x_state.shape[0]),
        config["beta_min"],
        config["beta_max"],
    ).clamp_min(1e-8).view(-1, 1)
    return (-0.5 * beta) * x_state - 0.5 * beta * score


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
    checkpoint_path: str = config_path,
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


def plot_sample_scatter(true_samples: np.ndarray, gen_samples: np.ndarray, config: Dict):
    fig, _ = scatter_side_by_side_shared(
        true_samples,
        gen_samples,
        config,
        max_show=min(config["n_test"], config["n_gen"]),
    )
    plt.show()
    return fig


def report_statistics(gen: np.ndarray, config: Dict, te: np.ndarray):
    mu_gen = gen.mean(axis=0)
    cov_gen = np.cov(gen, rowvar=False)
    var_gen = np.diag(cov_gen)
    mu_true = np.asarray(config["mu"], dtype=np.float64)
    cov_true = np.asarray(config["sigma"], dtype=np.float64)
    var_true = np.diag(cov_true)
    mu_test = te.mean(axis=0)
    cov_test = np.cov(te, rowvar=False)
    var_test = np.diag(cov_test)

    print("mean_gen =", fmt(mu_gen))
    print("var_gen  =", fmt(var_gen))
    print("cov_gen  =\n", fmt(cov_gen))
    print("\n=== True (parameters) ===")
    print("mean_true =", fmt(mu_true))
    print("var_true  =", fmt(var_true))
    print("cov_true  =\n", fmt(cov_true))
    print("\n=== Test sample (te) ===")
    print("mean_test =", fmt(mu_test))
    print("var_test  =", fmt(var_test))
    print("cov_test  =\n", fmt(cov_test))
    print("\n--- Errors vs TRUE ---")
    print("||mean_gen - mean_true||_2 =", float(norm(mu_gen - mu_true)))
    print("||cov_gen  - cov_true ||_F =", float(norm(cov_gen - cov_true, ord='fro')))
