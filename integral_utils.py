from __future__ import annotations
from typing import Callable, Dict
import torch

from sde import drift_pf, divergence_exact, prior_logp_standard_normal


def drift_and_divergence(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    config: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate drift and divergence at the provided state/time."""
    x_req = x.detach().requires_grad_(True)
    drift = drift_pf(model, x_req, t, config)
    div = divergence_exact(drift, x_req)
    return drift.detach(), div


def euler_step(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    config: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_t = t_next - t_curr
    h = delta_t.view(-1, 1)

    drift_curr, div_curr = drift_and_divergence(model, x, t_curr, config)

    x_next = x + h * drift_curr
    g_next = g - delta_t * div_curr
    return x_next, g_next


def heun_step(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    config: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_t = t_next - t_curr
    h = delta_t.view(-1, 1)

    drift_curr, div_curr = drift_and_divergence(model, x, t_curr, config)

    x_tilde = x + h * drift_curr
    drift_next, div_next = drift_and_divergence(model, x_tilde, t_next, config)

    x_next = x + 0.5 * h * (drift_curr + drift_next)
    g_next = g - 0.5 * delta_t * (div_curr + div_next)
    return x_next, g_next


def rk4_step(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    config: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_t = t_next - t_curr
    half_dt = 0.5 * delta_t
    h = delta_t.view(-1, 1)
    h_half = half_dt.view(-1, 1)

    drift_k1, div_k1 = drift_and_divergence(model, x, t_curr, config)

    x_mid1 = x + h_half * drift_k1
    t_mid = t_curr + half_dt
    drift_k2, div_k2 = drift_and_divergence(model, x_mid1, t_mid, config)

    x_mid2 = x + h_half * drift_k2
    drift_k3, div_k3 = drift_and_divergence(model, x_mid2, t_mid, config)

    x_end = x + h * drift_k3
    drift_k4, div_k4 = drift_and_divergence(model, x_end, t_next, config)

    x_next = x + (h / 6.0) * (drift_k1 + 2.0 * drift_k2 + 2.0 * drift_k3 + drift_k4)
    g_next = g - (delta_t / 6.0) * (div_k1 + 2.0 * div_k2 + 2.0 * div_k3 + div_k4)
    return x_next, g_next


STEP_MAP: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "euler": euler_step,
    "heun": heun_step,
    "rk4": rk4_step,
}


def integrate_log_probability(
    model,
    x0: torch.Tensor,
    config: Dict,
    steps: int,
    eps: float,
    method: str,
) -> torch.Tensor:

    step_key = method.lower()
    if step_key not in STEP_MAP:
        raise ValueError(f"Unsupported integration method '{method}'. Expected one of Euler, Heun, RK4.")

    step_fn = STEP_MAP[step_key]
    device = x0.device
    dtype = x0.dtype
    batch = x0.shape[0]

    ts = torch.linspace(eps, 1.0, steps + 1, device=device, dtype=dtype)
    x = x0.detach()
    g = torch.zeros(batch, device=device, dtype=dtype)

    for idx in range(steps):
        t_curr = ts[idx].expand(batch)
        t_next = ts[idx + 1].expand(batch)
        x, g = step_fn(model, x, g, t_curr, t_next, config)

    log_p1 = prior_logp_standard_normal(x)
    return log_p1 - g

__all__ = ["integrate_log_probability"]
