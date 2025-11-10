from typing import Dict, Iterable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from sde import (
    dsm_loss_batch,
    drift_pf,
    prior_logp_standard_normal,
    divergence_exact,
    terminal_logp_gaussian
    )


def build_time_grid(
    steps: int,
    eps_val: float,
    device: torch.device,
    dtype: torch.dtype,
    config: Dict,
) -> torch.Tensor:
    schedule = str(config.get("nll_time_schedule", "linear")).lower()
    if schedule == "power":
        gamma = max(float(config.get("nll_time_gamma", 2.0)), 1e-6)
        tau = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
        scaled = torch.pow(tau, gamma)
        return eps_val + (1.0 - eps_val) * scaled
    return torch.linspace(eps_val, 1.0, steps, device=device, dtype=dtype)

def integrate_log_probability(
    model,
    x0: torch.Tensor,
    config: Dict,
    *,
    steps: int,
    eps: float | None = None,
    method: str = "Heun",
) -> torch.Tensor:
    
    method_name = method.lower()
    eps_val = float(eps or config.get("nll_eval_eps", config.get("eps", 1e-3)))
    if steps < 2:
        raise ValueError("steps must be at least 2 to perform integration.")

    device, dtype = x0.device, x0.dtype
    batch = x0.shape[0]
    ts = build_time_grid(steps, eps_val, device, dtype, config)
    dts = ts[1:] - ts[:-1]
    x = x0.detach()
    accum = torch.zeros(batch, device=device, dtype=dtype)

    if method_name == "heun":
        for i in range(steps - 1):
            dt = dts[i]
            t_curr = ts[i].expand(batch)
            x_req = x.detach().requires_grad_(True)
            drift_curr = drift_pf(model, x_req, t_curr, config)         
            div_curr = divergence_exact(drift_curr, x_req)            

            drift_curr_det = drift_curr.detach()
            div_curr_det = div_curr.detach()

            t_next = ts[i + 1].expand(batch)
            x_pred = x + dt * drift_curr_det

            x_req_next = x_pred.detach().requires_grad_(True)

            drift_next = drift_pf(model, x_req_next, t_next, config)   
            div_next = divergence_exact(drift_next, x_req_next) 

            drift_next_det = drift_next.detach()
            div_next_det = div_next.detach()

            x = x + 0.5 * dt * (drift_curr_det + drift_next_det)
            accum = accum + 0.5 * dt * (div_curr_det + div_next_det)

        return terminal_logp_gaussian(x, config) + accum

    if method_name == "euler":
        x = x0.detach()
        accum = torch.zeros(batch, device=device, dtype=dtype)

        for i in range(steps - 1):
            dt = dts[i]
            t_curr = ts[i].expand(batch)
            x_req = x.detach().requires_grad_(True)
            drift_curr = drift_pf(model, x_req, t_curr, config)
            div_curr = divergence_exact(drift_curr, x_req)
            drift_det = drift_curr.detach()
            div_det = div_curr.detach()
            x = x_req.detach() + dt * drift_det
            accum = accum + dt * div_det

        return terminal_logp_gaussian(x, config) + accum

    if method_name == "rk45":
        x = x0.detach()
        accum = torch.zeros(batch, device=device, dtype=dtype)
        for i in range(steps - 1):
            dt = dts[i]
            half_dt = 0.5 * dt
            t_curr = ts[i].expand(batch)
            x1 = x.detach().requires_grad_(True)
            drift1 = drift_pf(model, x1, t_curr, config)
            div1 = divergence_exact(drift1, x1)
            k1 = drift1.detach()
            div1_det = div1.detach()

            t_half = (ts[i] + half_dt).expand(batch)
            x2 = x1.detach() + half_dt * k1
            x2_req = x2.detach().requires_grad_(True)
            
            drift2 = drift_pf(model, x2_req, t_half, config)
            div2 = divergence_exact(drift2, x2_req)
            k2 = drift2.detach()
            div2_det = div2.detach()

            x3 = x1.detach() + half_dt * k2
            x3_req = x3.detach().requires_grad_(True)
            drift3 = drift_pf(model, x3_req, t_half, config)
            div3 = divergence_exact(drift3, x3_req)
            k3 = drift3.detach()
            div3_det = div3.detach()

            t_next = ts[i + 1].expand(batch)
            x4 = x1.detach() + dt * k3
            x4_req = x4.detach().requires_grad_(True)
            drift4 = drift_pf(model, x4_req, t_next, config)
            div4 = divergence_exact(drift4, x4_req)
            k4 = drift4.detach()
            div4_det = div4.detach()

            x = x1.detach() + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            accum = accum + (dt / 6.0) * (div1_det + 2.0 * div2_det + 2.0 * div3_det + div4_det)

        return terminal_logp_gaussian(x, config) + accum

    raise ValueError(
        f"Unsupported integration method '{method}'. Expected one of {{'Heun', 'Euler', 'RK45'}}."
    )

def evaluation_nll(model, data_loader, device, config: Dict) -> float:
    steps = int(config.get("nll_eval_steps", config.get("nll_steps", 1200)))
    eps = float(config.get("nll_eval_eps", config.get("eps", 1e-3)))
    prev_mode = model.training
    model.eval()
    total_nll = 0.0
    total_items = 0
    params = tuple(model.parameters())
    requires_grad_flags = tuple(p.requires_grad for p in params)
    try:
        for param in params:
            param.requires_grad_(False)
        with torch.no_grad():
            for (xb,) in data_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.enable_grad():
                    log_likelihood = integrate_log_probability(
                        model,
                        xb,
                        config,
                        steps=steps,
                        eps=eps,
                        method="Heun",
                    )
                total_nll += float((-log_likelihood).sum().item())
                total_items += log_likelihood.numel()
    finally:
        for param, flag in zip(params, requires_grad_flags):
            param.requires_grad_(flag)

        if prev_mode:
            model.train()
    
    if total_items == 0:
        return float("inf")
    return float(total_nll / total_items)

def evaluation_loss(model, data_loader, device, config: Dict) -> float:
    prev_mode = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for (xb,) in data_loader:
            xb = xb.to(device, non_blocking=True)
            loss = dsm_loss_batch(model, xb, config)
            total_loss += float(loss.item())
            total_batches += 1

    if prev_mode:
        model.train()

    if total_batches == 0:
        return float("inf")
    return float(total_loss / total_batches)
