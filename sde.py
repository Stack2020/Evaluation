import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

def beta_t(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


def integral_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min * t + 0.5 * (beta_max - beta_min) * t * t


def alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.exp(-integral_beta(t, beta_min, beta_max))

@torch.no_grad()
def forward_perturb(
    x0: torch.Tensor,
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = alpha_bar(t, beta_min, beta_max).unsqueeze(-1)
    mean = torch.sqrt(a) * x0
    std = torch.sqrt(torch.clamp(1.0 - a, min=1e-12))
    eps = torch.randn_like(x0)
    xt = mean + std * eps
    return xt, mean, std


def dsm_loss_batch(model, x0: torch.Tensor, config: Dict) -> torch.Tensor:
    beta_min, beta_max = config["beta_min"], config["beta_max"]
    eps = config["eps"]
    gamma = config.get("lambda_gamma", 1.0)
    batch = x0.shape[0]
    t = torch.rand(batch, device=x0.device) * (1.0 - eps) + eps
    xt, mean, std = forward_perturb(x0, t, beta_min, beta_max)
    var = torch.clamp(std**2, min=1e-12)
    target = -(xt - mean) / var
    pred = model(xt, t)
    mse = F.mse_loss(pred, target, reduction="none").sum(dim=1)
    weight = (1.0 - alpha_bar(t, beta_min, beta_max)).detach() ** gamma
    return (weight * mse).mean()


def prior_logp_standard_normal(z: torch.Tensor) -> torch.Tensor:
    d = z.shape[1]
    return -0.5 * (z.pow(2).sum(dim=1) + d * math.log(2 * math.pi))


def drift_pf(model, x: torch.Tensor, t: torch.Tensor, config: Dict) -> torch.Tensor:
    beta = beta_t(t, config["beta_min"], config["beta_max"]).clamp_min(1e-8).view(-1, 1)
    score = model(x, t)
    return (-0.5 * beta) * x - 0.5 * beta * score


def divergence_exact(v: torch.Tensor, x_req: torch.Tensor) -> torch.Tensor:
    batch, dim = x_req.shape
    terms = []
    for j in range(dim):
        grad_j = torch.autograd.grad(
            outputs=v[:, j].sum(),
            inputs=x_req,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        terms.append(grad_j[:, j])
    return torch.stack(terms, dim=1).sum(dim=1)


def pf_ode_log_likelihood(
    model,
    x0: torch.Tensor,
    config: Dict,
    steps: int = 1200,
    eps: float = 1e-3,
) -> torch.Tensor:
    device = x0.device
    dtype = x0.dtype
    batch = x0.shape[0]
    
    ts = torch.linspace(eps, 1.0, steps, device=device, dtype=dtype)
    dt = ts[1] - ts[0]   
    
    x = x0.detach()
    accum = torch.zeros(batch, device=device, dtype=dtype)
    
    t_curr = ts[0].expand(batch)
    x_req = x.requires_grad_(True)
    v_curr = drift_pf(model, x_req, t_curr, config)
    div_curr = divergence_exact(v_curr, x_req)
    
    for i in range(steps - 1):
        t_next = ts[i + 1].expand(batch)
        
        # 使用当前的v进行预测
        x_pred = x + dt * v_curr.detach()
        
        # 计算下一步的drift和divergence
        x_req_next = x_pred.requires_grad_(True)
        v_next = drift_pf(model, x_req_next, t_next, config)
        div_next = divergence_exact(v_next, x_req_next)
        
        # 梯形法则更新
        x = x + 0.5 * dt * (v_curr.detach() + v_next.detach())
        accum = accum - 0.5 * dt * (div_curr + div_next)
        
        # 复用：下一步的v_next变成当前的v_curr
        v_curr = v_next
        div_curr = div_next
    
    return prior_logp_standard_normal(x) + accum



@torch.no_grad()
def probability_flow_sample(
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
