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


def diffusion_coeff(t: torch.Tensor, config: Dict) -> torch.Tensor:
    """Diffusion coefficient g(t) for the configured SDE."""
    beta = beta_t(t, config["beta_min"], config["beta_max"]).to(t).clamp_min(1e-8)
    sde_type = config.get("sde_type", "vp").lower()
    if sde_type == "subvp":
        lambda_t = integral_beta(t, config["beta_min"], config["beta_max"])
        noise_scale = 1.0 - torch.exp(-2.0 * lambda_t)
        return torch.sqrt(beta * torch.clamp(noise_scale, min=0.0))
    return torch.sqrt(beta)

@torch.no_grad()
def forward_perturb(
    x0: torch.Tensor,
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
    sde_type: str = "subvp",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward perturbation kernel for SDE.
    
    For VP SDE: std = sqrt(1 - α̅(t))
    For Sub-VP SDE: std = sqrt(1 - exp(-2∫β))
    """
    lambda_t = integral_beta(t, beta_min, beta_max).unsqueeze(-1)
    a = torch.exp(-lambda_t)  # α̅(t)
    mean = torch.sqrt(a) * x0
    
    if sde_type.lower() == "subvp":
        # Sub-VP: std = sqrt(1 - exp(-2∫β))
        std = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * lambda_t), min=1e-12))
    else:
        # VP: std = sqrt(1 - α̅(t))
        std = torch.sqrt(torch.clamp(1.0 - a, min=1e-12))
    
    eps = torch.randn_like(x0)
    xt = mean + std * eps
    return xt, mean, std


def dsm_loss_batch(model, x0: torch.Tensor, config: Dict) -> torch.Tensor:
    """Denoising Score Matching loss for Sub-VP SDE.
    
    The weight function should match the forward variance:
    - VP SDE: weight = (1 - α̅(t))^γ
    - Sub-VP SDE: weight = (1 - exp(-2∫β))^γ = var^γ
    """
    beta_min, beta_max = config["beta_min"], config["beta_max"]
    eps = config["eps"]
    gamma = config.get("lambda_gamma", 1.0)
    sde_type = config.get("sde_type", "subvp")
    batch = x0.shape[0]
    t = torch.rand(batch, device=x0.device) * (1.0 - eps) + eps
    xt, mean, std = forward_perturb(x0, t, beta_min, beta_max, sde_type)
    var = torch.clamp(std**2, min=1e-12)
    target = -(xt - mean) / var
    pred = model(xt, t)
    mse = F.mse_loss(pred, target, reduction="none").sum(dim=1)
    
    # Use variance as weight for Sub-VP SDE
    if sde_type.lower() == "subvp":
        weight = var.squeeze(-1).detach() ** gamma
    else:
        # VP SDE: use (1 - α̅(t))
        weight = (1.0 - alpha_bar(t, beta_min, beta_max)).detach() ** gamma
    
    return (weight * mse).mean()


def prior_logp_standard_normal(z: torch.Tensor) -> torch.Tensor:
    d = z.shape[1]
    return -0.5 * (z.pow(2).sum(dim=1) + d * math.log(2 * math.pi))


def terminal_logp_gaussian(x: torch.Tensor, config: Dict) -> torch.Tensor:
    """Exact log density of x at t=1 for the sub-VP forward SDE.
    
    For sub-VP SDE with initial distribution N(μ, Σ), at t=1:
    x₁ ~ N(sqrt(α̅(1)) * μ, α̅(1) * Σ + (1 - exp(-2∫β)) * I)
    """
    device = x.device
    dtype = x.dtype
    sde_type = config.get("sde_type", "subvp").lower()
    if sde_type != "subvp":
        raise ValueError("terminal_logp_gaussian now supports only sub-VP SDE.")
    
    dim = x.shape[1]
    eye = torch.eye(dim, device=device, dtype=dtype)
    
    # Get parameters
    beta_min = config["beta_min"]
    beta_max = config["beta_max"]
    mu = torch.tensor(config["mu"], device=device, dtype=dtype)
    sigma = torch.tensor(config["sigma"], device=device, dtype=dtype)
    
    # Compute α̅(1) = exp(-∫₀¹ β(s)ds)
    t_terminal = torch.tensor(1.0, device=device, dtype=dtype)
    lambda_1 = integral_beta(t_terminal, beta_min, beta_max)
    alpha_bar_1 = alpha_bar(t_terminal, beta_min, beta_max)
    
    # Compute noise variance: 1 - exp(-2∫₀¹ β(s)ds)
    noise_var = 1.0 - torch.exp(-2.0 * lambda_1)
    
    # Terminal mean: sqrt(α̅(1)) * μ
    mean_1 = torch.sqrt(alpha_bar_1) * mu
    
    # Terminal covariance: α̅(1) * Σ + noise_var * I
    cov_1 = alpha_bar_1 * sigma + noise_var * eye
    
    # Compute log probability
    cov_inv = torch.inverse(cov_1)
    diff = x - mean_1
    exponent = (diff @ cov_inv * diff).sum(dim=1)
    log_det = torch.logdet(cov_1)
    
    return -0.5 * (exponent + log_det + dim * math.log(2.0 * math.pi))


#PF-ODE计算
def drift_pf(model, x: torch.Tensor, t: torch.Tensor, config: Dict) -> torch.Tensor:
    beta = beta_t(
        t,
        config["beta_min"],
        config["beta_max"],
    ).clamp_min(1e-8).reshape(-1, 1)
    score = model(x, t)
    diffusion = diffusion_coeff(t, config).reshape(-1, 1)
    diffusion_sq = diffusion * diffusion
    return (-0.5 * beta) * x - 0.5 * diffusion_sq * score


def divergence_exact(v: torch.Tensor, x_req: torch.Tensor) -> torch.Tensor:
    _, dim = x_req.shape
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





@torch.no_grad()
def probability_flow_sample(
    model,
    config: Dict,
    device: torch.device,
    n_samples: int,
    steps: int | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """Sample from the model using probability flow ODE.
    
    Integrate backward from t=1 (noise) to t=eps (data).
    """
    steps = config["num_steps"] if steps is None else steps
    eps = config["eps"] if eps is None else eps

    x = torch.randn(n_samples, 2, device=device)
    ts = torch.linspace(1.0, eps, steps, device=device)

    for i in range(steps - 1):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t  # negative, since we go from 1.0 to eps
        t_curr = t.expand(n_samples)

        drift_t = drift_pf(model, x, t_curr, config)
        x_pred = x + dt * drift_t

        t_next_expand = t_next.expand(n_samples)
        drift_next = drift_pf(model, x_pred, t_next_expand, config)
        x = x + 0.5 * dt * (drift_t + drift_next)

    return x
