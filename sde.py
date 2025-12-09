import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def match_shape(coeff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast helper to align coeff dims with target."""
    while coeff.dim() < target.dim():
        coeff = coeff.unsqueeze(-1)
    return coeff


def vp_coefficients(t: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return α(t), σ(t), α̅(t) for the VP-SDE."""
    a_bar = alpha_bar(t, config["beta_min"], config["beta_max"])
    alpha = torch.sqrt(a_bar)
    sigma = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
    return alpha, sigma, a_bar


def v_to_x0(v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, config: Dict) -> torch.Tensor:
    alpha, sigma, _ = vp_coefficients(t, config)
    alpha = match_shape(alpha, x_t)
    sigma = match_shape(sigma, x_t)
    return alpha * x_t - sigma * v_pred


def v_to_epsilon(v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, config: Dict) -> torch.Tensor:
    alpha, sigma, _ = vp_coefficients(t, config)
    alpha = match_shape(alpha, x_t)
    sigma = match_shape(sigma, x_t)
    return sigma * x_t + alpha * v_pred


def v_to_score(v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, config: Dict) -> torch.Tensor:
    _, sigma, _ = vp_coefficients(t, config)
    sigma = match_shape(sigma, x_t)
    eps = v_to_epsilon(v_pred, x_t, t, config)
    return -eps / torch.clamp(sigma, min=1e-12)

def beta_t(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


def integral_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min * t + 0.5 * (beta_max - beta_min) * t * t


def alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.exp(-integral_beta(t, beta_min, beta_max))


def diffusion_coeff(t: torch.Tensor, config: Dict) -> torch.Tensor:
    """Diffusion coefficient g(t) for VP-SDE: g(t) = sqrt(beta(t))."""
    beta = beta_t(t, config["beta_min"], config["beta_max"]).to(t).clamp_min(1e-8)
    return torch.sqrt(beta)

@torch.no_grad()
def forward_perturb(
    x0: torch.Tensor,
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward perturbation kernel for VP-SDE.
    
    VP-SDE: x_t = sqrt(α̅(t)) x_0 + sqrt(1 - α̅(t)) ε
    where α̅(t) = exp(-∫₀ᵗ β(s) ds)
    """
    lambda_t = integral_beta(t, beta_min, beta_max).unsqueeze(-1)
    alpha_bar = torch.exp(-lambda_t)  # α̅(t)
    mean = torch.sqrt(alpha_bar) * x0
    std = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-12))
    eps = torch.randn_like(x0)
    xt = mean + std * eps
    return xt, mean, std


def dsm_loss_batch(model, x0: torch.Tensor, config: Dict) -> torch.Tensor:
    """v-prediction loss for vp."""
    beta_min, beta_max = config["beta_min"], config["beta_max"]
    batch = x0.shape[0]
    eps = float(config.get("eps", 1e-3))
    gamma = float(config.get("loss_time_gamma", 1.0))
    u = torch.rand(batch, device=x0.device)
    if abs(gamma - 1.0) < 1e-6:
        t = eps + (1.0 - eps) * u
        weight = None
    else:
        t = eps + (1.0 - eps) * torch.pow(u, gamma)
        if config.get("loss_time_reweight", False):
            scaled = torch.clamp((t - eps) / (1.0 - eps), min=1e-12, max=1.0)
            exponent = 1.0 - 1.0 / max(gamma, 1e-6)
            weight = gamma * torch.pow(scaled, exponent)
            max_w = float(config.get("loss_time_weight_max", 0.0))
            if max_w > 0.0:
                weight = torch.clamp(weight, max=max_w)
        else:
            weight = None

    lambda_t = integral_beta(t, beta_min, beta_max).unsqueeze(-1)
    alpha_bar = torch.exp(-lambda_t)
    alpha = torch.sqrt(alpha_bar)

    sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-12))
    eps_true = torch.randn_like(x0)

    xt = alpha * x0 + sigma * eps_true

    v_true = alpha * eps_true - sigma * x0
    
    v_pred = model(xt, t)

    mse = F.mse_loss(v_pred, v_true, reduction="none").sum(dim=1)
    if weight is not None:
        mse = mse * weight
    return mse.mean()


def prior_logp_standard_normal(z: torch.Tensor) -> torch.Tensor:
    d = z.shape[1]
    return -0.5 * (z.pow(2).sum(dim=1) + d * math.log(2 * math.pi))


def terminal_logp_gaussian(x: torch.Tensor, config: Dict) -> torch.Tensor:
    """Exact log density of x at t=1 for the VP forward SDE.
    
    If x0 ~ N(μ, Σ), then for VP-SDE at t=1:
    x₁ ~ N(sqrt(α̅(1)) * μ, α̅(1) * Σ + (1 - α̅(1)) * I)
    """
    device = x.device
    dtype = x.dtype

    dim = x.shape[1]
    eye = torch.eye(dim, device=device, dtype=dtype)

    # Parameters
    beta_min = config["beta_min"]
    beta_max = config["beta_max"]
    mu = torch.tensor(config["mu"], device=device, dtype=dtype)
    sigma = torch.tensor(config["sigma"], device=device, dtype=dtype)

    # α̅(1) = exp(-∫₀¹ β(s) ds)
    t_terminal = torch.tensor(1.0, device=device, dtype=dtype)
    alpha_bar_1 = alpha_bar(t_terminal, beta_min, beta_max)

    # VP noise variance: 1 - α̅(1)
    noise_var = 1.0 - alpha_bar_1

    # Terminal mean and covariance
    mean_1 = torch.sqrt(alpha_bar_1) * mu
    cov_1 = alpha_bar_1 * sigma + noise_var * eye

    # Log probability under N(mean_1, cov_1)
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
    score = v_to_score(model(x, t), x, t, config)
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
