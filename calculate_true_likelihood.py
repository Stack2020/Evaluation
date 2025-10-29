from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
from data_loader import generate_test

def gaussian_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """Log-density of a multivariate Gaussian N(mean, cov)."""
    dim = mean.shape[0]
    diff = x - mean
    exponent = torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    log_det = torch.logdet(cov)
    return -0.5 * (exponent + log_det + dim * np.log(2.0 * np.pi))


def calculate_true_likelihood(
    config: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Return per-sample log-likelihoods and mean NLL using the true Gaussian."""
    mean_np = np.asarray(config["mu"], dtype=np.float64)
    cov_np = np.asarray(config["sigma"], dtype=np.float64)

    test_np = generate_test(config)
    test_tensor = torch.from_numpy(test_np).to(device=device, dtype=torch.float64)

    mean = torch.as_tensor(mean_np, device=device, dtype=test_tensor.dtype)
    cov = torch.as_tensor(cov_np, device=device, dtype=test_tensor.dtype)
    cov_inv = torch.inverse(cov)

    log_probs = gaussian_log_prob(test_tensor, mean, cov, cov_inv)
    log_probs_tensor = log_probs.to(dtype=torch.float32)
    mean_nll = float((-log_probs_tensor).mean().item())
    return log_probs_tensor, mean_nll


__all__ = ["calculate_true_likelihood"]
