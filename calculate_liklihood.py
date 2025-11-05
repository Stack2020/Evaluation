from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch

from data_loader import generate_test
from util import integrate_log_probability

def calculate_liklihood(
    model,
    config: Dict,
    device: torch.device,
    *,
    steps: Optional[int] = None,
    eps: Optional[float] = None,
    batch_size: int = 128,
    method: str = "Heun",
    test_tensor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    allowed_methods = {"euler", "heun", "rk4"}
    method = method.lower()
    if method not in allowed_methods:
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of {{'Euler', 'Heun', 'RK4'}}."
        )
    print(steps)
    eps = float(eps or config.get("nll_eval_eps", config.get("eps", 1e-3)))

    if test_tensor is None:
        test_np = generate_test(config)
        test_tensor = torch.from_numpy(test_np)
    test_tensor = test_tensor.to(device)

    prev_mode = model.training
    model.eval()

    params = tuple(model.parameters())
    requires_grad_flags = tuple(p.requires_grad for p in params)
    for param in params:
        param.requires_grad_(False)

    log_probs = []
    try:
        for start in range(0, test_tensor.shape[0], batch_size):
            xb = test_tensor[start : start + batch_size]
            with torch.enable_grad():
                log_px0 = integrate_log_probability(
                    model,
                    xb,
                    config,
                    steps=steps,
                    eps=eps,
                    method=method,
                )
            log_probs.append(log_px0.detach().cpu())
    finally:
        for param, flag in zip(params, requires_grad_flags):
            param.requires_grad_(flag)
        if prev_mode:
            model.train()

    log_probs_tensor = torch.cat(log_probs, dim=0)
    mean_nll = float(-log_probs_tensor.mean().item())

    return log_probs_tensor, mean_nll


__all__ = ["calculate_liklihood"]
