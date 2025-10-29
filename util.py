from typing import Dict, Iterable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from sde import dsm_loss_batch, pf_ode_log_likelihood

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
                    log_likelihood = pf_ode_log_likelihood(
                        model,
                        xb,
                        config,
                        steps=steps,
                        eps=eps,
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
