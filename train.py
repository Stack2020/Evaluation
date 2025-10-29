from typing import Dict, Tuple
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from config import config_path
from sde import dsm_loss_batch
from util import evaluation_loss, evaluation_nll


def train_model(
    model,
    config: Dict,
    device: torch.device,
    train_loader,
    val_loader,
    checkpoint_path: str = config_path,
    enable_nll: bool = False,
) -> None:

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    warmup_epochs = min(5, config["epochs"] // 10 if config["epochs"] >= 10 else 1)
    warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    cos = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config["epochs"] - warmup_epochs),
        eta_min=config["lr"] * 0.1,
    )
    scheduler = SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])

    best_metric = float("inf")
    metric_name = "nll" if enable_nll else "loss"
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = 0.0

        num_batches = len(train_loader)
        for batch_idx, (xb,) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            loss = dsm_loss_batch(model, xb, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            running += loss.item()
            if batch_idx % 100 == 0 or batch_idx == num_batches:
                print(
                    f"Epoch {epoch:03d} Step {batch_idx:04d}/{num_batches:04d} "
                    f"train_loss {loss.item():.4f}"
                )

        train_loss = running / len(train_loader)

        if enable_nll:
            val_metric = evaluation_nll(model, val_loader, device, config)
        else:
            val_metric = evaluation_loss(model, val_loader, device, config)

        if val_metric < best_metric:
            best_metric = val_metric
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
            
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val_{metric_name} {val_metric:.4f} "
            f"| best_{metric_name} {best_metric:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )


    print(f"Best checkpoint: {checkpoint_path} best_{metric_name}={best_metric:.4f}")
