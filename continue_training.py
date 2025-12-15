import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, _LRScheduler

from config import config, get_device
from data_loader import build_train_loader, build_val_loader, generate_train_val
from model import build_model
from sde import dsm_loss_batch
from train import EMA, build_scheduler_after_reset
from util import evaluation_loss, evaluation_nll, set_seed

def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict):
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    total_epochs = int(cfg["epochs"])
    warmup_epochs = min(5, total_epochs // 10 if total_epochs >= 10 else 1)

    warm = LinearLR(
        optimizer,
        start_factor=float(cfg.get("warmup_start_factor", 0.1)),
        total_iters=warmup_epochs,
    )

    eta_min = float(cfg.get("lr_min", 1e-5))
    cos = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=eta_min,
    )

    scheduler = SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])
    return optimizer, scheduler


def load_checkpoint_states(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, float, str, _LRScheduler]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    restored_scheduler: _LRScheduler = scheduler
    if "scheduler" in ckpt:
        sched_state = ckpt["scheduler"]
        try:
            scheduler.load_state_dict(sched_state)
            restored_scheduler = scheduler
        except Exception as e:
            if isinstance(sched_state, dict) and "_schedulers" not in sched_state and "T_max" in sched_state:
                # Backward compatibility: checkpoint saved after manual reset uses a plain CosineAnnealingLR.
                
                fallback = CosineAnnealingLR(
                    optimizer,
                    T_max=int(sched_state.get("T_max", 1)),
                    eta_min=float(sched_state.get("eta_min", 0.0)),
                )
                fallback.load_state_dict(sched_state)
                restored_scheduler = fallback
                print("Loaded legacy cosine scheduler state from checkpoint.")
            else:
                print(f"Warning: scheduler state in {checkpoint_path} incompatible ({e}); using freshly initialized scheduler.")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("inf")))
    metric_name = str(ckpt.get("metric_name", "loss"))
    return start_epoch, best_metric, metric_name, restored_scheduler


def continue_train(enable_nll: bool = False) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    train_np, val_np = generate_train_val(config, seed=seed)
    train_tensor = torch.from_numpy(train_np)
    val_tensor = torch.from_numpy(val_np)

    model = build_model(config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    checkpoint_path = Path(config["config_path"])
    start_epoch, best_metric, metric_name = 1, float("inf"), "nll" if enable_nll else "loss"
    if checkpoint_path.exists():
        start_epoch, best_metric, metric_name, scheduler = load_checkpoint_states(
            model, optimizer, scheduler, checkpoint_path, device
        )
        print(f"Loaded checkpoint from {checkpoint_path}, resume at epoch {start_epoch}, best_{metric_name}={best_metric:.4f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, training from scratch.")

    train_loader = build_train_loader(train_tensor, config["batch_size"], seed=seed)
    val_loader = build_val_loader(val_tensor, config["batch_size"], seed=seed)

    patience = int(config.get("degradation_patience", 100))
    min_delta = float(config.get("min_improve_delta", 0.0))
    target_loss = float(config.get("target_loss", 0.1))
    epochs_no_improve = 0
    degraded = False

    use_ema = bool(config.get("use_ema", True))
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_start_step = int(config.get("ema_start", 0))
    
    ema = EMA(model, decay=ema_decay) if use_ema else None
    global_step = 0

    total_epochs = int(config["epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running = 0.0
        num_batches = len(train_loader)

        for batch_idx, (xb,) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            loss = dsm_loss_batch(model, xb, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["clip_grad_norm"])
            )
            optimizer.step()
            global_step += 1

            if ema is not None and global_step >= ema_start_step:
                ema.update(model)

            running += float(loss.item())

            if batch_idx % 100 == 0 or batch_idx == num_batches:
                grad_norm_val = (
                    float(total_grad_norm.item())
                    if hasattr(total_grad_norm, "item")
                    else float(total_grad_norm)
                )
                print(
                    f"Epoch {epoch:03d} Step {batch_idx:04d}/{num_batches:04d} "
                    f"train_loss {loss.item():.4f} "
                    f"grad_norm {grad_norm_val:.4f} "
                )

        train_loss = running / num_batches

        model_was_training = model.training
        if ema is not None:
            ema.apply_shadow(model)

        with torch.no_grad():
            if enable_nll:
                val_metric = evaluation_nll(model, val_loader, device, config)
            else:
                val_metric = evaluation_loss(model, val_loader, device, config)

        if ema is not None:
            ema.restore(model)
        if model_was_training:
            model.train()

        if val_metric < best_metric - min_delta:
            best_metric = float(val_metric)
            to_save = ema.ema_state_dict(model) if ema is not None else model.state_dict()
            torch.save(
                {
                    "model": to_save,
                    "config": config,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "metric_name": metric_name,
                },
                checkpoint_path,
            )
            epochs_no_improve = 0
            degraded = False
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                degraded = True

        if degraded:
            drop_factor = float(config.get("manual_drop_factor", 0.3))
            lr_min = float(config.get("lr_min", 1e-6))
            reset_lr = float(config.get("reset_lr", config.get("lr", 1e-3)))
            for g in optimizer.param_groups:
                current = float(g["lr"])
                new_lr = max(current * drop_factor, reset_lr, lr_min)
                g["lr"] = new_lr
            remaining_epochs = max(1, int(config["epochs"]) - epoch)
            scheduler = build_scheduler_after_reset(optimizer, config, remaining_epochs)
            epochs_no_improve = 0
            degraded = False
            print(
                f"[Plateau fix] LR -> {optimizer.param_groups[0]['lr']:.2e} "
                f"(x{drop_factor:.2f}, reset floor {reset_lr:.2e}); "
                f"reset scheduler with T_max={scheduler.T_max}."
            )

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val_{metric_name} {val_metric:.4f} "
            f"| best_{metric_name} {best_metric:.4f} | lr {curr_lr:.2e} "
            f"| degraded {degraded} ({epochs_no_improve}/{patience})"
        )

        if not enable_nll and val_metric <= target_loss:
            print(f"Target reached: val_loss <= {target_loss:.4f}. Stopping early at epoch {epoch}.")
            break

    print(f"Best checkpoint: {checkpoint_path} best_{metric_name}={best_metric:.4f}")


def main() -> None:
    continue_train(enable_nll=False)


if __name__ == "__main__":
    main()
