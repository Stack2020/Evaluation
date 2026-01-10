from typing import Dict
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sde import dsm_loss_batch, forward_perturb, v_to_score
from util import evaluation_loss

def build_scheduler_after_reset(optimizer, config, remaining_epochs: int):
    eta_min = float(config.get("lr_min", 1e-6))          # 比 0.1*lr 更激进
    T_max_after = int(config.get("T_max_after_reset", min(30, max(1, remaining_epochs))))
    return CosineAnnealingLR(optimizer, T_max=max(1, T_max_after), eta_min=eta_min)

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # EMA: shadow = d * shadow + (1 - d) * param
            self.shadow[name] = d * self.shadow[name] + (1.0 - d) * param.detach()

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.copy_(self.backup[name])
        self.backup = {}

    def ema_state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        # 用当前 state_dict 的结构，替换其中的可学习参数为 EMA 权重
        sd = model.state_dict()
        for name, avg in self.shadow.items():
            if name in sd:
                sd[name] = avg.detach().clone()
        return sd


def train_model(
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    train_loader,
    val_loader,
    checkpoint_path: str,
) -> None:
    curl_flag = config.get("curl", False)

    def lambda_curl(t: torch.Tensor) -> torch.Tensor:
        """Time-dependent curl regularization weight."""
        lambda_max=config.get("lambda_max", 0.01)
        t_cutoff = config.get("t_cutoff", 0.01)     
        k =config.get("k", 0.01)     
        return lambda_max* torch.sigmoid(-k * (t-t_cutoff))

    if curl_flag:
        print(f"Using curl regularization schedule")
    # ----- Optimizer -----
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    # ----- LR Scheduler: Linear warmup + Cosine -----
    total_epochs = int(config["epochs"])
    warmup_epochs = min(5, total_epochs // 10 if total_epochs >= 10 else 1)

    # 注意：PyTorch 的 LinearLR 没有 end_factor 参数，这里只用 start_factor/total_iters
    warm = LinearLR(
        optimizer,
        start_factor=float(config.get("warmup_start_factor", 0.1)),
        total_iters=warmup_epochs,
    )

    eta_min = float(config.get("lr_min", 1e-5))  # 比 0.1*lr 更激进，利于跳出 plateau
    
    cos = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=eta_min,
    )

    scheduler = SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])

    # ----- Early stop / plateau 控制 -----
    best_metric = float("inf")

    patience = int(config.get("degradation_patience",60))
    min_delta = float(config.get("min_improve_delta", 0.0))
    target_loss = float(config.get("target_loss", 0.1))
    epochs_no_improve = 0
    degraded = False

    # ----- EMA -----
    use_ema = bool(config.get("use_ema", True))
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_start_step = int(config.get("ema_start", 0))  # 开始累计 EMA 的 step
    ema = EMA(model, decay=ema_decay) if use_ema else None
    global_step = 0

    # ----- 训练 -----
    for epoch in range(1, total_epochs + 1):
        model.train()
        running = 0.0
        num_batches = len(train_loader)

        for batch_idx, (xb,) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)

            loss = dsm_loss_batch(model, xb, config)

            if curl_flag:
                eps = float(config.get("eps", 1e-3))
                gamma = float(config.get("loss_time_gamma", 1.0))
                u = torch.rand(xb.shape[0], device=xb.device, dtype=xb.dtype)
                if abs(gamma - 1.0) < 1e-6:
                    t = eps + (1.0 - eps) * u
                else:
                    t = eps + (1.0 - eps) * torch.pow(u, gamma)
                xt, _, _ = forward_perturb(xb, t, config["beta_min"], config["beta_max"])
                xt = xt.detach().requires_grad_(True)
                with torch.enable_grad():
                    score = v_to_score(model(xt, t), xt, t, config)
                    jac_rows = []
                    for comp in range(score.shape[1]):
                        grad = torch.autograd.grad(
                            outputs=score[:, comp].sum(),
                            inputs=xt,
                            retain_graph=True,
                            create_graph=True,
                            allow_unused=False,
                        )[0]
                        jac_rows.append(grad.unsqueeze(1))
                    jac = torch.cat(jac_rows, dim=1)
                    curl_pen_batch = ((jac - jac.transpose(1, 2)) ** 2).sum(dim=(1, 2))
                    curl_pen = (lambda_curl(t) * curl_pen_batch).mean()
                    loss = loss + curl_pen

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

        # ----- Validation（EMA + eval + no_grad）-----
        model_was_training = model.training

        if ema is not None:
            ema.apply_shadow(model)

        with torch.no_grad():
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
                    "best_metric": best_metric
                },
                checkpoint_path,
            )
            epochs_no_improve = 0
            degraded = False
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                degraded = True

        # ====== 平台触发：手动降一档 LR 并重置调度器 ======
        if degraded:
            drop_factor = float(config.get("manual_drop_factor", 0.3))  # 一般取 0.2~0.5
            lr_min = float(config.get("lr_min", 1e-6))
            for g in optimizer.param_groups:
                new_lr = max(float(g["lr"]) * drop_factor, lr_min)
                g["lr"] = new_lr
            # 用剩余 epoch 重新构造更短周期的余弦调度器
            remaining_epochs = max(1, int(config["epochs"]) - epoch)
            scheduler = build_scheduler_after_reset(optimizer, config, remaining_epochs)
            # 复位计数器，继续训练
            epochs_no_improve = 0
            degraded = False
            print(f"[Plateau fix] LR x{drop_factor:.2f} -> {optimizer.param_groups[0]['lr']:.2e}; "
                  f"reset scheduler with T_max={scheduler.T_max}.")
        # ----- Step scheduler（按 epoch）-----

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val_loss {val_metric:.4f} "
            f"| best_loss {best_metric:.4f} | lr {curr_lr:.2e} "
            f"| degraded {degraded} ({epochs_no_improve}/{patience})"
        )

        # ----- 目标提前停止（仅 loss 模式）-----
        if val_metric <= target_loss:
            print(f"Target reached: val_loss <= {target_loss:.4f}. Stopping early at epoch {epoch}.")
            break

    print(f"Best checkpoint: {checkpoint_path} best_loss={best_metric:.4f}")