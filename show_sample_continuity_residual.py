import math
import numpy as np
import torch

from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint
from sde import drift_pf, divergence_exact, v_to_score
from util import integrate_log_probability


def finite_difference_time(logp_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central/forward/backward finite difference over time for log p."""
    T, N = logp_times.shape
    dlog_dt = np.empty_like(logp_times, dtype=np.float64)
    for k in range(T):
        if k == 0:
            dt = times[1] - times[0]
            dlog_dt[k] = (logp_times[1] - logp_times[0]) / dt
        elif k == T - 1:
            dt = times[-1] - times[-2]
            dlog_dt[k] = (logp_times[-1] - logp_times[-2]) / dt
        else:
            dt = times[k + 1] - times[k - 1]
            dlog_dt[k] = (logp_times[k + 1] - logp_times[k - 1]) / dt
    return dlog_dt


def compute_logp_over_times(model, cfg, device, x0: np.ndarray, times: np.ndarray, batch_size: int) -> np.ndarray:
    """Compute model log p_t(x0) for each time t, using fixed spatial samples x0."""
    model.eval()
    num_times = times.shape[0]
    num_samples = x0.shape[0]
    logp_all = np.empty((num_times, num_samples), dtype=np.float32)
    clip_val = float(cfg.get("continuity_logp_clip", 0.0))

    for ti, t_val in enumerate(times):
        logs: list[np.ndarray] = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            xt = torch.from_numpy(x0[start:end]).to(device=device, dtype=torch.float32)
            with torch.enable_grad():
                logp = integrate_log_probability(
                    model,
                    xt,
                    cfg,
                    steps=int(cfg.get("sample_continuity_steps", 100)),
                    eps=float(t_val),
                    method=str(cfg.get("density_method", "Heun")),
                )
            logs.append(logp.detach().cpu().numpy())
        merged = np.concatenate(logs, axis=0)
        if clip_val > 0:
            merged = np.clip(merged, -clip_val, clip_val)
        logp_all[ti] = merged
        print(f"[logp] t={t_val:.3f} done")
    return logp_all


def compute_residual_for_time(
    model,
    cfg,
    device,
    x_fixed: np.ndarray,
    t_val: float,
    dlog_dt_slice: np.ndarray,
    batch_size: int,
) -> float:
    """Compute mean of R_cont^2 at a single time over samples."""
    num_samples = x_fixed.shape[0]
    accum = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        xb = torch.from_numpy(x_fixed[start:end]).to(device=device, dtype=torch.float32)
        t_batch = torch.full((xb.shape[0],), float(t_val), device=device, dtype=xb.dtype)
        t_batch = torch.clamp(t_batch, min=float(cfg.get("eps", 1e-3)), max=1.0)

        xt = xb.detach().requires_grad_(True)

        with torch.enable_grad():
            velocity = drift_pf(model, xt, t_batch, cfg)
            divergence = divergence_exact(velocity, xt)
            score = v_to_score(model(xt, t_batch), xt, t_batch, cfg)
        conv = (velocity * score).sum(dim=1)
        residual = dlog_dt_slice[start:end] + conv.detach().cpu().numpy() + divergence.detach().cpu().numpy()
        accum.append(residual * residual)
    all_sq = np.concatenate(accum, axis=0)
    return float(all_sq.mean())


def main() -> None:
    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(config["config_path"]), device)
    ckpt_cfg = checkpoint.get("config", {})
    cfg = config.copy()
    cfg.update(ckpt_cfg)
    cfg["config_path"] = str(config["config_path"])

    # Evaluate continuity residual on test-set samples
    x0_np = generate_test(cfg, seed=int(cfg.get("eval_seed_offset", 0))).astype(np.float32)
    times = np.arange(0.01, 1.0 + 1e-9, 0.01, dtype=np.float32)
    batch_size = int(cfg.get("score_batch_size", 512))

    # Step 1: log p_t(x_t) over times
    logp_times = compute_logp_over_times(model, cfg, device, x0_np, times, batch_size=batch_size)
    dlog_dt = finite_difference_time(logp_times.astype(np.float64), times.astype(np.float64))

    # Step 2: continuity residual per time using model drift/score/divergence
    per_time_means = []
    for ti, t_val in enumerate(times):
        mean_sq = compute_residual_for_time(
            model,
            cfg,
            device,
            x0_np,
            float(t_val),
            dlog_dt[ti],
            batch_size=batch_size,
        )
        per_time_means.append(mean_sq)
        print(f"[R_cont] t={t_val:.3f} mean(R^2)={mean_sq:.6e}")

    per_time_means = np.asarray(per_time_means, dtype=np.float64)
    agg_value = math.sqrt(per_time_means.mean())

    print(f"Times: {times.shape[0]} points from 0.01 to 1.00 (step 0.01)")
    print(f"Test samples: {x0_np.shape[0]}")
    print(f"Eq.(30) RMS continuity residual (test points): {agg_value:.6e}")


if __name__ == "__main__":
    main()
