import math
from pathlib import Path
import numpy as np
import torch
from config_regularization import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint
from sde import forward_perturb, v_to_score

def compute_rot_residual_batch(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    x_batch: torch.Tensor,
    t_val: float,
) -> np.ndarray:
    """Compute rotational residual at time t on noised samples x_t."""
    eps = float(cfg.get("eps", 1e-3))
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])

    x0 = x_batch.to(device=device, dtype=torch.float32)
    t_batch = torch.full((x0.shape[0],), float(t_val), device=device, dtype=x0.dtype)
    t_batch = torch.clamp(t_batch, min=eps, max=1.0)

    with torch.no_grad():
        x_t, _, _ = forward_perturb(x0, t_batch, beta_min, beta_max)

    x_req = x_t.detach().requires_grad_(True)

    with torch.enable_grad():
        score = v_to_score(model(x_req, t_batch), x_req, t_batch, cfg)
        jac_rows = []
        for comp in range(score.shape[1]):
            grad = torch.autograd.grad(
                outputs=score[:, comp].sum(),
                inputs=x_req,
                retain_graph=comp + 1 < score.shape[1],
                create_graph=False,
                allow_unused=False,
            )[0]
            jac_rows.append(grad.unsqueeze(1))
        jacobian = torch.cat(jac_rows, dim=1)
    diff = jacobian - jacobian.transpose(1, 2)
    fro_sq = (diff * diff).sum(dim=(1, 2))
    rot_residual = torch.sqrt(torch.clamp(fro_sq, min=0.0)) / math.sqrt(2.0)
    return rot_residual.detach().cpu().numpy()


def main() -> None:
    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(config["config_path"]), device)
    ckpt_cfg = checkpoint.get("config", {})
    merged_cfg = config.copy()
    merged_cfg.update(ckpt_cfg)
    merged_cfg["config_path"] = str(config["config_path"])

    # Base samples x0 drawn from the test set; evaluate residual directly on x0
    test_np = generate_test(merged_cfg, seed=int(merged_cfg.get("eval_seed_offset", 0)))
    test_tensor = torch.from_numpy(test_np).to(device=device, dtype=torch.float32)

    times = np.arange(0.01, 1.0 + 1e-9, 0.01, dtype=np.float32)
    batch_size = int(merged_cfg.get("score_batch_size", 512))

    model.eval()
    per_time_means: list[float] = []
    for t_val in times:
        rot_list = []
        for start in range(0, test_tensor.shape[0], batch_size):
            end = min(start + batch_size, test_tensor.shape[0])
            xb = test_tensor[start:end]
            rot = compute_rot_residual_batch(model, merged_cfg, device, xb, float(t_val))
            rot_list.append(rot)
        rot_vals = np.concatenate(rot_list, axis=0)
        per_time_means.append(float(rot_vals.mean()))

    per_time_means = np.asarray(per_time_means, dtype=np.float64)
    agg_value = float(per_time_means.mean())

    print("Mean R_rot per time (t, mean):")
    for t_val, mean_val in zip(times, per_time_means):
        print(f"  t={float(t_val):.2f}  mean={mean_val:.6e}")
    print(f"Aggregate mean over all times: {agg_value:.6e}")

    ckpt_path = Path(merged_cfg["config_path"])
    prefix = ckpt_path.stem
    output_npz = ckpt_path.with_name(f"{prefix}_mean_rot_series.npz")
    np.savez(
        output_npz,
        times=times,
        mean_rot=per_time_means,
        agg_mean=agg_value,
    )

    print(f"Times: {times.shape[0]} points from 0.01 to 1.00 (step 0.01)")
    print(f"Test samples (spatial probes): {test_np.shape[0]}")
    print(f"Mean rotational residual over test x_t and time: {agg_value:.6e}")
    print(f"Saved mean R_rot time series to {output_npz}")


if __name__ == "__main__":
    main()
