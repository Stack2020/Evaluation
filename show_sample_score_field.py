import math
import numpy as np
import torch
from config import config, get_device
from miniboone_data_loader import load_miniboone_array, split_dataset, z_standardize
from model import build_model
from sde import forward_perturb, v_to_score


def load_miniboone_test_set(cfg: dict) -> tuple[np.ndarray, int, dict[str, np.ndarray | int]]:
    """
    Load and z-standardize the MiniBooNE test split, returning a deterministic subset
    for evaluating residuals.
    """
    data_path = cfg.get("miniboone_path", "./data/miniboone.npy")
    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    seed = int(cfg.get("seed", 42))

    data = load_miniboone_array(data_path)
    train_raw, val_raw, test_raw = split_dataset(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_z, _val_z, test_z, mean, std = z_standardize(train_raw, val_raw, test_raw)

    data_dim = int(test_raw.shape[1]) if test_raw.ndim > 1 else 1

    eval_points_cfg = cfg.get("residual_eval_points", cfg.get("density_eval_points"))
    eval_points = test_z.shape[0] if eval_points_cfg is None else int(eval_points_cfg)
    eval_points = max(1, min(eval_points, test_z.shape[0]))

    subset_seed = int(seed + int(cfg.get("eval_seed_offset", 0)))
    if eval_points < test_z.shape[0]:
        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(test_z.shape[0], size=eval_points, replace=False)
        test_eval = test_z[idx]
    else:
        test_eval = test_z

    stats = {
        "num_total": int(test_z.shape[0]),
        "num_eval": int(eval_points),
        "mean": mean,
        "std": std,
    }
    return test_eval.astype(np.float32, copy=False), data_dim, stats

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
    checkpoint_path = str(config["config_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", {})
    merged_cfg = config.copy()
    merged_cfg.update(ckpt_cfg)
    merged_cfg["config_path"] = checkpoint_path

    # MiniBooNE: standardize with train statistics, then take a deterministic eval subset.
    test_np, data_dim, data_stats = load_miniboone_test_set(merged_cfg)
    if int(merged_cfg.get("data_dim", data_dim)) != data_dim:
        print(f"Updating data_dim from {merged_cfg.get('data_dim')} to {data_dim}")
    merged_cfg["data_dim"] = data_dim

    model = build_model(merged_cfg, device)
    model.load_state_dict(checkpoint["model"])

    test_tensor = torch.from_numpy(test_np).to(device=device, dtype=torch.float32)

    eps = float(merged_cfg.get("eps", 1e-3))
    t_start = max(eps, 0.01)
    times = np.arange(t_start, 1.0 + 1e-9, 0.01, dtype=np.float32)
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
        mean_val = float(rot_vals.mean())
        per_time_means.append(mean_val)
        print(f"t={t_val:.4f} mean_rot_residual={mean_val:.6e}")

    per_time_means = np.asarray(per_time_means, dtype=np.float64)
    agg_value = float(per_time_means.mean())

    print(f"Times: {times.shape[0]} points from {times[0]:.4f} to {times[-1]:.4f} (step 0.01)")
    print(
        f"MiniBooNE test probes (z-standardized): {test_np.shape[0]} "
        f"(full test_z={data_stats['num_total']})"
    )
    print(f"Mean rotational residual over MiniBooNE x_t and time: {agg_value:.6e}")


if __name__ == "__main__":
    main()
