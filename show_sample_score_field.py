import math
import numpy as np
import torch
from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint
from sde import v_to_score

def compute_rot_residual_batch(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    x_batch: torch.Tensor,
    t_val: float,
) -> np.ndarray:
    """Compute rotational residual at time t using base samples x0."""
    eps = float(cfg.get("eps", 1e-3))
    x0 = x_batch.to(device=device, dtype=torch.float32)
    t_batch = torch.full((x0.shape[0],), float(t_val), device=device, dtype=x0.dtype)
    t_batch = torch.clamp(t_batch, min=eps, max=1.0)

    x_req = x0.detach().requires_grad_(True)

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
    per_time_expectations: list[float] = []
    for t_val in times:
        rot_sq_list = []
        for start in range(0, test_tensor.shape[0], batch_size):
            end = min(start + batch_size, test_tensor.shape[0])
            xb = test_tensor[start:end]
            rot = compute_rot_residual_batch(model, merged_cfg, device, xb, float(t_val))
            rot_sq_list.append(rot * rot)
        rot_sq = np.concatenate(rot_sq_list, axis=0)
        
        per_time_expectations.append(float(rot_sq.mean()))

    per_time_expectations = np.asarray(per_time_expectations, dtype=np.float64)
    agg_value = math.sqrt(per_time_expectations.mean())

    print(f"Times: {times.shape[0]} points from 0.01 to 1.00 (step 0.01)")
    print(f"Test samples (spatial probes): {test_np.shape[0]}")
    print(f"RMS rotational residual (test points across t): {agg_value:.6e}")


if __name__ == "__main__":
    main()
