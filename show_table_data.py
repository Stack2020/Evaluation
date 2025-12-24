from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
from config import config, get_device
from miniboone_data_loader  import load_miniboone_array, split_dataset, z_standardize
from model import build_model
from util import integrate_log_probability


def evaluate_model_log_density(
    model,
    cfg: dict,
    device: torch.device,
    points: np.ndarray,
    *,
    steps: int,
    eps: float,
    method: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate model log-density on arbitrary points (dimension inferred from cfg)."""
    model.eval()
    outputs: list[np.ndarray] = []
    total = points.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = torch.from_numpy(points[start:end]).to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                xb,
                cfg,
                steps=steps,
                eps=eps,
                method=method,
            )
        outputs.append(logp.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    # Load checkpoint config first so model dims (e.g., t_embed_dim) match
    checkpoint_path = str(config["config_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = checkpoint.get("config", {})
    cfg = config.copy()
    cfg.update(ckpt_cfg)
    cfg["config_path"] = checkpoint_path

    device = get_device()
    # Load gas data, split, and standardize using train statistics
    miniboone_path = cfg.get("miniboone_path", "./data/miniboone.npy")

    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    seed = int(cfg.get("seed", 42))

    data = load_miniboone_array(miniboone_path)

    train_raw, val_raw, test_raw = split_dataset(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_z, val_z, test_z, mean, std = z_standardize(train_raw, val_raw, test_raw)

    data_dim = int(test_raw.shape[1]) if test_raw.ndim > 1 else 1
    if int(cfg.get("data_dim", data_dim)) != data_dim:
        print(f"Updating data_dim from {cfg.get('data_dim')} to {data_dim}")
    cfg["data_dim"] = data_dim

    # Build model with merged cfg, then load weights
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model"])

    batch_size = int(cfg.get("score_batch_size", 512))
    steps_list: Iterable[int] = (50, 100, 150, 200, 250, 300)

    method = str(cfg.get("density_method", "Heun"))
    eps_val = float(cfg.get("nll_eval_eps", cfg.get("eps", 1e-3)))

    # log|det J| for x = z * std + mean where J = diag(std) => p_X = p_Z / prod(std)
    log_std_sum = float(np.log(np.abs(std)).sum())

    print(f"Test samples: {test_raw.shape[0]} (dim={data_dim})")
    print("NFE\tNLL_raw\tNLL_z\tJac_correction")

    for steps in steps_list:
        model_logp_z = evaluate_model_log_density(
            model,
            cfg,
            device,
            test_z,
            steps=steps,
            eps=eps_val,
            method=method,
            batch_size=batch_size,
        )

        # PF-ODE NLL in standardized (z) space
        nll_z = float(-model_logp_z.mean())
        # Transform back to raw space: log p_X(x) = log p_Z(z) - log|det J|
        nll_raw = float(-(model_logp_z - log_std_sum).mean())

        print(
            f"{steps}\t"
            f"{nll_raw:.8f}\t"
            f"{nll_z:.8f}\t"
            f"{log_std_sum:.8f}"
        )


if __name__ == "__main__":
    main()
