from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from config import config, get_device
from data_loader import build_train_loader, build_val_loader, generate_train_val
from model import build_model
from seed_utils import derive_seed, make_worker_init_fn, set_global_seed
from train import train_model


def format_checkpoint_path(base_path: str, seed: int) -> str:
    path = Path(base_path)
    suffix = path.suffix
    seed_name = f"{path.stem}_seed{seed}"
    return str(path.with_name(f"{seed_name}{suffix}"))


def iter_train_seeds(cfg: dict) -> Iterable[int]:
    seeds = cfg.get("train_seeds")
    if seeds:
        return [int(s) for s in seeds]
    base_seed = int(cfg.get("seed", 0))
    return [base_seed]


def train_one_seed(base_cfg: dict, device: torch.device, seed: int) -> None:
    run_cfg = dict(base_cfg)
    run_cfg["seed"] = seed
    deterministic = bool(run_cfg.get("deterministic", True))
    print(f"\n=== Training run for seed={seed} (deterministic={deterministic}) ===")

    torch_gen = set_global_seed(seed, deterministic=deterministic)

    data_seed = derive_seed(seed, int(run_cfg.get("data_seed_offset", 10_000)))

    train_np, val_np = generate_train_val(run_cfg, seed=data_seed)
    print("Train/Val shapes:", train_np.shape, val_np.shape)

    train_tensor = torch.from_numpy(train_np)
    val_tensor = torch.from_numpy(val_np)

    model = build_model(run_cfg, device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):.1f}")

    loader_seed = derive_seed(seed, int(run_cfg.get("loader_seed_offset", 20_000)))
    
    num_workers = int(run_cfg.get("loader_workers", 2))

    train_loader = build_train_loader(
        train_tensor,
        run_cfg["batch_size"],
        generator=torch_gen,
        worker_init_fn=make_worker_init_fn(loader_seed),
        num_workers=num_workers,
    )

    val_loader = build_val_loader(
        val_tensor,
        run_cfg["batch_size"],
        generator=torch_gen,
        worker_init_fn=make_worker_init_fn(loader_seed + 1),
        num_workers=num_workers,
    )

    checkpoint_path = format_checkpoint_path(run_cfg["config_path"], seed)
    
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    run_cfg["config_path"] = checkpoint_path

    train_model(
        model,
        run_cfg,
        device,
        train_loader,
        val_loader,
        checkpoint_path,
        enable_nll=False,
    )


def main() -> None:
    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    for seed in iter_train_seeds(config):
        train_one_seed(config, device, seed)


if __name__ == "__main__":
    main()
