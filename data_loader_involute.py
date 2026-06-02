from typing import Dict
import numpy as np
import torch
import torch.utils.data as tud
from config import config as global_config
from util import set_seed


def involute_params(config: Dict) -> tuple[float, float, float]:
    r_min = float(config.get("r_min", 0.0))
    r_max = float(config.get("r_max", 2.0 * np.pi))
    noise_std = float(config.get("noise_std", 0.4))
    return r_min, r_max, noise_std


def sample_involute(
    n: int,
    r_min: float,
    r_max: float,
    noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    r = rng.uniform(r_min, r_max, size=n).astype(np.float32)
    base = np.empty((n, 2), dtype=np.float32)
    base[:, 0] = r * np.sin(2.0 * r)
    base[:, 1] = r * np.cos(2.0 * r)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n, 2)).astype(np.float32)
    return base + noise


def generate_train_val(config: Dict, seed: int | None = None):
    seed_val = int(seed if seed is not None else config.get("seed", 42))
    set_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    r_min, r_max, noise_std = involute_params(config)
    train = sample_involute(config["n_train"], r_min, r_max, noise_std, rng=rng)
    val = sample_involute(config["n_val"], r_min, r_max, noise_std, rng=rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    seed_val = int(seed if seed is not None else config.get("seed", 42))
    set_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    r_min, r_max, noise_std = involute_params(config)
    return sample_involute(config["n_test"], r_min, r_max, noise_std, rng=rng)


def build_train_loader(train_tensor: torch.Tensor, batch_size: int, seed: int | None = None) -> tud.DataLoader:
    generator = torch.Generator()
    seed_val = int(seed if seed is not None else global_config.get("seed", 42))
    generator.manual_seed(seed_val)

    dataset = tud.TensorDataset(train_tensor)
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        generator=generator,
    )


def build_val_loader(val_tensor: torch.Tensor, batch_size: int, seed: int | None = None) -> tud.DataLoader:
    generator = torch.Generator()
    seed_val = int(seed if seed is not None else global_config.get("seed", 42))
    generator.manual_seed(seed_val)

    dataset = tud.TensorDataset(val_tensor)
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        generator=generator,
    )
