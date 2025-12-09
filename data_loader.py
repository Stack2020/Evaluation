from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.utils.data as tud


def sample_gaussian(n: int, mu, sigma, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    mu_arr = np.asarray(mu, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return rng.multivariate_normal(mu_arr, sigma_arr, size=n).astype(np.float32)

def sample_involute(
    n: int,
    noise_scale: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    r = rng.uniform(0.0, 2.0 * np.pi, size=n)
    mu1 = r * np.sin(2.0 * r)
    mu2 = r * np.cos(2.0 * r)
    mu = np.stack((mu1, mu2), axis=1, dtype=np.float32)
    eps = rng.normal(loc=0.0, scale=1.0, size=(n, 2)).astype(np.float32)
    return mu + float(noise_scale) * eps


def draw_samples(config: Dict, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = str(config.get("data_distribution", "gaussian")).lower()
    if dist == "gaussian":
        return sample_gaussian(n, config["mu"], config["sigma"], rng=rng)
    if dist == "involute":
        noise_scale = float(config.get("noise_scale", 0.4))
        return sample_involute(n, noise_scale=noise_scale, rng=rng)
    raise ValueError(f"Unsupported data_distribution '{dist}'")


def generate_train_val(config: Dict, seed: int = None):
    rng = np.random.default_rng(seed)
    train = draw_samples(config, int(config["n_train"]), rng)
    val = draw_samples(config, int(config["n_val"]), rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return draw_samples(config, int(config["n_test"]), rng)



def build_train_loader(
    train_tensor: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    num_workers: int = 2,
) -> tud.DataLoader:

    dataset = tud.TensorDataset(train_tensor)

    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def build_val_loader(
    val_tensor: torch.Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    num_workers: int = 2,
) -> tud.DataLoader:
    dataset = tud.TensorDataset(val_tensor)
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
