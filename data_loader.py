from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.utils.data as tud

def sample_gaussian(n: int, mu, sigma, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    mu_arr = np.asarray(mu, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return rng.multivariate_normal(mu_arr, sigma_arr, size=n).astype(np.float32)

def generate_train_val(config: Dict, seed: int = None):
    rng = np.random.default_rng(seed)
    train = sample_gaussian(config["n_train"], config["mu"], config["sigma"],rng=rng)  
    val   = sample_gaussian(config["n_val"],   config["mu"], config["sigma"],rng=rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sample_gaussian(config["n_test"], config["mu"], config["sigma"], rng=rng)



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
