from typing import Dict, Tuple
import numpy as np
import torch
import torch.utils.data as tud

def sample_gaussian(n: int, mu, sigma, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    mu_arr = np.asarray(mu, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return rng.multivariate_normal(mu_arr, sigma_arr, size=n).astype(np.float32)

def generate_train_val(config: Dict, seed: int | None = None):
    rng = np.random.default_rng(seed)
    train = sample_gaussian(config["n_train"], config["mu"], config["sigma"], rng=rng)  
    val   = sample_gaussian(config["n_val"],   config["mu"], config["sigma"], rng=rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sample_gaussian(config["n_test"], config["mu"], config["sigma"], rng=rng)



def build_train_loader(train_tensor: torch.Tensor, batch_size: int, seed: int | None = None) -> tud.DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

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
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

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