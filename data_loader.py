import math
from typing import Dict, Tuple
import numpy as np
import torch
import torch.utils.data as tud

def banana_forward(z: np.ndarray, a: float) -> np.ndarray:
    """Warp standard Gaussian samples into the banana distribution."""
    x1 = z[..., 0]
    x2 = z[..., 1] + a * (x1 **2 - 1.0)
    return np.stack((x1, x2), axis=-1)

def banana_inverse(x: np.ndarray, a: float) -> np.ndarray:
    """Inverse warp used for analytic density computation."""
    x_arr = np.asarray(x, dtype=np.float64)
    x1 = x_arr[..., 0]
    z1 = x1
    z2 = x_arr[..., 1] - a * (x1 **2- 1.0)
    return np.stack((z1, z2), axis=-1)


def banana_log_density(points: np.ndarray, a: float) -> np.ndarray:
    """Analytic log-density of the banana distribution via change of variables."""
    z = banana_inverse(points, a)
    quad = np.sum(z * z, axis=-1)
    dim = z.shape[-1]
    log_norm = -0.5 * dim * math.log(2.0 * math.pi)
    return log_norm - 0.5 * quad


def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov_inv = np.linalg.inv(cov)
    diff = points - mean
    exponent = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
    dim = mean.shape[0]
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + np.log(np.linalg.det(cov)))
    return log_norm - 0.5 * exponent


def sample_gaussian(n: int, mu, sigma, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    mu_arr = np.asarray(mu, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return rng.multivariate_normal(mu_arr, sigma_arr, size=n).astype(np.float32)


def sample_banana(n: int, a: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    z = rng.standard_normal(size=(n, 2), dtype=np.float32)
    x = banana_forward(z, a)
    return x.astype(np.float32)


def sample_target_distribution(n: int, config: Dict, rng: np.random.Generator | None = None) -> np.ndarray:
    dist = str(config.get("distribution", "gaussian")).lower()
    if dist == "banana":
        return sample_banana(n, float(config.get("banana_a", 0.2)), rng=rng)
    return sample_gaussian(n, config["mu"], config["sigma"], rng=rng)


def get_distribution_stats(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    dist = str(config.get("distribution", "gaussian")).lower()
    if dist == "banana":
        a = float(config.get("banana_a", 0.2))
        mean = np.zeros(2, dtype=np.float64)
        var2 = 1.0 + 2.0 * (a * a)
        cov = np.array([[1.0, 0.0], [0.0, var2]], dtype=np.float64)
        return mean, cov
    #默认支线
    mean = np.asarray(config["mu"], dtype=np.float64)
    cov = np.asarray(config["sigma"], dtype=np.float64)
    return mean, cov


def generate_train_val(config: Dict, seed: int | None = None):
    rng = np.random.default_rng(seed)
    train = sample_target_distribution(config["n_train"], config, rng=rng)
    val = sample_target_distribution(config["n_val"], config, rng=rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sample_target_distribution(config["n_test"], config, rng=rng)


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
