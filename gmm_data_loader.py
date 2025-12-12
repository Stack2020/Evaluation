from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.utils.data as tud

GMM_DATA = {
    "means": np.array(
        [
            [1.5, -1.0],
            [-1.0, 1.2],
            [2.5, 2.0],
        ],
        dtype=np.float32,
    ),
    "covariances": np.array(
        [
            [[0.8, 0.2], [0.2, 0.5]],
            [[0.6, -0.1], [-0.1, 0.9]],
            [[0.4, 0.3], [0.3, 0.7]],
        ],
        dtype=np.float32,
    ),
    "weights": np.array([0.4, 0.35, 0.25], dtype=np.float32),
}

def sample_gmm_gaussian(
    n: int,
    rng: np.random.Generator | None = None,
    data: Dict | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng

    mixture = GMM_DATA  if data is None else data

    means = mixture["means"]
    covariances = mixture["covariances"]

    weights = mixture.get("weights")
    num_components = means.shape[0]
    probs = (
        np.full(num_components, 1.0 / num_components, dtype=np.float32)
        if weights is None
        else np.asarray(weights, dtype=np.float32) / np.asarray(weights, dtype=np.float32).sum()
    )


    component_ids = rng.choice(num_components, size=n, p=probs)
    
    samples = np.empty((n, means.shape[1]), dtype=np.float32)
    for idx in range(num_components):
        mask = component_ids == idx
        count = int(mask.sum())
        if count == 0:
            continue
        samples[mask] = rng.multivariate_normal(
            means[idx],
            covariances[idx],
            size=count,
        ).astype(np.float32)

    return samples


def generate_train_val(config: Dict, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = sample_gmm_gaussian(config["n_train"], rng=rng)
    val = sample_gmm_gaussian(config["n_val"], rng=rng)
    return train, val


def generate_test(config: Dict, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sample_gmm_gaussian(config["n_test"], rng=rng)


def build_train_loader(train_tensor: torch.Tensor, batch_size: int) -> tud.DataLoader:
    dataset = tud.TensorDataset(train_tensor)
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )


def build_val_loader(val_tensor: torch.Tensor, batch_size: int) -> tud.DataLoader:
    dataset = tud.TensorDataset(val_tensor)
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
