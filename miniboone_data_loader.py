from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data as tud


def load_miniboone_array(path: str = "data/miniboone.npy") -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.float32)


def split_dataset(
    data: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle and split X into train/val/test arrays.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n = data.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return data[train_idx], data[val_idx], data[test_idx]


def z_standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_z = (train - mean) / std
    val_z = (val - mean) / std
    test_z = (test - mean) / std

    return train_z, val_z, test_z, mean, std


def build_loader(tensor: torch.Tensor, batch_size: int, shuffle: bool, drop_last: bool, seed: int | None = None) -> tud.DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataset = tud.TensorDataset(tensor)

    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        generator=generator,
    )


def prepare_miniboone_loaders(
    path: str = "data/miniboone.npy",
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> Tuple[tud.DataLoader, tud.DataLoader, tud.DataLoader, Dict[str, np.ndarray]]:

    X = load_miniboone_array(path)

    train_np, val_np, test_np = split_dataset(
        X,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_z, val_z, test_z, mean, std = z_standardize(train_np, val_np, test_np)

    train_loader = build_loader(torch.from_numpy(train_z), batch_size, shuffle=True, drop_last=True, seed=seed)
    val_loader = build_loader(torch.from_numpy(val_z), batch_size, shuffle=False, drop_last=False, seed=seed)
    test_loader = build_loader(torch.from_numpy(test_z), batch_size, shuffle=False, drop_last=False, seed=seed)

    stats: Dict[str, np.ndarray] = {
        "train_raw": train_np,
        "val_raw": val_np,
        "test_raw": test_np,
        "mean": mean,
        "std": std,
    }

    return train_loader, val_loader, test_loader, stats
