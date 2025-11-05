import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import (
    load_checkpoint,
    oracle_sample,
    pc_sample,
    pf_sample,
    report_statistics,
)

warnings.filterwarnings("ignore")


def show_sample_scatter(
    true_samples: np.ndarray,
    generated_samples: np.ndarray,
    method_label: str,
) -> None:
    """Display real vs generated samples with shared axes for easy comparison."""
    true_samples = np.asarray(true_samples, dtype=float)
    generated_samples = np.asarray(generated_samples, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    combined = np.vstack([true_samples, generated_samples])
    finite_mask = np.isfinite(combined).all(axis=1)
    valid = combined[finite_mask]

    if valid.size == 0:
        x_limits = (-1.0, 1.0)
        y_limits = (-1.0, 1.0)
    else:
        x_min, x_max = float(valid[:, 0].min()), float(valid[:, 0].max())
        y_min, y_max = float(valid[:, 1].min()), float(valid[:, 1].max())

        def pad_limits(lo: float, hi: float) -> tuple[float, float]:
            span = hi - lo
            if span <= 0:
                pad = max(1.0, max(abs(lo), abs(hi)) * 0.1 + 1e-6)
                return lo - pad, hi + pad
            pad = max(0.05 * span, 1e-6)
            return lo - pad, hi + pad

        x_limits = pad_limits(x_min, x_max)
        y_limits = pad_limits(y_min, y_max)

    axes[0].scatter(true_samples[:, 0], true_samples[:, 1], s=10, alpha=0.6, color="tab:blue")
    axes[0].set_title("Test Samples")

    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=10, alpha=0.6, color="tab:orange")
    axes[1].set_title(f"{method_label} Samples")

    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{method_label} Sampling Comparison")
    fig.tight_layout()
    plt.show()


def main() -> None:

    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    model = build_model(config, device)
    config_path = config["config_path"]
    checkpoint = load_checkpoint(model, config_path, device)
    checkpoint_config = checkpoint.get("config", config)
    test_np = generate_test(checkpoint_config)

    with torch.no_grad():
        pc_samples = pc_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()
        pf_samples = pf_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()
        oracle_samples = oracle_sample(
            checkpoint_config,
            device,
            checkpoint_config["n_gen"],
        ).cpu().numpy()

    print("=== Predictor-Corrector (PC) Samples ===")
    show_sample_scatter(test_np, pc_samples, "PC")
    report_statistics(pc_samples, checkpoint_config, test_np)

    print("\n=== Probability Flow (PF) Samples ===")
    show_sample_scatter(test_np, pf_samples, "PF-ODE")
    report_statistics(pf_samples, checkpoint_config, test_np)

    print("\n=== Oracle Samples ===")
    show_sample_scatter(test_np, oracle_samples, "Oracle")
    report_statistics(oracle_samples, checkpoint_config, test_np)

if __name__ == "__main__":
    main()
