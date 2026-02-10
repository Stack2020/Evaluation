import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config, get_device
from miniboone_data_loader import generate_test

from model import build_model
from sample import (
    oracle_sample,
    pc_sample,
    pf_sample,
)
from util import set_seed
warnings.filterwarnings("ignore")

def show_sample_scatter(
    true_samples: np.ndarray,
    generated_samples: np.ndarray,
    method_label: str,
    output_path: str | Path | None = None,
    show_plot: bool = True,
    generated_color: str = "tab:green",
    mean_marker_size: int = 40,
) -> None:
    
    """Display real vs generated samples with shared axes for easy comparison."""
    true_samples = np.asarray(true_samples, dtype=float)
    generated_samples = np.asarray(generated_samples, dtype=float)

    # Match paper-ready aspect with consistent canvas size.
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

    axes[0].scatter(
        true_samples[:, 0],
        true_samples[:, 1],
        s=10,
        alpha=0.6,
        color="tab:blue",
        label="true scatter",
    )
    axes[0].set_title("true samples")

    axes[1].scatter(
        generated_samples[:, 0],
        generated_samples[:, 1],
        s=10,
        alpha=0.6,
        color=generated_color,
        label=f"{method_label} scatter",
    )
    axes[1].set_title(f"{method_label} Samples")

    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    axes[0].legend(loc="upper right", frameon=True, framealpha=0.8)
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.8)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {method_label} scatter to {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    cfg = dict(config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    checkpoint_path = str(cfg["config_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = dict(checkpoint.get("config", {}))
    merged_cfg = dict(cfg)
    merged_cfg.update(ckpt_cfg)
    merged_cfg["config_path"] = checkpoint_path

    test_np = generate_test(merged_cfg, seed=seed)
    data_dim = int(test_np.shape[1]) if test_np.ndim > 1 else 1
    merged_cfg["data_dim"] = data_dim

    model = build_model(merged_cfg, device)
    model.load_state_dict(checkpoint["model"])

    checkpoint_config = merged_cfg

    output_paths = {
        "PC": Path("pc_samples.png"),
        "PF-ODE": Path("pf_samples.png"),
        "Oracle": Path("oracle_samples.png"),
    }

    with torch.no_grad():
        pc_samples = pc_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()
        
        pf_samples = pf_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()
        
        oracle_samples = oracle_sample(
            checkpoint_config,
            device,
            checkpoint_config["n_gen"],
        ).cpu().numpy()

    print("=== Predictor-Corrector (PC) Samples ===")
    show_sample_scatter(
        test_np,
        pc_samples,
        "PC",
        output_path=output_paths["PC"],
    )

    print("\n=== Probability Flow (PF) Samples ===")
    show_sample_scatter(
        test_np,
        pf_samples,
        "PF-ODE",
        output_path=output_paths["PF-ODE"],
    )

    print("\n=== Oracle Samples ===")
    show_sample_scatter(
        test_np,
        oracle_samples,
        "Oracle",
        output_path=output_paths["Oracle"],
    )

if __name__ == "__main__":
    main()
