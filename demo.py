import matplotlib.pyplot as plt
import numpy as np

from config import config
from data_loader import generate_test

def main() -> None:
    seed = int(config.get("seed", 42))

    n_samples = int(config.get("n_test", 5000))

    samples = generate_test(config, seed=seed)
    
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.6, color="tab:blue")
    ax.set_title(f"Involute samples (n={n_samples}, seed={seed})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    # add small text with stats for quick sanity
    text = f"mean = [{mean[0]:.2f}, {mean[1]:.2f}]\n" \
           f"cov = [[{cov[0,0]:.2f}, {cov[0,1]:.2f}],\n       [{cov[1,0]:.2f}, {cov[1,1]:.2f}]]"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    out_path = "involute_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()
