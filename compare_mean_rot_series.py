from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with np.load(path) as data:
        times = np.asarray(data["times"], dtype=float)
        mean_rot = np.asarray(data["mean_rot"], dtype=float)
    return times, mean_rot


def main() -> None:
    label_lambda0 = r"$\lambda=0$"
    label_lambda_t = r"$\lambda=\lambda(t)$"
    files = [
        (label_lambda0, Path("vp_sde_2d_best_mean_rot_series.npz")),
        (label_lambda_t, Path("vp_sde_2d_regularization_mean_rot_series.npz")),
    ]

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, path in files:
        times, mean_rot = load_series(path)
        series.append((label, times, mean_rot))

    ref_times = series[0][1]
    for label, times, _ in series[1:]:
        if times.shape != ref_times.shape or np.max(np.abs(times - ref_times)) > 1e-8:
            raise ValueError(f"Time grid mismatch for series '{label}'.")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = {
        label_lambda0: "#0a3a5c",
        label_lambda_t: "#d1495b",
    }

    for label, times, mean_rot in series:
        ax.plot(
            times,
            mean_rot,
            label=label,
            linewidth=2.0,
            color=colors.get(label, None),
        )

    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.legend(frameon=False)

    output_path = Path("mean_rot_series_comparison.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
