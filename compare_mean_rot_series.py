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
    label_lambda_1e5 = r"$\lambda=1\times 10^{-5}$"
    label_lambda_1e4 = r"$\lambda=1\times 10^{-4}$"
    label_lambda_1e3 = r"$\lambda=1\times 10^{-3}$"
    label_lambda_t = r"$\lambda=\lambda(t)$"
    label_lambda0 = r"$\lambda=0$"

    files = [
        (label_lambda_1e5, Path("vp_sde_2d_regularization_low_mean_rot_series.npz")),
        (label_lambda_1e4, Path("vp_sde_2d_regularization_mean_mean_rot_series.npz")),
        (label_lambda_1e3, Path("vp_sde_2d_regularization_high_mean_rot_series.npz")),
        (label_lambda_t, Path("vp_sde_2d_regularization_sigmoid_mean_rot_series.npz")),
        (label_lambda0, Path("vp_sde_2d_best_mean_rot_series.npz")),
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
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.set_size_inches(5, 4)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Distinct, publication-friendly palette + line styles.
    colors = {
        label_lambda_1e5: "#1b9e77",  # green
        label_lambda_1e4: "#d95f02",  # orange
        label_lambda_1e3: "#7570b3",  # purple
        label_lambda_t: "#c1121f",    # red
        label_lambda0: "#005f73",     # deep teal
    }

    linestyles = {
        label_lambda_1e5: "-",
        label_lambda_1e4: (0, (5, 2)),           # long dash
        label_lambda_1e3: (0, (5, 2, 1.3, 2)),   # dash-dot
        label_lambda_t: "-",                      # solid for lambda(t)
        label_lambda0: (0, (9, 2, 2, 2, 2, 2)),  # long dash + two dots
    }

    for label, times, mean_rot in series:
        ax.plot(
            times,
            mean_rot,
            label=label,
            linewidth=2.2,
            color=colors.get(label, None),
            linestyle=linestyles.get(label, "-"),
        )

    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.legend(frameon=False)

    output_path = Path("mean_rot_series_comparison.png")
    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.10, top=0.90)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
