import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Parameters that define the schedule. Kept as constants so the plot and callable
# always stay in sync.
LAMBDA_MAX = 0.1
T_CUTOFF = 0.1
K_SHARPNESS = 120


def lambda_curl(t: torch.Tensor) -> torch.Tensor:
    """Time-dependent curl regularization weight."""
    return LAMBDA_MAX * torch.sigmoid(-K_SHARPNESS * (t - T_CUTOFF))


def plot_lambda_curl_curve(
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_points: int = 400,
    save_path: str = "lambda_curl_schedule.png",
) -> None:
    """Plot lambda_curl with publication-ready styling."""
    # Configure a clean serif look suitable for print.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 300,
        }
    )

    ts = torch.linspace(t_min, t_max, num_points)
    y = lambda_curl(ts).cpu().numpy()
    x = ts.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.set_size_inches(5, 4)

    # Main curve.
    ax.plot(x, y, color="#7b1fa2", linewidth=2.2, label=r"$\lambda_{curl}(t)$")
    ax.fill_between(x, y, color="#b39ddb", alpha=0.2)

    ax.axvline(
        T_CUTOFF,
        color="#c43c30",
        linestyle="--",
        linewidth=1.5,
        label=r"$t_{cutoff}$",
    )

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, LAMBDA_MAX * 1.05)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\lambda_{curl}(t)$")
    ax.set_title("Curl Regularization Schedule")
    ax.legend(frameon=False, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.10, top=0.90)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_lambda_curl_curve()
