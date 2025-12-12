import matplotlib.pyplot as plt
import numpy as np

from config import config
from data_loader import generate_train_val
from util import set_seed


def main() -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    samples, _ = generate_train_val(config, seed=seed)
    if samples.shape[0] > 5000:
        samples = samples[:5000]

    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5, color="tab:blue")
    plt.title("Involute dataset (seed={})".format(seed))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.2)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
