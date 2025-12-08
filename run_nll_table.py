from __future__ import annotations
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
from calculate_liklihood import calculate_liklihood
from calculate_true_likelihood import calculate_true_likelihood
from config import config, get_device
from data_loader import generate_test
from model import build_model
from sample import load_checkpoint

def ae_stats(errors: torch.Tensor) -> Tuple[float, float, float, float]:
    """Return mean, std (ddof=1), and 95% CI bounds for absolute errors."""
    errs = errors.detach().cpu().numpy()
    mean = float(errs.mean())
    std = float(errs.std(ddof=1)) if errs.size > 1 else 0.0
    n_total = errs.size
    if n_total == 0:
        return mean, std, mean, mean
    ci_half = 1.96 * std / math.sqrt(n_total) if std > 0 else 0.0
    return mean, std, mean - ci_half, mean + ci_half


def observed_p(E_N: float, E_2N: float) -> float:
    return math.log(E_N / E_2N) / math.log(2.0)


def richardson_bias(E_N: float, E_2N: float, p_obs: float) -> float:
    if abs(2 * p_obs - 1) < 1e-12:
        return float("nan")
    L_inf = E_2N + (E_2N - E_N) / (2.0 ** p_obs - 1.0)
    return abs(E_2N - L_inf) / (2.0 ** p_obs - 1.0)


def ensure_table_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, config["config_path"] if "config_path" in config else "vp_sde_2d_best.pt", device)

    checkpoint_config: Dict = checkpoint.get("config", config)
    #向量化编程
    seeds = 5
    nfe_list = [100, 200, 400, 800, 1600, 3200]

    all_errors: Dict[int, List[float]] = {nfe: [] for nfe in nfe_list}

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        test_np = generate_test(checkpoint_config, seed=seed)
        test_tensor = torch.from_numpy(test_np).to(device)
        true_log_probs, _ = calculate_true_likelihood(
            checkpoint_config,
            device,
            test_tensor=test_tensor,
        )
        true_log_probs_cpu = true_log_probs.cpu()

        with torch.enable_grad():
            for nfe in nfe_list:
                log_probs, _ = calculate_liklihood(
                    model,
                    checkpoint_config,
                    device,
                    steps=nfe,
                    method="Heun",
                    test_tensor=test_tensor,
                )
                errors = (log_probs.cpu() - true_log_probs_cpu).abs()
                all_errors[nfe].extend(errors.tolist())

    table_rows: List[Dict[str, float | str]] = []
    for idx, nfe in enumerate(nfe_list):
        errors = torch.tensor(all_errors[nfe], dtype=torch.float32)
        AE_mean, AE_std, CI_lo, CI_hi = ae_stats(errors)

        row = {
            "NFE": nfe,
            "AE_mean": AE_mean,
            "AE_std": AE_std,
            "AE_CI_lo": CI_lo,
            "AE_CI_hi": CI_hi,
            "Observed_p": "--",
            "Bias_est": "--",
        }

        if idx > 0:
            prev_errors = torch.tensor(all_errors[nfe_list[idx - 1]], dtype=torch.float32)
            E_N = float(prev_errors.mean().item())
            E_2N = float(errors.mean().item())
            if E_N > 0 and E_2N > 0:
                p_obs = observed_p(E_N, E_2N)
                bias_est = richardson_bias(E_2N, E_N, p_obs)
                row["Observed_p"] = p_obs
                row["Bias_est"] = bias_est
        table_rows.append(row)

    table_dir = Path("table")
    ensure_table_dir(table_dir)
    output_path = table_dir / "nll_table.csv"

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "NFE",
                "AE_mean",
                "AE_std",
                "AE_CI_lo",
                "AE_CI_hi",
                "Observed_p",
                "Bias_est",
            ],
        )
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    print(f"Saved NLL table to {output_path}")


if __name__ == "__main__":
    main()
