from __future__ import annotations
import math
from typing import Iterable
import numpy as np
import torch

from config import config, get_device
from data_loader import generate_test
from gmm_data_loader import GMM_DATA, generate_test as generate_gmm_test
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability


def gmm_gaussian_log_density(
    points: np.ndarray,
    mixture: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Analytic log-density of a Gaussian mixture."""
    mixture = GMM_DATA if mixture is None else mixture

    means = np.asarray(mixture["means"], dtype=np.float64)
    covariances = np.asarray(mixture["covariances"], dtype=np.float64)
    weights = mixture.get("weights")
    num_components = means.shape[0]

    if weights is None:
        weights_arr = np.full(num_components, 1.0 / num_components, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()

    log_components = []
    for mean, cov, weight in zip(means, covariances, weights_arr):
        # Log-density of a single Gaussian component
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance must be positive definite.")
        diff = points - mean
        exponent = np.einsum("...i,ij,...j->...", diff, np.linalg.inv(cov), diff)
        dim = mean.shape[0]
        log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + logdet)
        component_log = log_norm - 0.5 * exponent + math.log(weight)
        log_components.append(component_log)

    stacked = np.stack(log_components, axis=-1)
    return np.logaddexp.reduce(stacked, axis=-1)


def evaluate_model_log_density(
    model,
    cfg: dict,
    device: torch.device,
    points: np.ndarray,
    *,
    steps: int,
    eps: float,
    method: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate model log-density on arbitrary 2D points."""
    model.eval()
    outputs: list[np.ndarray] = []
    total = points.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = torch.from_numpy(points[start:end]).to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                xb,
                cfg,
                steps=steps,
                eps=eps,
                method=method,
            )
        outputs.append(logp.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(config["config_path"]), device)
    ckpt_cfg = checkpoint.get("config", {})
    cfg = config.copy()
    cfg.update(ckpt_cfg)
    cfg["config_path"] = str(config["config_path"])

    # Gaussian mixture only
    gmm_means = np.asarray(cfg.get("gmm_means", GMM_DATA["means"]), dtype=np.float64)
    gmm_covs = np.asarray(cfg.get("gmm_covariances", GMM_DATA["covariances"]), dtype=np.float64)
    gmm_weights = cfg.get("gmm_weights", GMM_DATA.get("weights"))
    mixture = {
        "means": gmm_means,
        "covariances": gmm_covs,
        "weights": gmm_weights,
    }
    test_np = generate_gmm_test(cfg, seed=int(cfg.get("eval_seed_offset", 0))).astype(np.float32)

    # Fixed test set (2D) for all evaluations
    batch_size = int(cfg.get("score_batch_size", 512))
    
    steps_list: Iterable[int] = (100, 150, 200, 400)

    analytic_logp = gmm_gaussian_log_density(test_np.astype(np.float64), mixture)
    analytic_nll = float(-analytic_logp.mean())

    method = str(cfg.get("density_method", "Heun"))
    eps_val = float(cfg.get("nll_eval_eps", cfg.get("eps", 1e-3)))

    print(f"Test samples: {test_np.shape[0]} (2D)")
    print("NFE\tPF-ODE NLL\tAnalytic NLL\t|Δ|\tKL(p_data||p_theta)")

    for steps in steps_list:
        model_logp = evaluate_model_log_density(
            model,
            cfg,
            device,
            test_np,
            steps=steps,
            eps=eps_val,
            method=method,
            batch_size=batch_size,
        )
        # PF-ODE NLL value (expected -log p under the data distribution)
        pf_nll_value = float(-model_logp.mean())
        delta = abs(pf_nll_value - analytic_nll)
        KL = float((analytic_logp - model_logp).mean())

        print(
            f"{steps}\t"
            f"{pf_nll_value:.8f}\t"
            f"{analytic_nll:.8f}\t"
            f"{delta:.8f}\t"
            f"{KL:.8f}"
        )


if __name__ == "__main__":
    main()
