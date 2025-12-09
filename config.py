import torch
config = {
    "n_gen": 5000,
    "n_train": 50000,
    "n_val": 10000,
    "n_test": 5000,
    # data distribution
    "data_distribution": "involute",
    "noise_scale": 0.4,

    "mu": [-0.5, 0.0],
    "sigma": [[6.42, -0.8], [-0.8, 6.82]],

    "beta_min": 0.1,
    "beta_max": 20.0,
    "eps": 1e-3,

    "density_eval_points": 2000,
    "density_eval_steps": 600,
    "nll_time_schedule": "power",
    "nll_time_gamma": 2.5,
    "loss_time_gamma": 2.2,
    "loss_time_reweight": False,
    "loss_time_weight_max": 0.0,
    "hidden": 64,
    "layers": 3,
    "t_embed_dim": 64,
    "batch_size": 128,
    "epochs": 3000,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    "lambda_gamma": 1.0,
    "num_steps": 5000,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
    "config_path": "vp_sde_involute_best.pt",
    # ===== reproducibility =====
    "deterministic": True,
    "train_seeds": 42,
    "seed": 42,
    "data_seed_offset": 10_000,
    "loader_seed_offset": 20_000,
    "eval_seed_offset": 30_000,
    "sample_seed_offset": 40_000,
    "loader_workers": 2
}


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
