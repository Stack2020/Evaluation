import torch
config = {
    "n_gen":60000,
    "n_train": 50000,
    "n_val": 10000,
    "n_test": 5000,
    "n_gen": 5000,
    "mu": [1.0, -1.0],
    "sigma": [[1.0, 0.6], [0.6, 1.5]],
    "beta_min": 0.1,
    "beta_max": 20.0,
    "eps": 1e-3,
    "hidden": 128,
    "layers": 8,
    "t_embed_dim": 32,
    "batch_size": 64,
    "epochs": 3000,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 1.0,
    "lambda_gamma": 1.0,
    "num_steps": 800,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
}

config_path = "vp_sde_2d_best.pt"

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
