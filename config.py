import torch

config = {
    "seed": 52,
    "data_dim": 5,
    "n_train": 100000,
    "n_val": 10000,
    "n_test": 10000,
    "n_gen": 10000,
    "mu": [1.0, -1.0, 0.5, 0.0, 1.5],
    "sigma": [
        [1.0, 0.4, 0.2, 0.0, -0.1],
        [0.4, 1.5, 0.3, 0.1, 0.0],
        [0.2, 0.3, 0.8, -0.2, 0.05],
        [0.0, 0.1, -0.2, 1.2, 0.25],
        [-0.1, 0.0, 0.05, 0.25, 1.0],
    ],
    "beta_min": 0.1,
    "beta_max": 20.0,
    "eps": 1e-3,

    "density_eval_points": 2000,
    "density_eval_steps": 1200,

    "nll_time_schedule": "power",
    "nll_time_gamma": 2.5,
    "loss_time_gamma": 2.2,
    "loss_time_reweight": False,
    "loss_time_weight_max": 0.0,

    "hidden": [64,128,256,512],
    "layers": 4,
    "t_embed_dim": 64,
    "batch_size": 128,
    "epochs": 3500,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    "lambda_gamma": 1.0,
    #正则项惩罚动态调整
    "curl":True,

    "num_steps": 5000,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
    "density_method":"heun",
    "config_path": "vp_sde_best_gaussian_high.pt",
}


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")