import torch

config = {
    "seed": 42,
    "data_dim": 5,
    "n_train": 50000,
    "n_val": 10000,
    "n_test": 5000,
    "n_gen": 5000,
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
    "density_eval_steps": 600,
    
    "nll_time_schedule": "power",
    "nll_time_gamma": 2.0,
    "loss_time_gamma": 1.8,         # tilt sampling toward low-noise region
    "loss_time_reweight": True,     # reweight to emphasize late times
    "loss_time_weight_max": 5.0,    # clamp weight to avoid extremes
    "hidden": 512,
    "layers": 5,
    "t_embed_dim": 128,
    "batch_size": 256,
    "epochs": 4000,
    "lr": 1e-3,
    "lr_min": 1e-4,                 # keep lr from decaying too low
    "reset_lr": 8e-4,               # plateau时重置到的lr
    "weight_decay": 1e-5,
    "clip_grad_norm": 5.0,
    
    "manual_drop_factor": 0.5,      # plateau时衰减一半，再用reset_lr兜底
    "T_max_after_reset": 15,        # shorter cosine after reset
    "degradation_patience": 10,     # 更快触发plateau
    "min_improve_delta": 0.01,      # 提升阈值
    "num_steps": 4000,
    "snr": 0.08,
    "corrector_steps": 2,
    "config_path": "vp_sde_5d_gaussian.pt",
}


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")