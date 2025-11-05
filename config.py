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
    "sde_type": "subvp",
    "eps": 1e-3,
    "hidden": 256,  # 从128增加到256
    "layers": 10,   # 从8增加到10
    "t_embed_dim": 256,
    "batch_size": 128,
    "epochs": 2000,  # 训练更久
    "lr": 1e-3,      # 降低学习率，更稳定
    "weight_decay": 1e-4,
    "clip_grad_norm": 1.0,
    "lambda_gamma": 0.5,  # 降低权重的影响，更平衡地训练所有时间步
    "num_steps": 5000,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
    "config_path":"vp_sde_2d_best.pt"
}
         

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
