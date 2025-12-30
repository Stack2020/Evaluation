import torch
config = {
    "n_gen":60000,
    "n_train": 50000,
    "n_val": 10000,
    "n_test": 5000,
    "n_gen": 5000,   
    "seed":42,
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
    "hidden": 64,  # 从128增加到256
    "layers": 3,   # 从8增加到10
    "t_embed_dim": 64,
    "batch_size": 128,
    "epochs": 3500,  # 训练更久
    "lr": 1e-3,      # 降低学习率，更稳定
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    "lambda_gamma": 1.0,  # 降低权重的影响，更平衡地训练所有时间步
    #正则项惩罚动态调整
    "curl":True,
    "num_steps": 5000,
    "snr": 0.08,
    "corrector_steps": 2,
    "scatter_alpha": 0.5,
    "scatter_size": 10,
    "config_path":"vp_sde_2d_best_gmm.pt"
}
         

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")