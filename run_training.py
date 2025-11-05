import torch
from config import  config, get_device
from data_loader import build_train_loader, build_val_loader, generate_train_val
from model import build_model
from train import train_model

import warnings
warnings.filterwarnings("ignore")
def main() -> None:
    
    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    train_np, val_np = generate_train_val(config)
    print("Train/Val shapes:", train_np.shape, val_np.shape)

    train_tensor = torch.from_numpy(train_np)
    val_tensor = torch.from_numpy(val_np)

    model = build_model(config, device)
   
    print(f"Params: {sum(p.numel() for p in model.parameters()):.1f}")

    train_loader = build_train_loader(train_tensor, config["batch_size"])

    val_loader = build_val_loader(val_tensor, config["batch_size"])

    train_model(
        model,
        config,
        device,
        train_loader,
        val_loader,
        config["config_path"],
        enable_nll=False)

if __name__ == "__main__":
    main()
