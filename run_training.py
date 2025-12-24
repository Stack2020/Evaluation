from config import  config, get_device
from miniboone_data_loader import prepare_miniboone_loaders
from model import build_model
from train import train_model
from util import set_seed
import warnings
warnings.filterwarnings("ignore")

def main() -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    train_loader, val_loader, _test_loader, stats = prepare_miniboone_loaders(
        path=config.get("miniboone_path", "./data/miniboone.npy"),
        batch_size=config["batch_size"],
        seed=seed,
    )

    data_dim = stats["train_raw"].shape[1]
    if int(config.get("data_dim", data_dim)) != data_dim:
        print(f"Updating data_dim from {config.get('data_dim')} to {data_dim}")
    config["data_dim"] = data_dim

    print(
        "Splits (train/val/test):",
        stats["train_raw"].shape,
        stats["val_raw"].shape,
        stats["test_raw"].shape,
    )

    model = build_model(config, device)
   
    print(f"Params: {sum(p.numel() for p in model.parameters()):.1f}")

    checkpoint_path = config.get("config_path", "vp_sde_miniboone_best.pt")
    config["config_path"] = checkpoint_path
    
    print(f"Checkpoint path: {checkpoint_path}")

    train_model(
        model,
        config,
        device,
        train_loader,
        val_loader,
        checkpoint_path)

if __name__ == "__main__":
    main()
