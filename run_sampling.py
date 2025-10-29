import torch
from config import config_path, config, get_device
from data_loader import generate_test
from model import build_model
from sample import (
    load_checkpoint,
    pc_sample,
    pf_sample,
    plot_sample_scatter,
    report_statistics,
)
import warnings


def main() -> None:
    warnings.filterwarnings("ignore")
    device = get_device()
    if device.type == "cuda":
        print("Using device:", device)

    model = build_model(config, device)
    checkpoint = load_checkpoint(model, config_path, device)
    checkpoint_config = checkpoint.get("config", config)
    test_np = generate_test(checkpoint_config)
    with torch.no_grad():
        pc_samples = pc_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()
        pf_samples = pf_sample(model, checkpoint_config, device, checkpoint_config["n_gen"]).cpu().numpy()

    print("=== Predictor-Corrector (PC) Samples ===")
    plot_sample_scatter(test_np, pc_samples, checkpoint_config)
    report_statistics(pc_samples, checkpoint_config, test_np)
    

    print("\n=== Probability Flow (PF) Samples ===")
    plot_sample_scatter(test_np, pf_samples, checkpoint_config)
    report_statistics(pf_samples, checkpoint_config, test_np)
   


if __name__ == "__main__":
    main()
