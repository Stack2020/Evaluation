import torch
import torch.nn as nn
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.linspace(1.0, 100.0, dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        angles = t * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 3, t_embed_dim: int = 32):
        super().__init__()
        self.temb = TimeEmbedding(t_embed_dim)
        in_dim = 2 + t_embed_dim
        modules = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32)
        temb = self.temb(t)
        return self.net(torch.cat([x, temb], dim=-1))


def build_model(config: dict, device: torch.device) -> ScoreNet:
    model = ScoreNet(
        hidden=config["hidden"],
        layers=config["layers"],
        t_embed_dim=config["t_embed_dim"],
    )
    return model.to(device)
