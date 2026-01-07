import torch
import torch.nn as nn
from typing import Sequence


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.linspace(1.0, 100.0, dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        angles = t * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class ScoreNet(nn.Module):
    def __init__(
        self,
        hidden: Sequence[int] | int = 128,
        layers: int | None = 3,
        t_embed_dim: int = 32,
        data_dim: int = 2,
    ):
        super().__init__()
        self.temb = TimeEmbedding(t_embed_dim)
        in_dim = data_dim + t_embed_dim

        if isinstance(hidden, int):
            hidden_dims = [hidden] * (layers or 1)
        else:
            hidden_dims = list(hidden)
            if layers is not None and layers != len(hidden_dims):
                raise ValueError(
                    f"layers ({layers}) does not match number of hidden dimensions ({len(hidden_dims)})"
                )
        modules = []
        for h_dim in hidden_dims:
            modules.extend([nn.Linear(in_dim, h_dim), nn.SiLU()])
            in_dim = h_dim
        modules.append(nn.Linear(in_dim, data_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32)
        temb = self.temb(t)
        return self.net(torch.cat([x, temb], dim=-1))



def build_model(config: dict, device: torch.device) -> ScoreNet:
    data_dim = int(config.get("data_dim", len(config.get("mu", [])) or 2))
    model = ScoreNet(
        hidden=config["hidden"],
        layers=config.get("layers"),
        t_embed_dim=config["t_embed_dim"],
        data_dim=data_dim,
    )
    return model.to(device)