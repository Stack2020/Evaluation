from typing import Sequence
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
    def __init__(
        self,
        data_dim: int = 2,
        hidden: int | Sequence[int] = 128,
        layers: int = 3,
        t_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.temb = TimeEmbedding(t_embed_dim)
        self.data_dim = int(data_dim)
        in_dim = self.data_dim + t_embed_dim

        if isinstance(hidden, Sequence) and not isinstance(hidden, (str, bytes)):
            hidden_dims = [int(h) for h in hidden]
        else:
            hidden_dims = [int(hidden)] * int(layers)

        if not hidden_dims:
            raise ValueError("At least one hidden dimension is required for ScoreNet.")

        dims = [in_dim, *hidden_dims, self.data_dim]
        modules: list[nn.Module] = []
        for in_size, out_size in zip(dims[:-1], dims[1:]):
            modules.append(nn.Linear(in_size, out_size))
            if out_size != self.data_dim:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32)
        temb = self.temb(t)
        return self.net(torch.cat([x, temb], dim=-1))


def build_model(config: dict, device: torch.device) -> ScoreNet:
    data_dim = int(config.get("data_dim", 2))
    model = ScoreNet(
        data_dim=data_dim,
        hidden=config["hidden"],
        layers=config["layers"],
        t_embed_dim=config["t_embed_dim"],
    )
    return model.to(device)
