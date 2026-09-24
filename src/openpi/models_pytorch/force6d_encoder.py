"""PyTorch MLP encoder for 1D 6D force vectors (matches JAX Force6DEncoder)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class Force6DEncoder(nn.Module):
    """6D force → prefix token. Random init at SFT; not in pi05_base."""

    def __init__(self, output_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Linear] = []
        in_dim = 6
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, force6d: torch.Tensor) -> torch.Tensor:
        if force6d.shape[-1] != 6:
            raise ValueError(f"Expected force6d last dim 6, got {tuple(force6d.shape)}")
        x = force6d.float()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return self.layer_norm(x)
