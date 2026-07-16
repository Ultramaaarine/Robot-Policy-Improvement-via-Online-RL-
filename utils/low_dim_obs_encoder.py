import torch
import torch.nn as nn
from typing import Literal


class Low_Dim_Obs_Encoder(nn.Module):
    """
    Encode low-dimensional temporal condition.

    Input:
        cond: [B, T, cond_dim]

    Output:
        if mode == "sequence":
            [B, T, out_dim]   # for diffusion
        if mode == "global":
            [B, out_dim]      # for critic
    """
    def __init__(
        self,
        cond_dim: int,
        out_dim: int,
        conv_hidden_dim: int = 128,
        #gru_hidden_dim: int = 128,
        #gru_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.cond_dim = int(cond_dim or 0)
        self.out_dim = int(out_dim)
        self.conv_hidden_dim = int(conv_hidden_dim)
        #self.gru_hidden_dim = int(gru_hidden_dim)
        self.bidirectional = bool(bidirectional)

        if self.cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {self.cond_dim}")

        # temporal local feature extractor
        self.conv_net = nn.Sequential(
            nn.Conv1d(
                in_channels=self.cond_dim,
                out_channels=self.conv_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv_hidden_dim,
                out_channels=self.conv_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # temporal memory
        # self.gru = nn.GRU(
        #     input_size=self.conv_hidden_dim,
        #     hidden_size=self.gru_hidden_dim,
        #     num_layers=gru_layers,
        #     batch_first=True,
        #     bidirectional=self.bidirectional
        # )

        #gru_out_dim = self.gru_hidden_dim * (2 if self.bidirectional else 1)

        # projection head
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_hidden_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim)
        )

    def forward(
        self,
        cond: torch.Tensor,
        mode: Literal["sequence", "global"] = "sequence"
    ) -> torch.Tensor:
        """
        Args:
            cond: [B, T, cond_dim]
            mode:
                - "sequence": return [B, T, out_dim]
                - "global":   return [B, out_dim]

        Returns:
            encoded cond
        """
        if cond.ndim != 3:
            raise ValueError(f"cond must be [B, T, cond_dim], got shape {cond.shape}")

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond last dim {cond.shape[-1]} != cond_dim {self.cond_dim}"
            )

        # [B, T, D] -> [B, D, T]
        x = cond.permute(0, 2, 1).contiguous()

        # Conv over time
        # [B, D, T] -> [B, C, T]
        x = self.conv_net(x)

        # [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1).contiguous()

        # GRU over time
        # [B, T, C] -> [B, T, H]
        #x, _ = self.gru(x)

        # projection
        # [B, T, H] -> [B, T, out_dim]
        x = self.mlp(x)

        if mode == "sequence":
            return x
        elif mode == "global":
            return x[:, -1, :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")