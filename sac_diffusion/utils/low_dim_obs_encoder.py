# Encode low-dimensional temporal observations with Conv1d + MLP.
# Supports sequence features for diffusion and global features for the critic.
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

        # Input condition feature dimension.
        self.cond_dim = int(cond_dim or 0)

        # Output embedding dimension.
        self.out_dim = int(out_dim)

        # Hidden channel dimension used by temporal Conv1d layers.
        self.conv_hidden_dim = int(conv_hidden_dim)

        #self.gru_hidden_dim = int(gru_hidden_dim)

        # Kept for compatibility with the previous GRU-based design.
        self.bidirectional = bool(bidirectional)

        # Low-dimensional condition must contain at least one feature.
        if self.cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {self.cond_dim}")

        # temporal local feature extractor
        # Conv1d operates along the temporal dimension and extracts
        # local temporal patterns from the condition sequence.
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
        # Previous GRU-based temporal encoder.
        # Currently disabled, so temporal encoding is handled only by Conv1d.
        # self.gru = nn.GRU(
        #     input_size=self.conv_hidden_dim,
        #     hidden_size=self.gru_hidden_dim,
        #     num_layers=gru_layers,
        #     batch_first=True,
        #     bidirectional=self.bidirectional
        # )

        #gru_out_dim = self.gru_hidden_dim * (2 if self.bidirectional else 1)

        # projection head
        # Project temporal Conv features into the final condition embedding space.
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

        # Input must be a temporal feature sequence.
        if cond.ndim != 3:
            raise ValueError(f"cond must be [B, T, cond_dim], got shape {cond.shape}")

        # Check that the input feature dimension matches the configured cond_dim.
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond last dim {cond.shape[-1]} != cond_dim {self.cond_dim}"
            )

        # [B, T, D] -> [B, D, T]
        # Conv1d expects the channel dimension before the temporal dimension.
        x = cond.permute(0, 2, 1).contiguous()

        # Conv over time
        # [B, D, T] -> [B, C, T]
        # Extract local temporal features while preserving sequence length.
        x = self.conv_net(x)

        # [B, C, T] -> [B, T, C]
        # Convert back to sequence-first feature layout.
        x = x.permute(0, 2, 1).contiguous()

        # GRU over time
        # [B, T, C] -> [B, T, H]
        # Recurrent temporal modeling is currently disabled.
        #x, _ = self.gru(x)

        # projection
        # [B, T, H] -> [B, T, out_dim]
        # Project each timestep feature into the final embedding dimension.
        x = self.mlp(x)

        # Return the full temporal embedding for diffusion conditioning.
        if mode == "sequence":
            return x

        # Return only the final timestep representation as a global feature,
        # typically used by the critic.
        elif mode == "global":
            return x[:, -1, :]

        else:
            raise ValueError(f"Unsupported mode: {mode}")
