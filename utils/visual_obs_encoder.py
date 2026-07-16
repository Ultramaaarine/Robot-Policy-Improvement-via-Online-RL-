import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisualEncoder(nn.Module):
    """
    input:
        x: [B, To, C, H, W]
        timestep_emb: [B, time_emb_dim] or None

    output:
        feat: [B, To, feat_dim]
    """
    def __init__(
        self,
        feat_dim: int,
        #gru_num_layers: int ,
        #gru_hidden_dim:int,
        in_channels: int = 3,
        
    ):
        super().__init__()
        self.feature_dim = feat_dim
        #self.gru_num_layers = gru_num_layers
        #self.gru_hidden_dims = gru_hidden_dim
        self.backbone = nn.Sequential(
            # 84 -> 42
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 42 -> 21
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 21 -> 11
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B*To, 128, 1, 1]
            nn.Flatten(),              # [B*To, 128]
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )
        # self.gru = nn.GRU(
        #    input_size=feat_dim,
        #    hidden_size=self.gru_hidden_dims,
        #    num_layers=self.gru_num_layers,
        #    bidirectional=False,

        # )


    def forward(
    self,
    x: torch.Tensor,
    
) -> torch.Tensor:
     """
     support:
        [B, To, C, H, W]
        [B, To, H, W, C]
        [To, C, H, W]
        [To, H, W, C]
     """
     if x.ndim == 4:
        # rollout 单条序列，没有 batch 维
        x = x.unsqueeze(0)
     elif x.ndim != 5:
        raise ValueError(f"x must be 4D or 5D visual tensor, got {x.shape}")

    # 统一成 [B, To, C, H, W]
     if x.shape[2] in [1, 3]:
        # already [B, To, C, H, W]
        pass
     elif x.shape[-1] in [1, 3]:
        # [B, To, H, W, C] -> [B, To, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
     else:
        raise ValueError(f"Cannot infer channel dim from shape {x.shape}")

     B, To, C, H, W = x.shape

    # normalize
     x = x.float()
     if x.max() > 1:
        x = x / 255.0

    # [B, To, C, H, W] -> [B*To, C, H, W]
     x = x.view(B * To, C, H, W)

     x = self.backbone(x)
     feat = self.head(x)                         # [B*To, feat_dim]
     feat = feat.view(B, To, self.feature_dim)   # [B, To, feat_dim]
     feat = feat #+ self.gru(feat)                # residual GRU

     return feat