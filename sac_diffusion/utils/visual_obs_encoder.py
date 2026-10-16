# Encode temporal visual observations from the gripper camera using a CNN.
# Each image is converted into a compact feature vector while preserving
# the temporal sequence for later fusion with low-dimensional conditions.

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

        # Output feature dimension for each visual timestep.
        self.feature_dim = feat_dim

        #self.gru_num_layers = gru_num_layers
        #self.gru_hidden_dims = gru_hidden_dim

        # CNN backbone for extracting spatial features from each image.
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

        # Convert CNN feature maps into one compact feature vector per image.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B*To, 128, 1, 1]
            nn.Flatten(),              # [B*To, 128]
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )

        # Optional temporal GRU encoder, currently disabled.
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

     # Add a batch dimension for a single rollout visual sequence.
     if x.ndim == 4:
        # rollout 单条序列，没有 batch 维
        x = x.unsqueeze(0)
     elif x.ndim != 5:
        raise ValueError(f"x must be 4D or 5D visual tensor, got {x.shape}")

    # 统一成 [B, To, C, H, W]
     # Detect whether the input is already channel-first.
     if x.shape[2] in [1, 3]:
        # already [B, To, C, H, W]
        pass

     # Convert channel-last images into channel-first format.
     elif x.shape[-1] in [1, 3]:
        # [B, To, H, W, C] -> [B, To, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
     else:
        raise ValueError(f"Cannot infer channel dim from shape {x.shape}")

     # Extract batch, temporal, channel, height, and width dimensions.
     B, To, C, H, W = x.shape

    # normalize
     # Convert images to floating-point representation.
     x = x.float()

     # Normalize uint8-style image values from [0, 255] to [0, 1].
     if x.max() > 1:
        x = x / 255.0

    # [B, To, C, H, W] -> [B*To, C, H, W]
     # Merge batch and temporal dimensions so each frame
     # can be processed independently by the CNN.
     x = x.view(B * To, C, H, W)

     # Extract spatial visual features.
     x = self.backbone(x)

     # Project each image into feat_dim.
     feat = self.head(x)                         # [B*To, feat_dim]

     # Restore the temporal sequence dimension.
     feat = feat.view(B, To, self.feature_dim)   # [B, To, feat_dim]

     # Placeholder for the previously considered temporal GRU residual.
     feat = feat #+ self.gru(feat)                # residual GRU

     return feat
