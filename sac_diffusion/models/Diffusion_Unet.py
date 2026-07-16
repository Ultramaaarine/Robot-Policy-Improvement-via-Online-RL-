# inject positional embedding in every part
import torch.nn as nn
import torch
import torch.nn.functional as F
from sac_diffusion.models.Unet_Modules import DownSamplingBlock, MiddleBlock, UpSamplingBlock
from typing import List, Optional
from sac_diffusion.models.mixed_obs_MLP import Mixed_Obs_MLP
from sac_diffusion.utils.obs_wrapper import OBSWrapper


def _align_len(x: torch.Tensor, ref_len: int) -> torch.Tensor:
    if x.size(-1) == ref_len:
        return x
    diff = ref_len - x.size(-1)
    return F.pad(x, (0, diff)) if diff > 0 else x[..., :ref_len]


class Lowdim_Unet(nn.Module):
    def __init__(
        self,
        dim_in: List[int],
        pos_emb_dim,
        diffusion_input_dim,
        cond_dim: Optional[int],
        enable_cond: Optional[bool],
        enable_visual_obs,
        low_dim_obs_encoder_out_dim,
        visual_obs_feature_dim,
        use_visual_obs_only: bool,
        combine_obs: bool,
        action_dim,
        gate_hidden_dim: Optional[int] = None,
        gate_num_layers: int = 1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.t_emb_dim = pos_emb_dim
        self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim
        self.visual_obs_feature_dim = visual_obs_feature_dim
        self.diffusion_input_dim = diffusion_input_dim
        self.use_visual_obs_only = use_visual_obs_only
        self.enable_visual_obs = enable_visual_obs
        self.combine_obs = combine_obs
        self.action_dim = action_dim

        self.obs_embeder = Mixed_Obs_MLP(
            self.t_emb_dim,
            self.cond_dim,
            self.enable_visual_obs,
            self.low_dim_obs_encoder_out_dim,
            self.visual_obs_feature_dim,
            self.diffusion_input_dim,
            self.use_visual_obs_only,
            self.combine_obs
        )

        if dim_in is None:
            self.dim_in = [32, 64, 128, 256]
        else:
            self.dim_in = dim_in

        assert len(self.dim_in) >= 2

        self.input_projection_layer = nn.Linear(action_dim, self.dim_in[0])

        self.mid_input_dim = self.dim_in[-1]
        self.up_sampling_dim = self.dim_in[::-1]
        self.mid_output_dim = self.mid_input_dim

        downblock = nn.ModuleList([])
        for idx in range(len(self.dim_in) - 1):
            downblock.append(
                DownSamplingBlock(
                    self.dim_in[idx],
                    self.dim_in[idx + 1],
                    self.diffusion_input_dim,
                    enable_cond,
                    kernel_size=3
                )
            )
        self.downblock = downblock

        self.middle_block = MiddleBlock(
            self.mid_input_dim,
            self.mid_output_dim,
            self.diffusion_input_dim,
            enable_cond=enable_cond
        )

        upsampling_block = nn.ModuleList([])
        fuse_convs = nn.ModuleList([])
        for idx in range(len(self.dim_in) - 1):
            cout = self.up_sampling_dim[idx + 1]

            upsampling_block.append(
                UpSamplingBlock(
                    self.up_sampling_dim[idx],
                    self.up_sampling_dim[idx + 1],
                    self.diffusion_input_dim,
                    enable_cond,
                    kernel_size=3
                )
            )

            fuse_convs.append(
                nn.Sequential(
                    nn.Conv1d(2 * cout, cout, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=cout),
                    nn.ReLU(inplace=True)
                )
            )

        self.fuse_convs = fuse_convs
        self.upsampling_block = upsampling_block

        # ===== 2 expert heads =====
        self.out_conv_reach = nn.Conv1d(self.up_sampling_dim[-1], self.dim_in[0], kernel_size=1)
        self.out_conv_pull = nn.Conv1d(self.up_sampling_dim[-1], self.dim_in[0], kernel_size=1)

        self.output_projection_layer_reach = nn.Linear(self.dim_in[0], action_dim)
        self.output_projection_layer_pull = nn.Linear(self.dim_in[0], action_dim)

        # ===== GRU gate =====
        self.gate_hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else self.up_sampling_dim[-1]
        self.gate_num_layers = gate_num_layers

        self.gate_rnn = nn.GRU(
            input_size=self.up_sampling_dim[-1],
            hidden_size=self.gate_hidden_dim,
            num_layers=self.gate_num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.gate_proj = nn.Linear(self.gate_hidden_dim, 1)

        # init
        nn.init.zeros_(self.out_conv_reach.weight)
        nn.init.zeros_(self.out_conv_reach.bias)
        nn.init.zeros_(self.out_conv_pull.weight)
        nn.init.zeros_(self.out_conv_pull.bias)

        # make initial gate around 0.5
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        x,
        pos: torch.Tensor,
        visual_obs: Optional[torch.Tensor],
        low_dim_obs: Optional[torch.Tensor],
        return_aux: bool = False
    ):
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float32, device=x.device)
        elif torch.is_tensor(pos) and len(pos.shape) == 0:
            B = x.shape[0]
            pos = torch.full((B,), int(pos), device=x.device, dtype=torch.float32)

        skips = []
        h = x

        # [B,T,Da] -> [B,T,C0]
        h = self.input_projection_layer(h)
        # [B,T,C] -> [B,C,T]
        h = h.permute(0, 2, 1).contiguous()

        encoded_cond = self.obs_embeder.forward(
            visual_obs=visual_obs,
            low_dim_obs=low_dim_obs,
            time_emb=pos
        )

        # ----- Down -----
        for i, blk in enumerate(self.downblock):
            h = blk(h, encoded_cond)
            if i < len(self.downblock) - 1:
                skips.append(h)

        # ----- Middle -----
        h = self.middle_block(h, encoded_cond)

        # ----- Up -----
        for up_blk, fuse in zip(self.upsampling_block, self.fuse_convs):
            h = up_blk(h, encoded_cond)
            if len(skips) > 0:
                skip = skips.pop()
                h = _align_len(h, skip.size(-1))
                h = torch.cat([h, skip], dim=1)
                h = fuse(h)

        # ===== reach head =====
        out_reach = self.out_conv_reach(h)                         # [B,C,T]
        out_reach = out_reach.permute(0, 2, 1).contiguous()       # [B,T,C]
        out_reach = self.output_projection_layer_reach(out_reach) # [B,T,Da]

        # ===== pull head =====
        out_pull = self.out_conv_pull(h)                          # [B,C,T]
        out_pull = out_pull.permute(0, 2, 1).contiguous()        # [B,T,C]
        out_pull = self.output_projection_layer_pull(out_pull)   # [B,T,Da]

        # ===== GRU gating =====
        h_seq = h.permute(0, 2, 1).contiguous()                  # [B,T,C]
        gate_feat, gate_hidden = self.gate_rnn(h_seq)            # [B,T,H]
        gate_logits = self.gate_proj(gate_feat)                  # [B,T,1]
        gate = torch.sigmoid(gate_logits)                        # [B,T,1]

        # mixed output
        out = out_reach * (1.0 - gate) + out_pull * gate

        if return_aux:
            return {
                "pred": out,
                "gate": gate,
                "gate_logits": gate_logits,
                "gate_feat": gate_feat,
                "gate_hidden": gate_hidden,
                "out_reach": out_reach,
                "out_pull": out_pull,
            }

        return out