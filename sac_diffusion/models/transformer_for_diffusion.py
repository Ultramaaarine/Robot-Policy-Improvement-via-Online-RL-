import torch
import torch.nn as nn
from typing import Literal, Optional
from sac_diffusion.utils.positional_embedder import SinusoidalEmbbeder, PositionalEmbeder
from sac_diffusion.models.mixed_obs_MLP import Mixed_Obs_MLP


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        enable_visual_obs: bool,
        low_dim_obs_encoder_out_dim: int,
        use_visual_obs_only: bool,
        diffusion_input_dim: int,
        visual_obs_feature_dim: int,
        combine_obs: bool,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        use_time_mlp: bool = True,
        num_stages: int = 3,
        stage_emb_dim: int = 32,
        cond_num_layers: int = 1,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_stages = int(num_stages)
        self.stage_emb_dim = int(stage_emb_dim)

        self.enable_visual_obs = enable_visual_obs
        self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim
        self.visual_obs_feature_dim = visual_obs_feature_dim
        self.diffusion_input_dim = diffusion_input_dim
        self.use_visual_obs_only = use_visual_obs_only
        self.combine_obs = combine_obs

        # input projection
        self.input_proj = nn.Linear(self.action_dim, self.d_model)

        # cond encoder -> [B,T,d_model]
        self.obs_embeder = Mixed_Obs_MLP(
            self.d_model,
            self.cond_dim,
            self.enable_visual_obs,
            self.low_dim_obs_encoder_out_dim,
            self.visual_obs_feature_dim,
            self.diffusion_input_dim,
            self.use_visual_obs_only,
            self.combine_obs
        )

        self.pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )
        self.cond_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )
        self.action_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )


        # diffusion timestep emb
        self.time_emb = SinusoidalEmbbeder(self.d_model)
        if use_time_mlp:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.SiLU(),
                nn.Linear(self.d_model * 4, self.d_model),
            )
        else:
            self.time_mlp = nn.Identity()

        # stage embedding (Learned Positional Embedding) not same as positional embedding
        self.stage_emb = nn.Embedding(self.num_stages, self.stage_emb_dim) # get 3 stages, for each stage, we embded it as a vector [1,2,3]-> [vector,vector,vector]
        
        self.stage_proj = nn.Sequential(
            nn.Linear(self.stage_emb_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        cond_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.cond_encoder = nn.TransformerEncoder(
            cond_encoder_layer,
            num_layers=int(cond_num_layers)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )


        encoder_layer = nn.TransformerEncoderLayer(
               d_model=self.d_model,
               nhead=nhead,
               dim_feedforward=dim_feedforward,
               dropout=dropout,
               activation="gelu",
               batch_first=True,
               norm_first=True,
         )
        self.transformer = nn.TransformerEncoder(
               encoder_layer,
               num_layers=num_layers
        ) # assembel a transformer with muitiple nn.TRansformerEncoderLayer
        
        self.final_norm = nn.LayerNorm(self.d_model)
        
        self.stage_head = nn.Linear(self.d_model,self.num_stages)
        self.out_proj_layer = nn.Linear(self.d_model, self.action_dim)

    def _to_bool_padding_mask(
        self,
        mask: Optional[torch.Tensor],
        shape: tuple[int, int],
        device: torch.device,
        name: str,
    ) -> Optional[torch.Tensor]:
        """
        Padding mask convention:
            True  = padding / ignore
            False = valid
        """
        if mask is None:
            return None

        if mask.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {mask.shape}"
            )

        return mask.to(device=device, dtype=torch.bool)

    def _zero_padded_tokens(
        self,
        tokens: torch.Tensor,                  # [B, T, D]
        padding_mask: Optional[torch.Tensor],  # [B, T], True = padding
    ) -> torch.Tensor:
        if padding_mask is None:
            return tokens

        return tokens.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

    def _build_action_stage_feat(
        self,
        stage: Optional[torch.Tensor],
        B: int,
        Ta: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        stage should align with future action horizon:
            stage: [B, Ta], [B, Ta, 1], or [B, 1]
        """
        if stage is None:
            stage_seq = torch.zeros(
                (B, Ta),
                device=device,
                dtype=torch.long,
            )
        else:
            if stage.dim() == 3 and stage.shape[-1] == 1:
                stage_seq = stage[..., 0]
            elif stage.dim() == 2:
                stage_seq = stage
            else:
                raise ValueError(f"Unexpected stage shape: {stage.shape}")

            if stage_seq.shape[0] != B:
                raise ValueError(
                    f"stage batch {stage_seq.shape[0]} != action batch {B}"
                )

            if stage_seq.shape[1] == 1:
                stage_seq = stage_seq.expand(B, Ta)

            if stage_seq.shape != (B, Ta):
                raise ValueError(
                    f"stage must broadcast to [B, Ta], got {stage_seq.shape}, "
                    f"expected {(B, Ta)}. "
                    f"Remember: stage should align with future action indices."
                )

            stage_seq = stage_seq.to(device=device)
            stage_seq = stage_seq.long().clamp(
                min=0,
                max=self.num_stages - 1,
            )

        stage_feat = self.stage_emb(stage_seq)       # [B, Ta, stage_emb_dim]
        stage_feat = self.stage_proj(stage_feat)     # [B, Ta, d_model]
        return stage_feat

    def forward(
        self,
        x: torch.Tensor,                    # [B, Ta, action_dim], noisy action
        low_dim_obs: torch.Tensor,          # [B, To, cond_dim]
        visual_obs: Optional[torch.Tensor], # [B, To, C, H, W] or None
        diffusion_t: torch.Tensor,          # [B]
        stage: Optional[torch.Tensor] = None,  # future action stage: [B, Ta, 1] / [B, Ta] / [B, 1]
        mode: Literal["sequence", "global"] = "sequence",


        # new names
        cond_padding_mask: Optional[torch.Tensor] = None,     # [B, To], True = padded cond token
        action_padding_mask: Optional[torch.Tensor] = None,  # [B, Ta], True = padded action token

        return_stage_logits: bool = False,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(f"x must be [B, Ta, action_dim], got {x.shape}")
        if x.shape[-1] != self.action_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} != action_dim {self.action_dim}"
            )

        if low_dim_obs.ndim != 3:
            raise ValueError(
                f"low_dim_obs must be [B, To, cond_dim], got {low_dim_obs.shape}"
            )
        if low_dim_obs.shape[-1] != self.cond_dim:
            raise ValueError(
                f"low_dim_obs last dim {low_dim_obs.shape[-1]} != cond_dim {self.cond_dim}"
            )

        B, Ta, _ = x.shape
        B_obs, To, _ = low_dim_obs.shape
        device = x.device

        if B_obs != B:
            raise ValueError(
                f"Batch mismatch: x batch={B}, low_dim_obs batch={B_obs}"
            )


        obs_padding_mask = self._to_bool_padding_mask(
            cond_padding_mask,
            shape=(B, To),
            device=device,
            name="obs_padding_mask",
        )
        #print(f"obs_padding_mask is {obs_padding_mask}")
        action_padding_mask = self._to_bool_padding_mask(
            action_padding_mask,
            shape=(B, Ta),
            device=device,
            name="action_padding_mask",
        )
        #print(f"action_padding_mask is {action_padding_mask}")
        # ------------------------------------------------------------
        # 1. cond / obs tokens
        # ------------------------------------------------------------
        encoded_cond = self.obs_embeder(
            visual_obs=visual_obs,
            low_dim_obs=low_dim_obs,
            time_emb=diffusion_t,
        )  # [B, To, d_model]

        if encoded_cond.ndim != 3:
            raise ValueError(
                f"encoded_cond must be [B, To, d_model], got {encoded_cond.shape}"
            )
        if encoded_cond.shape != (B, To, self.d_model):
            raise ValueError(
                f"encoded_cond shape {encoded_cond.shape} != expected {(B, To, self.d_model)}"
            )

        # padding token 先置零
        encoded_cond = self._zero_padded_tokens(
            encoded_cond,
            obs_padding_mask,
        )

        # cond positional embedding
        encoded_cond = self.cond_pos_emb(encoded_cond)

        # cond encoder
        memory = self.cond_encoder(
            encoded_cond,
            src_key_padding_mask=obs_padding_mask,
        )  # [B, To, d_model]

        # ------------------------------------------------------------
        # 2. noisy action tokens
        # ------------------------------------------------------------
        action_tokens = self.input_proj(x)  # [B, Ta, d_model]

        # add future action stage embedding
        stage_feat = self._build_action_stage_feat(
            stage=stage,
            B=B,
            Ta=Ta,
            device=device,
        )
        action_tokens = action_tokens + stage_feat

        # padding action token 置零
        action_tokens = self._zero_padded_tokens(
            action_tokens,
            action_padding_mask,
        )

        # action positional embedding
        action_tokens = self.action_pos_emb(action_tokens)

        # ------------------------------------------------------------
        # 3. decoder: action tokens cross-attend to cond memory
        # ------------------------------------------------------------
        action_tokens = self.action_decoder(
            tgt=action_tokens,
            memory=memory,
            tgt_key_padding_mask=action_padding_mask,
            memory_key_padding_mask=obs_padding_mask,
        )  # [B, Ta, d_model]

        action_tokens = self.final_norm(action_tokens)

        # ------------------------------------------------------------
        # 4. output
        # ------------------------------------------------------------
        stage_logits = self.stage_head(action_tokens)      # [B, Ta, num_stages]
        action_out = self.out_proj_layer(action_tokens)    # [B, Ta, action_dim]

        if return_stage_logits:
            return action_out, stage_logits

        if mode == "sequence":
            return action_out

        elif mode == "global":
            if action_padding_mask is None:
                return action_out.mean(dim=1)

            valid = (~action_padding_mask).float().unsqueeze(-1)  # [B, Ta, 1]
            denom = valid.sum(dim=1).clamp_min(1.0)
            return (action_out * valid).sum(dim=1) / denom

        else:
            raise ValueError(f"Unsupported mode: {mode}")
  