# Transformer-based diffusion model for action denoising.
# Observation features are encoded as condition memory, while noisy action tokens
# cross-attend to this memory together with stage and positional embeddings.

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

        # Basic model dimensions.
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_stages = int(num_stages)
        self.stage_emb_dim = int(stage_emb_dim)

        # Observation encoder configuration.
        self.enable_visual_obs = enable_visual_obs
        self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim
        self.visual_obs_feature_dim = visual_obs_feature_dim
        self.diffusion_input_dim = diffusion_input_dim
        self.use_visual_obs_only = use_visual_obs_only
        self.combine_obs = combine_obs

        # input projection
        # Project noisy action vectors into Transformer token dimension.
        self.input_proj = nn.Linear(self.action_dim, self.d_model)

        # cond encoder -> [B,T,d_model]
        # Encode low-dimensional and optional visual observations into condition tokens.
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

        # Generic positional embedding.
        self.pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )

        # Positional embedding for observation / condition tokens.
        self.cond_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )

        # Positional embedding for future action tokens.
        self.action_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )


        # diffusion timestep emb
        # Encode the diffusion timestep into d_model dimensions.
        self.time_emb = SinusoidalEmbbeder(self.d_model)

        # Optional MLP applied to the diffusion timestep embedding.
        if use_time_mlp:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.SiLU(),
                nn.Linear(self.d_model * 4, self.d_model),
            )
        else:
            self.time_mlp = nn.Identity()

        # stage embedding (Learned Positional Embedding) not same as positional embedding
        # Learn one embedding vector for each task stage.
        self.stage_emb = nn.Embedding(self.num_stages, self.stage_emb_dim) # get 3 stages, for each stage, we embded it as a vector [1,2,3]-> [vector,vector,vector]
        
        # Project stage embedding into Transformer model dimension.
        self.stage_proj = nn.Sequential(
            nn.Linear(self.stage_emb_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Transformer encoder for observation / condition tokens.
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

        # Transformer decoder for noisy action tokens.
        # Action tokens cross-attend to encoded observation memory.
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


        # Additional Transformer encoder definition.
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
        
        # Normalize decoder features before output heads.
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Predict task-stage logits from action-token features.
        self.stage_head = nn.Linear(self.d_model,self.num_stages)

        # Project Transformer features back to action dimension.
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

        # No padding mask provided.
        if mask is None:
            return None

        # Validate mask shape against the expected token sequence shape.
        if mask.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {mask.shape}"
            )

        # Transformer padding masks must be boolean tensors on the correct device.
        return mask.to(device=device, dtype=torch.bool)

    def _zero_padded_tokens(
        self,
        tokens: torch.Tensor,                  # [B, T, D]
        padding_mask: Optional[torch.Tensor],  # [B, T], True = padding
    ) -> torch.Tensor:

        # If no padding exists, keep tokens unchanged.
        if padding_mask is None:
            return tokens

        # Explicitly zero all padded token embeddings.
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

        # If no stage information is provided, default all actions to stage 0.
        if stage is None:
            stage_seq = torch.zeros(
                (B, Ta),
                device=device,
                dtype=torch.long,
            )

        else:
            # Accept either [B, Ta, 1] or [B, Ta].
            if stage.dim() == 3 and stage.shape[-1] == 1:
                stage_seq = stage[..., 0]
            elif stage.dim() == 2:
                stage_seq = stage
            else:
                raise ValueError(f"Unexpected stage shape: {stage.shape}")

            # Stage batch dimension must match action batch size.
            if stage_seq.shape[0] != B:
                raise ValueError(
                    f"stage batch {stage_seq.shape[0]} != action batch {B}"
                )

            # Broadcast one stage label across the entire action horizon.
            if stage_seq.shape[1] == 1:
                stage_seq = stage_seq.expand(B, Ta)

            # Final stage tensor must align with every future action token.
            if stage_seq.shape != (B, Ta):
                raise ValueError(
                    f"stage must broadcast to [B, Ta], got {stage_seq.shape}, "
                    f"expected {(B, Ta)}. "
                    f"Remember: stage should align with future action indices."
                )

            # Convert stage IDs to valid embedding indices.
            stage_seq = stage_seq.to(device=device)
            stage_seq = stage_seq.long().clamp(
                min=0,
                max=self.num_stages - 1,
            )

        # Convert stage IDs into learned stage embeddings.
        stage_feat = self.stage_emb(stage_seq)       # [B, Ta, stage_emb_dim]

        # Project stage embeddings to d_model.
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

        # Validate noisy action input.
        if x.ndim != 3:
            raise ValueError(f"x must be [B, Ta, action_dim], got {x.shape}")

        if x.shape[-1] != self.action_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} != action_dim {self.action_dim}"
            )

        # Validate low-dimensional observation sequence.
        if low_dim_obs.ndim != 3:
            raise ValueError(
                f"low_dim_obs must be [B, To, cond_dim], got {low_dim_obs.shape}"
            )

        if low_dim_obs.shape[-1] != self.cond_dim:
            raise ValueError(
                f"low_dim_obs last dim {low_dim_obs.shape[-1]} != cond_dim {self.cond_dim}"
            )

        # Extract batch size and temporal dimensions.
        B, Ta, _ = x.shape
        B_obs, To, _ = low_dim_obs.shape
        device = x.device

        # Observation batch and action batch must match.
        if B_obs != B:
            raise ValueError(
                f"Batch mismatch: x batch={B}, low_dim_obs batch={B_obs}"
            )


        # Convert observation padding mask to Transformer-compatible boolean format.
        obs_padding_mask = self._to_bool_padding_mask(
            cond_padding_mask,
            shape=(B, To),
            device=device,
            name="obs_padding_mask",
        )

        #print(f"obs_padding_mask is {obs_padding_mask}")

        # Convert action padding mask to Transformer-compatible boolean format.
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

        # Fuse and encode low-dimensional and optional visual observations.
        encoded_cond = self.obs_embeder(
            visual_obs=visual_obs,
            low_dim_obs=low_dim_obs,
            time_emb=diffusion_t,
        )  # [B, To, d_model]

        # Verify observation encoder output shape.
        if encoded_cond.ndim != 3:
            raise ValueError(
                f"encoded_cond must be [B, To, d_model], got {encoded_cond.shape}"
            )

        if encoded_cond.shape != (B, To, self.d_model):
            raise ValueError(
                f"encoded_cond shape {encoded_cond.shape} != expected {(B, To, self.d_model)}"
            )

        # padding token 先置零
        # Remove feature values from padded observation tokens.
        encoded_cond = self._zero_padded_tokens(
            encoded_cond,
            obs_padding_mask,
        )

        # cond positional embedding
        # Inject temporal position information into condition tokens.
        encoded_cond = self.cond_pos_emb(encoded_cond)

        # cond encoder
        # Encode the complete observation sequence into Transformer memory.
        memory = self.cond_encoder(
            encoded_cond,
            src_key_padding_mask=obs_padding_mask,
        )  # [B, To, d_model]

        # ------------------------------------------------------------
        # 2. noisy action tokens
        # ------------------------------------------------------------

        # Project noisy actions into Transformer token space.
        action_tokens = self.input_proj(x)  # [B, Ta, d_model]

        # add future action stage embedding
        # Add stage-dependent features aligned with each future action token.
        stage_feat = self._build_action_stage_feat(
            stage=stage,
            B=B,
            Ta=Ta,
            device=device,
        )

        action_tokens = action_tokens + stage_feat

        # padding action token 置零
        # Remove feature values from padded action tokens.
        action_tokens = self._zero_padded_tokens(
            action_tokens,
            action_padding_mask,
        )

        # action positional embedding
        # Inject future action timestep information.
        action_tokens = self.action_pos_emb(action_tokens)

        # ------------------------------------------------------------
        # 3. decoder: action tokens cross-attend to cond memory
        # ------------------------------------------------------------

        # Decode noisy action tokens while cross-attending to observation memory.
        action_tokens = self.action_decoder(
            tgt=action_tokens,
            memory=memory,
            tgt_key_padding_mask=action_padding_mask,
            memory_key_padding_mask=obs_padding_mask,
        )  # [B, Ta, d_model]

        # Final feature normalization before prediction heads.
        action_tokens = self.final_norm(action_tokens)

        # ------------------------------------------------------------
        # 4. output
        # ------------------------------------------------------------

        # Optional stage prediction for every action token.
        stage_logits = self.stage_head(action_tokens)      # [B, Ta, num_stages]

        # Predicted denoised action / diffusion output.
        action_out = self.out_proj_layer(action_tokens)    # [B, Ta, action_dim]

        # Return stage logits together with action prediction when requested.
        if return_stage_logits:
            return action_out, stage_logits

        # Return one action prediction for every future timestep.
        if mode == "sequence":
            return action_out

        # Aggregate the temporal action output into one global representation.
        elif mode == "global":

            # Simple mean pooling when no padding mask exists.
            if action_padding_mask is None:
                return action_out.mean(dim=1)

            # Masked mean pooling over valid action timesteps.
            valid = (~action_padding_mask).float().unsqueeze(-1)  # [B, Ta, 1]
            denom = valid.sum(dim=1).clamp_min(1.0)
            return (action_out * valid).sum(dim=1) / denom

        else:
            raise ValueError(f"Unsupported mode: {mode}")
