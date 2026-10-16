# Encode mixed low-dimensional and optional visual observations together with
# the diffusion timestep, then project them into the diffusion condition space.

#1. step-wise policy, not sequential policy like "diffusion policy visuomotor policy" algorithm
#2. use MLP as network
#3. input dim = [B*T,D] flattened [B,T,D]
#4. for each interaction step use sampling to generate an action
#5. obs are first normalized

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict
from typing import List, Optional
from sac_diffusion.utils.robot_mixed_obs_encoder import Robot_Mixed_Obs_Encoder
from sac_diffusion.utils.positional_embedder import SinusoidalEmbbeder
from sac_diffusion.utils.obs_wrapper import OBSWrapper 

class Mixed_Obs_MLP(nn.Module):
 def __init__(self,           
              pos_emb_dim,

              cond_dim: Optional[int],

              
              enable_visual_obs,
              low_dim_obs_encoder_out_dim,
              visual_obs_feature_dim:Optional[int],
              diffusion_input_dim,
              use_visual_obs_only:bool, 
              combine_obs:bool, 
              middle_dim = 256
              ):   # input: tensor [B*T,D] as action cond with cond action
  super().__init__()

  # Dimension of the low-dimensional condition input.
  self.cond_dim = cond_dim # dim of lowdim_condition

  # Dimension of the diffusion timestep embedding.
  self.t_emb_dim = pos_emb_dim

  # Output dimension of the low-dimensional observation encoder.
  self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim # dim of encoded lowdim_obs

  # Whether visual observations are enabled.
  self.enable_visual_obs = enable_visual_obs

  # Feature dimension of encoded visual observations.
  if self.enable_visual_obs:
     self.visual_obs_feature_dim = visual_obs_feature_dim # dim of encoded visual_obs
  elif not self.enable_visual_obs:
     self.visual_obs_feature_dim = 0

  # Final feature dimension passed into the diffusion model.
  self.diffusion_input_dim = diffusion_input_dim

  # Whether low-dimensional and visual observation features are combined.
  self.combine_obs = combine_obs

  # Whether only visual observations are used as condition input.
  self.use_visual_obs_only = use_visual_obs_only

  # Mixed observation encoder for low-dimensional and optional visual inputs.
  self.obs_encoder = Robot_Mixed_Obs_Encoder(        
              self.use_visual_obs_only, 
              self.cond_dim, # dim of lowdim_cond
              self.low_dim_obs_encoder_out_dim,
              self.visual_obs_feature_dim,
              self.enable_visual_obs, 
              self.combine_obs, 
              )

  # Sinusoidal embedding for the diffusion timestep.
  self.timestep_embedder = SinusoidalEmbbeder(self.t_emb_dim)
  
  # Fuse observation features with diffusion timestep embedding
  # and project them into the diffusion condition dimension.
  self.mlp = nn.Sequential(
   nn.Linear(self.low_dim_obs_encoder_out_dim + self.visual_obs_feature_dim + self.t_emb_dim, middle_dim),
   nn.SiLU(),
   nn.Linear(middle_dim,middle_dim),
   nn.SiLU(),
   nn.Linear(middle_dim,self.diffusion_input_dim), # diffusion input_dim is 256
   nn.SiLU()
  )
  
 def forward(self,visual_obs,low_dim_obs,time_emb): # x is noisy action obs in this nn are normalized, time_emb is [B,] timestep tensor

        # Encode low-dimensional and optional visual observations.
        encoded_cond = self.obs_encoder.forward(visual_obs=visual_obs,low_dim_obs=low_dim_obs)

        #print(f"encoded cond has shape:{encoded_cond.shape}") # [1,64,128]

        # Extract batch size and temporal length.
        B, T, _ = encoded_cond.shape

        #print(f"time emb has shape: {time_emb.shape}")

        # Convert scalar diffusion timesteps into embedding vectors.
        timestep_emb = self.timestep_embedder(time_emb)

        # Broadcast the timestep embedding across all observation timesteps.
        timestep_emb = timestep_emb[:, None, :].expand(B, T, -1)

        #print(f"timestep_emb vec has shape {timestep_emb.shape}")
        #print(f"encoded cond has shape {encoded_cond.shape}")

        # Concatenate observation features and timestep embedding.
        h = torch.cat([encoded_cond,timestep_emb],dim = -1)

        # Project the fused condition into diffusion input feature space.
        pred = self.mlp(h)

        #print(f"ObsMLP output has shape: {pred.shape}")

        return pred
