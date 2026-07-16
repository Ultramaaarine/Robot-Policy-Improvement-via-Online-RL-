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

  self.cond_dim = cond_dim # dim of lowdim_condition
  self.t_emb_dim = pos_emb_dim
  self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim # dim of encoded lowdim_obs
  self.enable_visual_obs = enable_visual_obs
  if self.enable_visual_obs:
     self.visual_obs_feature_dim = visual_obs_feature_dim # dim of encoded visual_obs
  elif not self.enable_visual_obs:
     self.visual_obs_feature_dim = 0
  self.diffusion_input_dim = diffusion_input_dim
  self.combine_obs = combine_obs
  self.use_visual_obs_only = use_visual_obs_only
  self.obs_encoder = Robot_Mixed_Obs_Encoder(        
              self.use_visual_obs_only, 
              self.cond_dim, # dim of lowdim_cond
              self.low_dim_obs_encoder_out_dim,
              self.visual_obs_feature_dim,
              self.enable_visual_obs, 
              self.combine_obs, 
              )
  self.timestep_embedder = SinusoidalEmbbeder(self.t_emb_dim)
  
  self.mlp = nn.Sequential(
   nn.Linear(self.low_dim_obs_encoder_out_dim + self.visual_obs_feature_dim + self.t_emb_dim, middle_dim),
   nn.SiLU(),
   nn.Linear(middle_dim,middle_dim),
   nn.SiLU(),
   nn.Linear(middle_dim,self.diffusion_input_dim), # diffusion input_dim is 256
   nn.SiLU()
  )
  
 def forward(self,visual_obs,low_dim_obs,time_emb): # x is noisy action obs in this nn are normalized, time_emb is [B,] timestep tensor
        encoded_cond = self.obs_encoder.forward(visual_obs=visual_obs,low_dim_obs=low_dim_obs)
        #print(f"encoded cond has shape:{encoded_cond.shape}") # [1,64,128]
        B, T, _ = encoded_cond.shape
        #print(f"time emb has shape: {time_emb.shape}")
        timestep_emb = self.timestep_embedder(time_emb)
        timestep_emb = timestep_emb[:, None, :].expand(B, T, -1)
        #print(f"timestep_emb vec has shape {timestep_emb.shape}")
        #print(f"encoded cond has shape {encoded_cond.shape}")
        h = torch.cat([encoded_cond,timestep_emb],dim = -1)
        pred = self.mlp(h)
        #print(f"ObsMLP output has shape: {pred.shape}")
        return pred
   
