# normalize obs before encode 
# send obs to diffusion model as cond for training
import torch
import torch.nn as nn
from typing import Optional
from sac_diffusion.utils.low_dim_obs_encoder import Low_Dim_Obs_Encoder
from sac_diffusion.utils.visual_obs_encoder import VisualEncoder

class Robot_Mixed_Obs_Encoder(nn.Module):
 
 def __init__(self, 
              use_visual_obs_only:bool,              
              cond_dim, #dim of lowdim_condition,or visual condition or combined, can be read from training script 
              low_dim_obs_encoder_out_dim:int,
              visual_obs_feature_dim:int, 
              enable_visual_obs:bool, 
              combine_obs:bool, 
              ):
  
  super().__init__()
  self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim # 128 set in yaml
  self.combine_obs = combine_obs
  self.lowdim_obs_encoder = Low_Dim_Obs_Encoder(                                         
                                                cond_dim, # lowdim_cond before encode
                                                self.low_dim_obs_encoder_out_dim,
                                                                     
                                                )
  self.enable_visual_obs = enable_visual_obs
  if self.enable_visual_obs: # in offline training, visual obs = false in online training, visual obs can be false
     self.feature_dim = visual_obs_feature_dim
     self.visual_obs_encoder = VisualEncoder(self.feature_dim) #self.feature_dim is the output dim of encoded visual obs
  else: 
    self.visual_obs_encoder = None
    self.feature_dim = 0
  #self.obs_wrapper = OBSWrapper()
  self.use_visual_obs_only = use_visual_obs_only
  if self.enable_visual_obs:
   assert (
    self.combine_obs or self.use_visual_obs_only
    ),(
     "when enable_visual_obs == True, combine_obs and use_visual_obs cannot be both False "
    )
  else:
   assert(
     not self.combine_obs and not self.use_visual_obs_only
   ),(
    "Both combine_obs and use_visual_obs_only should be false if use_visual_obs is false"
   )

 def forward(self,visual_obs=None,low_dim_obs=None):
  if not self.enable_visual_obs:
    if low_dim_obs is None:
     
     raise ValueError(
      "low_dim_cond is required when enable_visual_obs is False."
     )

    encoded_low_dim_cond = self.lowdim_obs_encoder.forward(low_dim_obs)#,online_training, online_training_step)  # dim of output: out_dim in class Low_Dim_Obs_Encoder
    return encoded_low_dim_cond
  
  if visual_obs is None:
    raise ValueError(
    " visual_obs is required when enable_visual_obs is True."
   )
  
  encoded_visual_cond = self.visual_obs_encoder(visual_obs)    # dim of output: B x feat_dim=256
  
  
  if self.use_visual_obs_only:
     return encoded_visual_cond
  
  if low_dim_obs is None:
    raise ValueError(
            "low_dim_obs is required when combine_obs is True and "
            "use_visual_obs_only is False."
            )
  encoded_low_dim_cond = self.lowdim_obs_encoder(low_dim_obs)# online_training, online_training_step)

  if self.combine_obs:
      encoded_combined_obs = torch.cat([encoded_low_dim_cond, encoded_visual_cond], dim=-1)
      return encoded_combined_obs # dim = [B, feat_dim+out_dim] 
  
  raise RuntimeError(
          "Invalid flag combination, please check your config: "
            f"enable_visual_obs={self.enable_visual_obs}, "
            f"use_visual_obs_only={self.use_visual_obs_only}, "
            f"combine_obs={self.combine_obs}"
  )
