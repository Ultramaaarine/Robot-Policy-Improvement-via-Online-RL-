import torch.nn as nn
import torch
from sac_diffusion.utils.robot_mixed_obs_encoder import Robot_Mixed_Obs_Encoder
from typing import Optional


class ValueNet(nn.Module):
  def __init__(self,encoder_output_dim,cond_dim,middle_dim,output_dim,num_gru_layers,obs_type:str):
    super().__init__()
    if obs_type == "lowdim_obs":
     self.obs_encoder = Robot_Mixed_Obs_Encoder(
                                    use_visual_obs_only = False,
                                    cond_dim = cond_dim,
                                    low_dim_obs_encoder_out_dim = encoder_output_dim, #128 according to yaml
                                    visual_obs_feature_dim = middle_dim,   #256
                                    enable_visual_obs = False,
                                    combine_obs = False
                                              )
    elif obs_type == "visual_obs_included":
     self.obs_encoder = Robot_Mixed_Obs_Encoder(
       use_visual_obs_only=True,
       cond_dim = cond_dim,
       low_dim_obs_encoder_out_dim = encoder_output_dim,
       visual_obs_feature_dim=0,
       enable_visual_obs=True,
       combine_obs=False
    )
    
    self.gru = nn.GRU(
      input_size=encoder_output_dim,
      hidden_size=encoder_output_dim,
      bidirectional=False,
      num_layers=num_gru_layers,
      batch_first=True
    ) 
    self.fc_layer = nn.Sequential(
      nn.Linear(encoder_output_dim,middle_dim),
      nn.ReLU(),
      nn.Linear(middle_dim,output_dim), # output_dim = 1 

    )
  
  def forward(self, low_dim_obs:Optional[torch.Tensor],visual_obs:Optional[torch.Tensor]= None):
    encoded_obs = self.obs_encoder(
                                    low_dim_obs=low_dim_obs, visual_obs =visual_obs  )
    out,_ = self.gru.forward(encoded_obs)
    out = self.fc_layer(out)
   
    return out