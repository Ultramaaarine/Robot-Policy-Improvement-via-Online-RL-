import torch.nn as nn
import torch
import numpy as np
from sac_diffusion.utils.robot_mixed_obs_encoder import Robot_Mixed_Obs_Encoder


#Feature-wise Linear Modulation
class LowDimModulator(nn.Module): # FiLM block, modulate encoded robot obs 
  def __init__(self, cond_dim, cond_output_dim,action_dim,gru_num_layers): # modulation_dim = 2*encode_input_dim
    super().__init__()
    self.lowdim_obs_encoder = Robot_Mixed_Obs_Encoder(
                                                     
                                                     use_visual_obs_only=False,
                                                     cond_dim=cond_dim,
                                                     low_dim_obs_encoder_out_dim=cond_output_dim, # cond_output_dim in forward()
                                                     visual_obs_feature_dim=0,
                                                     enable_visual_obs=False,
                                                     combine_obs=False)
    self.fc = nn.Sequential(
      nn.Linear(cond_output_dim,2*action_dim),
      nn.SiLU()
      )
    self.gru = nn.GRU(
      input_size=cond_output_dim,
      num_layers=gru_num_layers,
      hidden_size=cond_output_dim,
      bidirectional=False,
      batch_first=True
      )
    
    
  def forward(self,obs,visual_obs):
    encoded_obs = self.lowdim_obs_encoder(               
                                          visual_obs = visual_obs, # None
                                          low_dim_obs = obs)
    encoded_obs,_ = self.gru(encoded_obs)
    

    return encoded_obs
  
class VisualModulator(nn.Module): # 
  def __init__(self,cond_dim, cond_output_dim,action_dim,gru_num_layers):#encode_output_dim is 128
    super().__init__()
    self.fc = nn.Sequential(
      nn.Linear(cond_output_dim,2*action_dim),
      nn.SiLU()
      )
    self.visual_obs_encoder = Robot_Mixed_Obs_Encoder(use_visual_obs_only=False,
                                                     
                                                     cond_dim=cond_dim,
                                                     low_dim_obs_encoder_out_dim=cond_output_dim,
                                                     visual_obs_feature_dim=0,
                                                     enable_visual_obs=True,
                                                     combine_obs=True)
    self.gru = nn.GRU(
      input_size=cond_output_dim,
      num_layers=gru_num_layers,
      hidden_size=cond_output_dim,
      bidirectional=False,
      batch_first=True
      )
    
  def forward(self,lowdim_obs,visual_obs):
   encoded_obs = self.visual_obs_encoder(visual_obs = visual_obs,low_dim_obs = lowdim_obs) # [B,T,encoded_dim]
   encoded_obs,_ = self.gru(encoded_obs) # [B,T,encoded_dim]

   return encoded_obs

#Q-value network  
class MLPNetwork(nn.Module): 
 def __init__(self, input_dim,middle_dim,output_dim):
    super().__init__()
    self.input_dim = input_dim
    self.out_put_dim = output_dim
    self.first_layer = nn.Sequential(
      nn.Linear(input_dim,middle_dim),
      nn.ReLU(),
      nn.Linear(middle_dim,middle_dim),
      nn.ReLU()
    )
    self.second_layer = nn.Linear(middle_dim,output_dim)
     
    
 def forward(self,modulated_input):
   
   fc1_out = self.first_layer(modulated_input)
   out = self.second_layer(fc1_out)
   return out

# Double Q-value network
class DoubleQNetwork(nn.Module):
 def __init__(self, cond_output_dim, gru_num_layers, mlp_middle_dim, mlp_output_dim, obs_type:str, action_dim, cond_dim):
  super().__init__()
  if obs_type == "visual_obs_included":
   self.modulator = VisualModulator(cond_dim, cond_output_dim, action_dim,gru_num_layers=gru_num_layers) # modulation_dim = 2*action_dim
  elif obs_type == "lowdim_obs":
   self.modulator = LowDimModulator(cond_dim, cond_output_dim, action_dim,gru_num_layers=gru_num_layers) #这里的 input_dim要和 agent的 get obs dim 配合 具体要看replaybuffer 里面是什么
  
  self.fc_layer = nn.Linear(cond_output_dim,action_dim*2)

  mlp_input_dim = 2*action_dim

  self.q1_network = MLPNetwork(mlp_input_dim,mlp_middle_dim,mlp_output_dim)# mlp_input_dim = action_dim + modulation_dim [B, D+M]
  self.q2_network = MLPNetwork(mlp_input_dim,mlp_middle_dim,mlp_output_dim)

 def forward(self, state, visual_obs, action):
   
   modulated_obs = self.modulator(state, visual_obs) # mlp outdim: 1
   gamma_beta = self.fc_layer(modulated_obs)
   gamma,beta = gamma_beta.chunk(2,dim = -1)

   modulated_action = gamma*action + beta 
   modulated_input = torch.cat([action,modulated_action],dim = -1)
   
    # dim of modulated_obs = dim of action

   q1 = self.q1_network(modulated_input) # sample wise q value 
   
   q2 = self.q2_network(modulated_input)
   
 
   return  q1,q2
 
