import numpy as np
import torchvision
import torch
from sac_diffusion.models.normalizer import Normalizer


class VisualOBSNormalizer():
  def __init__(self):
    self.device = "cuda:0"
    
  def normalize(self,x):
    obs_tensor = torch.from_numpy(x).to(self.device)
    obs_tensor = obs_tensor.reshape(2,0,1) # [H,W,C] -> [C,H,W]
    obs_tensor = obs_tensor.unsqueeze(0) # [C H W]->[1 C H W]
    obs_tensor = obs_tensor.float()/255

    #Image_net normalization
    mean = torch.tensor([0.485, 0.456, 0.406],device=self.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225],device=self.device).view(1,3,1,1)
    n_obs_tensor = (obs_tensor-mean)/std
    return n_obs_tensor

class DepthOBSNormalizer():
  def __init__(self,depth_min,depth_max):
    self.depth_max = torch.from_numpy(depth_max).to(device="cuda:0")
    self.depth_min = torch.from_numpy(depth_min).to(device="cuda:0")
    
  def normalize_depth_obs(self, depth):
    depth_tensor = torch.from_numpy(depth).to(device="cuda:0")
    clamped_depth_tensor = torch.clamp(depth_tensor,min=self.depth_min,max=self.depth_max) 
    n_depth_tensor  = (clamped_depth_tensor-self.depth_min)/(self.depth_max - self.depth_min)
    return n_depth_tensor

class OBSNormalizer():
  def __init__(self,dim):
      self.dim = dim
      self.low_dim_normalizer = Normalizer()
  def normalize_low_dim_obs(self,low_dim_obs,params):
        
        n_low_dim_obs = self.low_dim_normalizer.normalize(low_dim_obs,params)
        return n_low_dim_obs # tensor

  def normalize_visual_obs(self,visual_obs):
   rgb_obs_normalizer = VisualOBSNormalizer() 
   n_visual_obs = rgb_obs_normalizer.normalize(visual_obs)
   return n_visual_obs # tensor
   
  def normalize_depth_obs(self,depth_obs,depth_max,depth_min):  
    depth_obs_normalizer = DepthOBSNormalizer(depth_max,depth_min)
    n_depth_obs = depth_obs_normalizer.normalize_depth_obs(depth_obs)
    return n_depth_obs # tensor