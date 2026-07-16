from typing import Dict
import torch
import torch.nn as nn
from sac_diffusion.models.mixin import Mixin
from sac_diffusion.models.normalizer import Normalizer

class BasePolicy(Mixin):
  
  
    
  
  def sampling_policy_from_demos(self):
  
   raise NotImplementedError
  
  def reset(self):
    pass
  
  def set_normalizer(self):
    raise NotImplementedError