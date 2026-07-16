import hydra
import torch
import numpy as np
import torch.nn as nn

class BaseWorkspace():
 def __init__(self,cfg,output_dir):
  self.cfg = cfg
  self._output_dir = output_dir
  
 
 def run():
  """
  
  """
  raise NotImplementedError
 
 def save_model(self):
  
  raise NotImplementedError
 
 def save_check_point(self):
  
  raise NotImplementedError