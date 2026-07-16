from omegaconf import OmegaConf, DictConfig
import os
import gym
import torch
import numpy as np
import torch.nn as nn
import hydra
from pathlib import Path

class BaseAgent():
 
 def __init__(self,cfg: DictConfig):
  self.env = hydra.utils.instantiate(cfg.env)
  self.policy = hydra.utils.instantiate(cfg.policy)
  self.skill = cfg.skill
  self.actor = hydra.utils.instantiate(cfg.actor)
  self.critic = hydra.utils.instantiate(cfg.critic)
 
 def get_features_from_observation(self):
  
  raise NotImplementedError

 def get_state_dim(self): #何意味
  
  raise NotImplementedError 
 
 def get_action_from_policy(self):
  
  raise NotImplementedError
 
 
 def evaluate_policy(self,rewards):
  
  raise NotImplementedError
 
 def play_step(self):
  
  raise NotImplementedError

 def reset_env(self):
  
  raise NotImplementedError


