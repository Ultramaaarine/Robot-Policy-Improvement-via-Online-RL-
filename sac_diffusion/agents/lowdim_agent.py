from omegaconf import OmegaConf, DictConfig 
import os 
import torch 
import numpy as np 
import torch.nn as nn 
import hydra 
from pathlib import Path 
from sac_diffusion.agents.base_agent import BaseAgent 
class LowDimAgent(BaseAgent): 
  def __init__(self, cfg): 
      self.cfg = cfg 
      self.policy = hydra.utils.instantiate(cfg.policy)
      self.replay_buffer = hydra.utils.instantiate(cfg)
      super().__init__(cfg) 

  def get_states_from_observation(self,obs): # lowdim(pos,ori,eef) or visual feature for further policy optimization
        assert isinstance(obs,dict)
        if "pos" in obs:
            state = torch.tensor(obs["pos"])
        if "ori" in obs:
            state = torch.tensor(obs["ori"])
        if "joints" in obs:
            state = torch.tensor(obs["joints"])
            
            

  def get_state_dim(self,obs): #读取obs的维度好送到神经网络
        state_dim = 0 
        if "pos" in obs: 
            state_dim += 3 
        if "rgb_gripper" in obs: 
            state_dim += self.encoder.featuresize ##what is this encoder  do we need this in lowdim policy? maybe we can keep it in the code 
        if "pos_ori" in obs:
          state_dim +=6
        return state_dim
  
  
  def get_action_from_policy(self):
       ckpt = self.policy.load_state_dict() #也许可以不全load
       actions:torch.Tensor
       actions = ckpt["action"]
       assert isinstance(actions,torch.Tensor)
       actions.numpy()
       print(actions)
       assert isinstance(actions,list)
       
       return actions

  
  def get_policy_improvement_weight(self,reward): #policy improvement action from actor 
    improvement = self.actor.get_improvement()
    return improvement  
  
  
  def evaluate_policy(self):
      total_reward:float = 0.0
      actions = self.get_action_from_policy()
      obs = self.env.reset()
      for actions_idx,action in enumerate(actions):
        next_obs,reward,info,done =  self.env.step(action)
        total_reward += reward

      return total_reward

      
  def play_step(self):
     actions = self.get_action_from_policy()
     episode=[]
     total_rewards = 0.0
     done = False
     obs = self.env.reset()
     
     current_step:int = 0 
     for action_idx,action in enumerate(actions): #到了 done 或者 total_steps 就结束这个 episode
         next_obs, reward, done, info = self.env.play_step(action)
         total_rewards += reward
         
         if current_step == self.cfg.total_step:
          done = True
         
         transition = {
          "current_step":current_step,
          "obs":obs,#obs 包括state  这里还是要区分一下 robot state 和 rgb obs
          "action":action[current_step],
          "next_obs":next_obs,
          "rewards":reward,
          "info":info,
          "done":done
          }
         episode.append(transition)
         self._save_to_replay_buffer(transition)
         obs = next_obs
         current_step +=1
         
     return transition
       

  def _save_to_replay_buffer(self,transition):
      
      self.replay_buffer.save_transitions(transition)
  


#   def evaluate_policy_episode(self, rewards):
    
#      episode_obs = []
#      episode_rewards = 0.0
#      done = False
#      rewards_per_episode = []
#      obs = self.env.reset()
#      episode_obs.append(obs)
#      current_step:int = 0
#      for step in range(self.cfg.total_steps):
#          obs, reward, done, info = self.env.play_step()
#          episode_rewards += reward
#          episode_obs.append(obs)
#          current_step +=1
#          if current_step % self.cfg.every_steps == 0:
#              rewards_per_episode.append(episode_rewards)

            
#          if done:
#              break
        
#      return episode_obs, episode_rewards, done