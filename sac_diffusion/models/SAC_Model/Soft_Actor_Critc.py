# critic loss(SAC): Q = r + (1-d)*Qtar(s,a),maximize E(||Q-Qtar||**2)
## critic loss(IQL): Q = r + (1-d)*Qtar(s,a),maximize E(||Q-Qtar||**2)
# v_net loss: Q = r + (1-d)*V(s')
import torch 
import torch.nn as nn
import torch.nn.functional as F
import hydra

class SAC(nn.Module):
 def __init__(self, cfg, gamma):
     super().__init__()
     self.cfg = cfg
     self.actor = hydra.utils.instantiate(self.cfg.actor)
     self.critic = hydra.utils.instantiate(self.cfg.critic)# pretrained before load ckpt in this file
     self.v_net = hydra.utils.instantiate(cfg.v_net) # pretrained before load ckpt in this file
     self.replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)
     self.gamma = gamma

 def compute_critic_and_Vnet_loss(self, tau, return_minq_and_v_value, device="cuda:0"): # implicit Q learninig, batch contains multiple transitions
     batch = self.replay_buffer.load_transition()
     actions = batch["action"].to(device)
     states = batch["obs"].to(device)
     reward = batch["rewards"].to(device)
     next_states = batch["next_obs"]
     done = batch["done"]
     q1,q2 = self.critic(actions,states)
     minq = torch.min(q1,q2)
     v_state = self.v_net(states)
     with torch.no_grad():
      minq.detach() 
      u = minq-v_value
     weight = torch.where(u>0,tau,1-tau)
     vnet_loss = (weight*u.pow(2)).mean()
     with torch.no_grad():
      v_value = self.v_net(next_states)
      y_q1 = reward +self.gamma*(1-done)*v_value
     critic_loss = F.mse_loss(minq,y_q1)
     if return_minq_and_v_value == True:
      return minq,v_value, critic_loss, vnet_loss #return minq, v_value every 100 epochs
     else: 
       return critic_loss, vnet_loss

 def compute_advantage(self,minq,v,gamma):
  
  w = torch.exp((minq-v)/gamma)
  return w

