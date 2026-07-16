#这里面都是 gpu数据
#w = exp(A+H/gamma) A is advantage, H is entropy
#functions: 1 get Q value
#           2 estimate V 
#           3   
import torch 
import torch.nn as nn
from sac_diffusion.models.SAC_Model.critic import DoubleQNetwork
class Actor(object):
 def __init__(self,Vnet_input_dim, Vnet_middle_dim, Vnet_output_dim,cfg): #cfg 包括 critic的参数
    super().__init__()
    
    self.critic = DoubleQNetwork()
 def compute_advantage(self, q1,q2,gamma, v_value):# q1 q2 from critic
   
   assert isinstance(v_value,float)
   
   q = min(q1,q2)

   advantages = torch.exp(q-v_value/gamma)
   return advantages
 def send_advantage_to_diffusion_model(self,gamma,entropy):
   advantage = self.compute_advantage(gamma,entropy)
   self._send(advantage)
 def _send(self,advantage):
  assert isinstance(advantage, torch.Tensor)


