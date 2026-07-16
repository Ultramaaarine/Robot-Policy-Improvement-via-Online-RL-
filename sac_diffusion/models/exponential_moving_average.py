# θema <- (1-decay)θnew + decay*θema 对参数进行加权平均

import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from sac_diffusion.policy.lowdim_policy import LowdimPolicy

class EMAModel():
 def __init__(self,model:LowdimPolicy,power:float,update_after_step:int,min_value,max_value,gamma):#model is copied from Unet
  
  self.power = power
  self.averaged_model = model
  self.averaged_model.eval()
  self.averaged_model.requires_grad_(False)
  self.min_value = min_value
  self.max_value = max_value
  self.gamma = gamma
  self.updat_after_step = update_after_step
  self.optimization_step = 0

 def get_decay(self,current_step):
  
  step = max(0,current_step-self.updat_after_step) 
  if step<=0:
   return  0.0
  
  value = 1-(1+step/self.gamma)**-(self.power)
  decay_value = max(0,min(self.max_value,value))
  return decay_value
 
 torch.no_grad()
 def step(self,new_model):
  self.decay = self.get_decay(self.optimization_step)

  all_dataptrs = set()
  for module, ema_module in zip(new_model.modules(),self.averaged_model.modules()):
    for param, ema_param in zip(module.parameters(recurse = False),ema_module.parameters(recurse = False)):
     if isinstance(param,dict):
      raise RuntimeError("dict parameters are not supported")

     if isinstance(module,_BatchNorm):
      ema_param.copy_(param.to(dtype=ema_param.dtype).data)
     elif not param.requires_grad:
      ema_param.copy_(param.to(dtype=ema_param.dtype).data) 
     else:
      ema_param.mul_(self.decay) #带下划线的结尾（mul_, add_）：在 PyTorch 里表示就地（in-place）操作，相当于 a *= 2 之类的操作
                                 #会直接修改 ema_param 本身的内存，而不是返回一个新张量。
      ema_param.add_(param.data.to(dtype=ema_param.dtype),alpha = 1-self.decay) #add_(…, alpha=…) 是 PyTorch 提供的乘加融合接口
  self.optimization_step +=1

 
 