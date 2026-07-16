#定义 sampling时要注意：
#一、我们的数据分成 ori_pos 和 pos 两种，当选择pos_ori时，dim为6，选择 pos 时为3
#二、sampling是逆向过程，从高斯噪声生成policy
#三、数据在训练脚本里归一化
#四、需要一个yaml文件实例化这个policy
#五、

from typing import Union,Tuple,Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader
from sac_diffusion.policy.base_policy import BasePolicy
from sac_diffusion.models.Diffusion_Unet import Lowdim_Unet
from sac_diffusion.models.normalizer import Normalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from sac_diffusion.datasets.calvin_skill import CalvinSkillDataset  #这是用来训练的 dataset

class LowdimPolicy(BasePolicy):
 def __init__(self, 
              model:Lowdim_Unet,
              num_sampling_steps:int,
              training_timesteps:int,
              beta_start,
              beta_end,
              beta_schedule:str,
              prediction_type:str,
              mode:str
              ):
  super().__init__()
  self.training_timesteps = training_timesteps
  self.num_sampling_steps = num_sampling_steps
  self.model = model
  self.mode = mode
 
  self.normalizer = Normalizer()
  self.scheduler = DDPMScheduler(self.training_timesteps,
                                 beta_start=beta_start,
                                 beta_end=beta_end,
                                 beta_schedule=beta_schedule,
                                 prediction_type=prediction_type
                                 )

 #=======Offline Training=======# 
 def set_normalizer(self,normalizer):
    self.normalizer = normalizer
    return self.normalizer

 def normalize_data(self,data:Union[np.ndarray,torch.Tensor],params)->Union[np.ndarray,torch.Tensor]: #params from dataset
   
   
   n_data = self.normalizer.normalize(data,params)
   return n_data
    
  
 def compute_loss(self,action:Union[np.ndarray,torch.Tensor],
                  cond:Union[Dict,torch.Tensor],action_norm_params,cond_norm_params,weight:Optional[torch.Tensor]): # compute weighted loss 
    assert len(action[0]) != 0
    assert len(cond[0]) != 0 
    if isinstance(cond, dict):
      for key,value in cond.items():
        print(f"obs has following keys: {key}")
   #print(f"condition has shape: {cond.shape}")
    model = self.model
    print(f"action shape: {action.shape}")

    n_action = self.normalize_data(action,action_norm_params).to("cuda:0")
    n_cond = self.normalize_data(cond,cond_norm_params).to("cuda:0")

    noise = torch.randn(n_action.shape,device=action.device)

   
    training_timesteps = torch.randint(0,self.training_timesteps,(n_action.shape[0],),device=action.device)
    
   #print(f"training timestep,condition are on {training_timesteps.device},{cond.device}")
   # print(f"n_action,n_cond,noise, training_timesteps are on {n_action.device},{n_cond.device},{noise.device},{ training_timesteps.device}")
    noisy_action = self.scheduler.add_noise(n_action,noise,training_timesteps)

   #print(f"noisy action has shape {noisy_action.shape}")#[32,64,3]
    pred = model(noisy_action, # shape: [32,64,D],shape 
                 training_timesteps,
                 visual_obs = None,
                 low_dim_obs = n_cond, 
                 
                 online_training_step = None)
    if weight == None:
     if self.scheduler.config.prediction_type == "epsilon": # 推理采样：必须用 scheduler.step() 做反向扩散更新,step() 需要知道模型输出代表什么（ε / x0 / v），所以它会强制检查 prediction_type 是否是 "epsilon" / "sample" / 所以必须写 epislon 或者 action
    #   loss = F.mse_loss(pred,noise) # F.mse_loss 的默认 reduction 是 mean 返回一个标量 
      loss = F.mse_loss(pred,noise)
      
     elif self.scheduler.config.prediction_type == "sample":
    #   loss = F.mse_loss(pred,n_action)
       loss = F.mse_loss(pred,n_action)
    elif weight != None:  # weight: [B]
      assert weight.shape[0] == action.shape[0]# sequence wise loss and weight
      raw_loss = F.mse_loss(pred,noise,reduction="none") # [B,T,D]
      per_sequence_loss = raw_loss.mean(dim=[1,2]) # [B,]
      
    return loss
 
 #===Offline Genaration===#
 @torch.no_grad()
 def sampling_action_from_states(self,states,action_mode,online_training_step): 
  if action_mode == "pos":
    Da = 3
  elif action_mode == "pos_ori" : 
    Da = 6
  elif action_mode == "pos_ori_gripper":
    Da = 7
  generator = None
  
  states = torch.from_numpy(states).to("cuda:0",dtype=torch.float32)
  
  if states.dim() == 3:
      
     model =self.model
     action = torch.randn(size=[states.shape[0],states.shape[1],Da],
                        dtype=states.dtype,
                        device=states.device,
                        generator=generator).to("cuda:0",dtype=torch.float32) #generator要传一个实例 别忘了
     
     self.scheduler.set_timesteps(self.num_sampling_steps)
  
 
     
     for t in self.scheduler.timesteps:
       model_output = model(action,t,states) # state action 在online是 [D,]维度需经过Unet里面的obs wrapper 处理, 在 offline 是 [B,63,D]会在 Unet里面 padding成 [32,64,D]
       action = self.scheduler.step(model_output,t,action,
                         generator=generator,
                         ).prev_sample #出现  RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float double 指的是 float 64 和 float 32

   
     return action #normalized denoised signal 
   ## Online generation, cond has shape [D,]
  elif states.dim() == 1:
     model = self.model
     action = torch.randn(size=[1,64,Da],
                         dtype=states.dtype,
                         device= states.device,
                         generator=generator).to("cuda:0")
     
     #print(f"action device: {action.device}, obs device: {states.device}")
     #print(f"action type: {action.dtype}, obs type: {states.dtype}")
     self.scheduler.set_timesteps(self.num_sampling_steps)
     for t in self.scheduler.timesteps:
       model_output = model(action,pos = t,
                            visual_obs = None,
                            low_dim_obs = states, # 这里传参数一定注意 最好把每个参数都明确
                            online_training_step = online_training_step)
       action = self.scheduler.step(model_output,t,action,
                                   generator=generator).prev_sample
       
     return action
     
 def predict_action(self,states,action_mode,params,online_training_step)->torch.Tensor:# unnormalize prediction
      
  
  n_predicted_action = self.sampling_action_from_states(states,action_mode,online_training_step)
  predicted_action = self.normalizer.unnormalize(n_predicted_action,params)
  # 在env里面 action会变成 仿真用的 ndarray

  return predicted_action
  