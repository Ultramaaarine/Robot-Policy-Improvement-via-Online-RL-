# 1 training diffusion with demo
# 2 agent interact with env
# 3 new states obs for critic training
# 4 get advantage
# 5 training diffusion with new advantage, new obs  

import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.nn as nn
import random
import tqdm
import wandb
import logging
from pathlib import Path  
#import JsonLogger
from torch.utils.data import DataLoader
from sac_diffusion.workspaces.base_workspace import BaseWorkspace
from sac_diffusion.policy.lowdim_policy import LowdimPolicy
from sac_diffusion.datasets.calvin_skill import CalvinSkillDataset
from sac_diffusion.models.exponential_moving_average import EMAModel
import copy
from sac_diffusion.models.lr_scheduler import get_scheduler
from sac_diffusion.models.SAC_Model.Soft_Actor_Critc import SAC

logger = logging.getLogger(__name__)
logger2 = logging.getLogger("Online training logger")

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer

class TrainSAC_DiffusionLowdimWorkSpace():
   def __init__(self,
                replaybuffer_batch_size,
                online_traning,
                cfg:OmegaConf,output_dir = Path(__file__).parent.parent.parent.joinpath("fullmodel_outputs")
                ):
    self.cfg = cfg
    self.warm_up_steps = self.cfg.warm_up_steps
    seed =cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)   
    self.replaybuffer_batch_size = replaybuffer_batch_size
    self.sac:SAC
    self.sac = hydra.utils.instantiate(self.cfg.sac)
    self.optimizer = hydra.utils.instantiate(self.cfg.optimizer)
    self.env = hydra.utils.instantiate(self.cfg.env)
    self.global_step = 0
    self.epoch = 0
    self.output_dir = output_dir 
    self.normalizer = hydra.utils.instantiate(cfg.normalizer)
    self.online_training = online_traning
  #configure model
    self.diffusion_policy:LowdimPolicy
    self.diffusion_policy = hydra.utils.instantiate(self.cfg.policy)
    self.ema_policy = None
    if cfg.training.use_ema:
     self.ema_policy = copy.deepcopy(self.diffusion_policy)

  #configure optimizer
    self.optimizer: torch.optim.AdamW
    self.optimizer = hydra.utils.instantiate(cfg.optimizer,params = self.diffusion_policy.parameters())
    self.replaybuffer = hydra.utils.instantiate(cfg.replaybuffer) 

   def run(self):
  # 1 offline_training
     cfg = copy.deepcopy(self.cfg)
   # configure dataset
     training_dataset: CalvinSkillDataset
     validation_dataset: CalvinSkillDataset
     training_dataset = hydra.utils.instantiate(cfg.datamodule.training_dataset)#demos extracted from datamodule, stored in demos folder
     validation_dataset = hydra.utils.instantiate(cfg.datamodule.val_dataset)
     assert isinstance(training_dataset, CalvinSkillDataset)
     assert isinstance(validation_dataset, CalvinSkillDataset)
     critic_net_training_dataset:CalvinSkillDataset
     critic_net_validation_dataset:CalvinSkillDataset
     critic_net_training_dataset = hydra.utils.instantiate(cfg.datamodule.critic_net_training_dataset)
     critic_net_validation_dataset = hydra.utils.instantiate(cfg.datamodule.critic_net_validation_dataset)
   # configure replaybuffer

   # configure dataloader
     training_dataloader = DataLoader(training_dataset,num_workers=8,batch_size=32,shuffle=True) #cfg.datamodule.dataloader 也可作为参数配置
     validation_dataloader = DataLoader(validation_dataset,num_workers=8,batch_size=32,shuffle=False)
     critic_net_training_dataloader = DataLoader(critic_net_training_dataset,num_workers=8,batch_size=32,shuffle=True)
     critic_net_validation_dataloader = DataLoader(critic_net_validation_dataset,num_workers=8,batch_size=32,shuffle=False)
   # configure lr_scheduler
     lr_scheduler = get_scheduler(cfg.training.lr_scheduler,
                                 optimizer=self.optimizer,
                                 num_warmup_steps=cfg.training.lr_warmup_steps,
                                 num_training_steps=(len(training_dataloader)*cfg.training.num_epochs)\
                                 //cfg.training.gradient_accumulate_every,
                                 last_epoch = self.global_step-1
                                 )
  
                                  
   # configure ema
     ema = None
     ema:EMAModel
     if cfg.training.use_ema:
      ema = hydra.utils.instantiate(cfg.ema,
                                 model = self.diffusion_policy)

   # device_transfer:
     device = torch.device(cfg.training.device)
     diffusion_policy = self.diffusion_policy.to(device)
     if self.ema_policy is not None:
       self.ema_policy.to(device)
     optimizer_to(self.optimizer,device)
     self.sac.to(device)
   # configure normalizer      

   # training loop
   # 1.1 training diffusion policy(offline)
     for epoch_idx in range(cfg.training.num_epochs):
       train_losses = list()
       with tqdm.tqdm(training_dataloader,desc=f"Training epoch:{epoch_idx}",
                    leave=False,mininterval=self.cfg.training.mininterval) as tepoch:
          for batch_idx,batch in enumerate(tepoch):
            
            action,state = batch # check if batch shape is [B,T,D] do not forget to fit data in dataset for normalization
            action = action.to(device)
            state = state.to(device)
            
            raw_loss = self.diffusion_policy.compute_loss(action,state) # normalize here, in the policy
           # optimization gradient_accumulate 
            loss = raw_loss/cfg.training.gradient_accumulate_every
            loss.backward()
            if self.global_steps % cfg.training.gradient_accumulate_every == 0:
             self.optimizer.step()
             self.optimizer.zero_grad()
             lr_scheduler.step()
           # update ema
            if cfg.use_ema:
              ema.step(self.diffusion_policy)
           # logging
            raw_loss_cpu = raw_loss.item()
            train_losses.append(raw_loss_cpu)
            step_log = {
             "train_loss":raw_loss_cpu,
             "global_step":self.global_step,
             "epoch":self.epoch,
             "lr": lr_scheduler.get_scheduler()[0]
            }
            is_last_batch = (batch_idx ==len(training_dataloader)-1)
            if not is_last_batch:
              logger.info(logger.info(
                "step=%d epoch=%d loss=%.6f lr=%.6f",
                step_log["global_step"],
                step_log["epoch"],
                step_log["lr"],
                step_log["train_loss"]
              ))
            self.global_step += 1
            if (cfg.training.max_train_steps is not None) \
               and batch_idx >= (cfg.training.max_train_steps-1):
             break
          train_loss  = np.mean(train_losses) #batch loss  
          step_log['train_loss'] = train_loss 
          self.global_step =+1 # 由于 python idx的特性 最后一个batch的 idx 比总 batch数少1 所以在循环结束后加上1对齐
          self.epoch =+1  

   # 1.2 validation
       self.diffusion_policy.eval()
       if (self.epoch%cfg.training.val_every) == 0:
         with torch.no_grad():
           val_losses = list()

           with tqdm.tqdm(validation_dataloader,desc=f"Validation epoch:{self.epoch}",
                      mininterval=self.cfg.training.mininterval) as tepoch:
             for batch_idx,batch in enumerate(tepoch) :
                action,state = batch
                action.to(device)
                batch.to(device)
                val_loss = diffusion_policy.compute_loss(action,state)
                val_losses.append(val_loss) 
                if (cfg.training.max_val_step is not None) \
                  and batch_idx >= (cfg.training.max_val_step-1):
                  break
           if len(val_losses>0):
                
              val_loss = torch.mean(torch.tensor(val_losses))
              step_log["val_loss"] = val_loss
         self.diffusion_policy.train()
    # 1.3 run diffusion sampling (sampling based on demo) offline generation for evaluation (still part of training)
        
     offline_action = diffusion_policy.predict_action(state,action_mode="pos_ori") #action sequence [B,T,D], generated from demo and condition from demo
    # 1.4 offline training critic v_net
    
    # 1.5 warming up (online warm up), instead of going to diffusion model, obs from env will go to critic for Q,V learning 
     
     offline_action.cpu().numpy()
     self.env.reset()
     step_count = 1
     while not self.warm_up_steps == step_count:
      while not step_count % offline_action.shape[1]==0:
         obs, reward, next_obs, done = self.env.step(action[0,step_count-1,:]) 
         self.replaybuffer.save_transition(obs, reward, next_obs, done) 
         step_count +=1     
     
    # 1.6 train critic, Vnet with offline data (warm up)
     
     critic_loss, vnet_loss = self.sac.compute_critic_and_Vnet_loss(tau=1,return_minq_and_v_value=False)

            
     step_log["critic_loss"] = critic_loss
     step_log["Vnet_loss"] = vnet_loss
     logger2.info(logger.info({
           step_log["critic_loss"],
           step_log["Vnet_loss"]
         }))
    #===online training===#  
     if self.online_training == True:
     
       online_initial_obs = self.env.reset()
       assert isinstance(online_initial_obs,np.ndarray)
    # 2.1 compute advantage
    # 2.2 improve policy 
       assert isinstance(advantage,torch.Tensor)
       with 
       self.diffusion_policy.compute_loss(action,state,advantage) # policy should be modified
     
    # 2.3 sampling new action from improved policy

@hydra.main(version_base="1.1",
            config_name=Path(__file__).stem, 
            config_path=str(Path(__file__).parent.parent.parent.joinpath("config")))
def main(cfg):
   workspace = TrainSAC_DiffusionLowdimWorkSpace(cfg)
   workspace.run()

if __name__ == "__main__":
  main()