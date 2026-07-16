# this entire generation process can be considered as a receding horizon prediction with prediction horizon length 1 step
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import hydra
import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from sac_diffusion.utils.env_maker import make_env
from sac_diffusion.policy.policy_with_siamese import LowdimPolicy
from sac_diffusion.models.Vision.autoencoder import SiameseEncoder
from sac_diffusion.models.replay_buffer import ReplayBuffer
from sac_diffusion.models.SAC_Model.critic import DoubleQNetwork
from sac_diffusion.models.SAC_Model.V_net import ValueNet
from sac_diffusion.datasets.online_dataset import OnlineDataset
from sac_diffusion.utils.target_selector import GMMTargetSelector
from sac_diffusion.utils.target_selector import TargetSelector
from sac_diffusion.datasets.calvin_critic_offline_dataset import CalvinCriticOfflineDataset
from sac_diffusion.models.normalizer import Normalizer

logger = logging.getLogger(__name__)

class Agent():
    def __init__(self,cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = make_env(cfg.env)
        self.siamese_encoder:SiameseEncoder
        self.siamese_encoder = hydra.utils.instantiate(cfg.siamese_encoder)
        self.diffusion_policy:LowdimPolicy
        self.diffusion_policy = hydra.utils.instantiate(cfg.diffusion_policy)
        self.critic:DoubleQNetwork
        self.critic = hydra.utils.instantiate(self.cfg.critic)
        self.replay_buffer:ReplayBuffer
        self.replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)
        self.v_net:ValueNet
        self.v_net = hydra.utils.instantiate(cfg.v_net)
        self.normalization_dataset: CalvinCriticOfflineDataset
        self.normalization_dataset = hydra.utils.instantiate(cfg.normalization_dataset)
        self.normalizer = Normalizer()
        self.critic_optimizer: torch.optim.AdamW
        self.critic_optimizer = hydra.utils.instantiate(cfg.optimizer.critic_optimizer,params=self.critic.parameters())
        self.v_net_optimizer: torch.optim.AdamW
        self.v_net_optimizer = hydra.utils.instantiate(cfg.optimizer.v_net_optimizer,params=self.v_net.parameters())
        self.diffusion_optimizer:torch.optim.AdamW
        self.diffusion_optimizer = hydra.utils.instantiate(cfg.optimizer.diffusion_optimizer,params=self.diffusion_policy.parameters())
        self.target_selector = TargetSelector(cfg.skill,sort_by="label")
        self.target_selector.scan()
    def online_training(self):
        run_id = datetime.datetime.now().strftime("%Y%M%d_%H%M%s")
        log_dir = Path(__file__).parent.parent.parent.joinpath("online_agent_training_log",run_id)
        self.writer = SummaryWriter(log_dir)
        # 1 warm up 5000 steps
        self.action_normalization_param = self.
        siamese_ckpt_path = self.cfg.siamese_ckpt_path
        diffusion_ckpt_path = self.cfg.diffusion_ckpt_path
        siamese_ckpt = torch.load(siamese_ckpt_path,map_location=self.device)
        diffusion_ckpt = torch.load(diffusion_ckpt_path,map_location=self.device)
        self.siamese_encoder.load_state_dict(siamese_ckpt["model"],strict=True)
        self.siamese_encoder.to(self.device)
        t_goal = self.target_selector.predict_t(T=64)
        self.action_normalization_dict = self.normalization_dataset.get_normalize_params("action")
        p = self.siamese_encoder.encoder.parameters() # freeze encoder parameters
        self.siamese_encoder.eval()
        p.requires_grad = False
        for head in self.siamese_encoder.heads:
            for param in head.parameters():
                param.requires_grad = True
        
        
        obs = self.env.reset()
        for step in self.cfg.total_step: # 
         robot_state = obs["robot_obs"][:3]
         goal_visual_obs = obs["gripper"][]
         visual_obs = obs["gripper"]
         dist = self.siamese_encoder(obs["state"][:3], visual_obs)
         target = dist + obs["state"]
         error = self.siamese_encoder(obs["state"][:3], visual_obs)
         dx = self.diffusion_policy.predict_action(target-robot_state) # velocity 
         print(f"pred_action has shape: {dx.shape}")
         single_step = dx[0,0,:] # receding horizon
         obs,reward,next_obs,done = self.env.step(single_step) 
         obs = next_obs
         seq = []
         transition = {
                "obs": obs,
                "action": dx,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "error":  error
         }
         self.replay_buffer.save_transitions(transition) # transition for critic update
         seq.append(transition)
         self.replay_buffer.save_sequence(seq)
         # 2 online critic network update (IQL)
          # 2.1 update V net
         if step % self.cfg.update_critic_interval ==0 and step > self.cfg.warm_up_steps: # update critic every 50 steps
             losses = []
             transition_batch = self.replay_buffer.load_transitions(self.cfg.replay_buffer.batch_size)
             actions = transition_batch["action"]
             state = transition_batch["obs"]
             next_state = transition_batch["next_obs"]
             reward = transition_batch["reward"] # [B,1] define reward in env 
             done = transition_batch["done"] # [B,1]
             with torch.no_grad():
              q1_new,q2_new = self.critic(state,actions)
              q_new = torch.min(q1_new,q2_new)
             v_value_new = self.v_net(state)
             u_new = q_new - v_value_new
             tau = self.cfg.quantile_tau
             weight = torch.where(u_new>0,tau,1-tau)
             v_value_new_loss = weight*(u_new.pow(2)).mean()
             self.v_net_optimizer.zero_grad()
             v_value_new_loss.backward()
             self.v_net_optimizer.step()
          # 2.2 update Critic network
             with torch.no_grad():
               new_target_q = reward+ self.cfg.training.discount_factor*(1-done)*self.v_net(next_state)
             current_q1,current_q2 = self.critic(transition_batch["obs"],transition_batch["action"])
             self.critic_optimizer.zero_grad()
             loss = F.mse_loss(new_target_q,current_q1) + F.mse_loss(new_target_q,current_q2)
             loss.backward()
             losses.append(loss.item())
             aver_loss = float(np.mean(losses))
             self.critic_optimizer.step()
         # 3 online diffusion model update
         if step % self.cfg.training.upgrade_diffusion_model == 0 and step > self.cfg.warm_up_steps:
            
            diffusion_batch = self.replay_buffer.load_sequence(self.cfg.replay_buffer.batch_size) # load sequence for diffusion model training  maybe load big batch then split into small batches      
           
            assert(diffusion_batch.shape[0] !=0)

            if diffusion_batch.shape[0] % self.cfg.batch_size != 0:
               valid_batch_num = (diffusion_batch.shape[0] // self.cfg.batch_size)*self.cfg.batch_size
               diffusion_batch = diffusion_batch[:valid_batch_num]
               split_num = int(valid_batch_num * 0.8) #向下取整
               split_num = (split_num // self.cfg.batch_size)*self.cfg.batch_size # ensure split num is multiple of batch size
               diffusion_training_seq = diffusion_batch[:split_num]
               diffusion_val_seq = diffusion_batch[split_num:]

            elif  diffusion_batch.shape[0] % self.cfg.batch_size == 0: 
               valid_batch_num = diffusion_batch.shape[0] 
               diffusion_batch = diffusion_batch[:valid_batch_num]
               split_num = int(valid_batch_num * 0.8) #向下取整              
               split_num = (split_num // self.cfg.batch_size)*self.cfg.batch_size # ensure split num is multiple of batch size
               diffusion_training_seq = diffusion_batch[:split_num]
               diffusion_val_seq = diffusion_batch[split_num:]  
            training_dataset =  OnlineDataset(diffusion_training_seq,batch_size=self.cfg.batch_size) 
            val_dataset = OnlineDataset(diffusion_val_seq,batch_size=self.cfg.batch_size)
            training_dataloader = DataLoader(training_dataset,batch_size=None,shuffle=True)
            val_datatloader = DataLoader(val_dataset,batch_size=None,shuffle=False)
            for epoch_idx in range(self.cfg.training.diffusion.diffusion_epoch):
              with tqdm.tqdm(training_dataloader,desc="Updating Diffusion Model",mininterval=1.0) as tepoch:
                for batch_idx, diffusion_batch in enumerate(tepoch):
                    policy_weight = None
                    loss = self.diffusion_policy.compute_loss(action=diffusion_batch["action"],cond=diffusion_batch["obs"],weight=policy_weight)
                    self.diffusion_optimizer.zero_grad()
                    loss.backward()
                    self.diffusion_optimizer.step()
                    if epoch_idx % self.cfg.training.diffusion.val_every_epochs == 0:
                       self.diffusion_policy.eval()
                       val_losses = []
                       with torch.no_grad():
                          with tqdm.tqdm(val_datatloader,desc="Validating Diffusion Model",mininterval=1.0) as vepoch:
                           for batch_idx,batch in enumerate(vepoch):
                              action = batch["action"][:3],cond = batch["state"][:3]
                              val_loss = self.diffusion_policy.compute_loss(action,cond)
                              val_losses.append(val_loss.item()) 

                            
         # 4 rollout
            if step % self.cfg.training.rollout_every == 0 and step > self.cfg.warm_up_steps:
               # rollout with diffusion policy roll out seq 
               rollout_obs = self.env.reset()
               rollout_seq = self.replay_buffer.load_sequence(batch_size=32)[0,:,:]


               for rollout_cur_step in range(self.cfg.training.rollout_length):
                   roll_out_goal_step = self.target_detector.sample_goal_time()

                   robot_state = rollout_obs["robot_obs"][:3]
                   target = self.siamese_encoder(rollout_obs["state"][:3])+ rollout_obs["state"][:3] # target shape: [B,T,D] error shape: [B,T,D]
                  
                   dx = self.diffusion_policy.predict_action(target-robot_state[:3]) # velocity for single step 
                   step = dx[0,0,:]
                   next_rollout_obs,reward,done,_ = self.env.step(step)
                   transition = {
                     "obs":rollout_obs,
                     "action":dx,
                     "reward":reward,
                     "next_obs":next_rollout_obs,
                     "done":done
                   }
                   rollout_seq.append(transition)
                   rollout_obs = next_rollout_obs
               self.replay_buffer.save_sequence(rollout_seq) # save rollout sequence to replay buffer for future training
         # 5 update target detector parameters
            if step % self.training.update_gmm_selector_every == 0 and step > self.cfg.warm_up_steps:
               # initialize gmm target selector
               summaries = self.target_selector.report_dict()
               mu_list = []
               std_list = []
               for idx, summary in summaries.items():
                mu_list.append(summary["mean"])
                std_list.append(summary["std"])
               
               
               # ∂Loss/∂action * ∂action/∂delta_pos * ∂delta_pos/∂P(t) * ∂P(t)/∂t_params(mu,sigma) 
               self.gmm_selector = GMMTargetSelector(num_components=self.cfg.gmm_selector.goal_nums,init_mu =mu_list,init_sigma= std_list)
               self.siamese_encoder.requires_grad_(False) # freeze siamese encoder parameters during gmm selector training
               self.gmm_selector.requires_grad_(True)
               t_goal = self.gmm_selector.
               delta_pos = self.siamese_encoder(t_goal)
               action = self.diffusion_policy.predict_action(delta_pos,action_mode="pos",params = self.action_normalization_dict)

         # 6 update siamese encoder parameters
            if step % self.training.siamese_upgrade_every == 0 and step > self.cfg.warm_up_steps:
                                 
               visual_state = self.replay_buffer.load_sequence(seq_type="visual_state",batch_size=self.cfg.training.siamese_batch_size) # [N,B,T,D]
               state_seq_for_siamese = visual_state["state"]
               visual_obs_seq = visual_state["visual"]
               
               assert state_seq_for_siamese.shape[0] !=0, "No state sequences available for Siamese encoder training"
               assert visual_obs_seq.shape[0] !=0, "No visual observation sequences available for Siamese encoder training"
               assert state_seq_for_siamese.shape[0] == visual_obs_seq.shape[0], "State sequence batch size and visual observation sequence batch size must be the same"
               training_split_num = int(state_seq_for_siamese.shape[0]*0.8) # split data
               training_split_num = (state_seq_for_siamese.shape[0]//self.cfg.training.siamese_batch_size)*self.cfg.training.siamese_batch_size
               training_dataset = OnlineDataset(state_seq_for_siamese[:training_split_num],batch_size=self.cfg.training.siamese_batch_size)
               val_dataset = OnlineDataset(state_seq_for_siamese[training_split_num:],batch_size=self.cfg.training.siamese_batch_size)

               state_dataloader = DataLoader(OnlineDataset(training_dataset,batch_size=self.cfg.training.siamese_batch_size),batch_size=None,shuffle=True) 
               for epoch_num in self.training.update_siamese_epoch:
                 # pick key steps
                 with tqdm.tqdm(range(epoch_num),desc="Updating Siamese Encoder",mininterval=1.0) as sepoch:
                      pass
                
    @hydra.main(version_base="1.1",config_name="online_agent_training",config_path=Path(__file__).parent.parent.parent.joinpath("config/online_agent_training"))
    def main(cfg):
        agent = Agent(cfg)
        agent.online_training()

    if __name__ == "__main__":
        main()
