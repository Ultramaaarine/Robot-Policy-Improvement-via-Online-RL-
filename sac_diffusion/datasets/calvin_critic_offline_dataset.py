# dataset for online data normalization
# and for offline critic training
# flatten = True
# normalize = True
# contains error values, which are derived from goal pos
import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pdb
import pybullet as p
from pathlib import Path
from omegaconf import DictConfig
from sac_diffusion.models.normalizer import Normalizer
from typing import Optional
#from sac_diffusion.utils.target_selector import TargetSelector

class CalvinCriticOfflineDataset(Dataset):
    def __init__(
        self,
        skill: DictConfig,
        goal_centered: bool,
        flatten: bool,
        demos_dir: str,
        data_type:str,
        is_quaternion:bool,
        use_visual_obs:bool = False,
        action_params:Optional[dict]=None,
        state_params:Optional[dict]=None,
        joint_pos_params:Optional[dict] = None,
        rgb_obs_params:Optional[dict]=None,
        depth_obs_params:Optional[dict]=None,
        mode:Optional[str]=None,
        
        
        
    ):
        self.skill = skill
        self.goal_centered = goal_centered
        self.demos_dir = Path(demos_dir).expanduser()
        self.state_type = self.skill.state_type
        self.dt = self.skill.dt
        self.normalized = self.skill.normalized
        print(f"self.normalized in CalvinCriticOfflineDataset: {self.normalized}") # False
        self.norm_range = [-1, 1]
        self.is_quaternion = is_quaternion
        self.X_mins = None
        self.X_maxs = None
        self.fixed_ori = None
        self.start = None
        self.goal = None
        
        self.normalizer = Normalizer()
        # self.target_selector = TargetSelector()
        # self.target_selector.scan()
        # self.target_dict = self.target_selector.predict_t(T=64)
        self.use_visual_obs = use_visual_obs
        self.action_params = action_params
        self.state_params = state_params
        self.joint_pos_params = joint_pos_params
        self.rgb_obs_params = rgb_obs_params
        self.depth_obs_params = depth_obs_params    
        assert self.demos_dir.is_dir(), "Demos directory does not exist!"
        if data_type =="training":
         self.robot_low_dim_obs_data_file = glob.glob(str(self.demos_dir / self.skill.name/"low_dim_obs" / "training.npy"))[0]
         self.robot_scene_obs_data_file = glob.glob(str(self.demos_dir / self.skill.name / "scene_obs" / "training.npy"))[0]
         self.robot_rgb_obs_data_path = Path(self.demos_dir / self.skill.name / "rgb_obs")
        # self.robot_rgb_static_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_static" / "training.npy"))[0]
         self.robot_rgb_gripper_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_gripper" / "training.npy"))[0]
         self.robot_rgb_tactile_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_tactile" / "training.npy"))[0]
        #  self.robot_depth_obs_data_path = Path(self.demos_dir / self.skill.name / "depth_obs")
        #  self.robot_depth_static_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_static" / "training.npy"))[0]
        #  self.robot_depth_gripper_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_gripper" / "training.npy"))[0]
        #  self.robot_depth_tactile_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_tactile" / "training.npy"))[0]
         
        elif data_type == "validation":
         self.robot_low_dim_obs_data_file = glob.glob(str(self.demos_dir / self.skill.name / "low_dim_obs"/"validation.npy"))[0]
         self.robot_scene_obs_data_file = glob.glob(str(self.demos_dir / self.skill.name / "scene_obs" / "validation.npy"))[0]
         self.robot_rgb_obs_data_path = Path(self.demos_dir / self.skill.name / "rgb_obs")
        # self.robot_rgb_static_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_static" / "validation.npy"))[0]
         self.robot_rgb_gripper_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_gripper" / "validation.npy"))[0]
         self.robot_rgb_tactile_obs_data_file = glob.glob(str(self.robot_rgb_obs_data_path / "rgb_tactile" / "validation.npy"))[0]
        #  self.robot_depth_obs_data_path = Path(self.demos_dir / self.skill.name / "depth_obs")
        #  self.robot_depth_static_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_static" / "validation.npy"))[0]
        #  self.robot_depth_gripper_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_gripper" / "validation.npy"))[0]
        #  self.robot_depth_tactile_obs_data_file = glob.glob(str(self.robot_depth_obs_data_path / "depth_tactile" / "validation.npy"))[0]
    
        cols = self.get_valid_columns(self.state_type)
        joint_pos_col = self.get_valid_columns("joint")
        self.X = np.load(self.robot_low_dim_obs_data_file)[:, :, cols]
        self.Q = np.load(self.robot_low_dim_obs_data_file)[:, :, joint_pos_col]
        print(f"joint pos data has shape: {self.Q.shape}")
        self.robot_scene_obs = np.load(self.robot_scene_obs_data_file)
        assert isinstance(self.robot_scene_obs, np.ndarray)
        self.robot_rgb_obs_data = {
           #"rgb_static":np.load(self.robot_rgb_static_obs_data_file), 
           "rgb_gripper":np.load(self.robot_rgb_gripper_obs_data_file), 
           #"rgb_tactile":np.load(self.robot_rgb_tactile_obs_data_file)
                                   }
        
        #self.robot_depth_obs_data = {
           #"depth_static_obs":np.load(str(self.robot_depth_static_obs_data_file)),
           #"depth_gripper_obs":np.load(str(self.robot_depth_gripper_obs_data_file)),
           #"depth_tactile_obs":np.load(str(self.robot_depth_tactile_obs_data_file))
        #}
        self.drawer_joint_state = self.robot_scene_obs[:,:,1:2] #[N,T,1]
        #print(f"self.drawer_joint_state: {self.drawer_joint_state}") # check the drawer joint state values
        max_reward = 3.0
        self.step_reward = -0.1
        max_drawer_joint_state = 0.24
        self.rewards = (self.drawer_joint_state-0.0)/(max_drawer_joint_state-0.0)*max_reward
        self.rewards = np.clip(self.rewards,0.0,max_reward).astype(np.float32) # state rewards
        self.transition_rewards = self.rewards[:,:-1,:] # align with self.obs shape [N,T-1,1]
        self.transition_rewards = self.transition_rewards + self.step_reward
        self.done = np.zeros_like(self.transition_rewards)
        self.done[:,-1,:] = 1
        self.done = torch.from_numpy(self.done)
        self.robot_rgb_gripper_obs_data_file
        # Get the last orientation from the trajectory (this is bad for orientation dependant tasks)
        cols_ori = self.get_valid_columns("ori")
        temp_ori = np.load(self.robot_low_dim_obs_data_file)[:, :, cols_ori]
        self.fixed_ori = temp_ori[0, -1, :]
   
        if self.state_type == "ori" and self.is_quaternion:
            self.X = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X)#p.getQuaternionFromEuler接受三元向量，返回四元数
        elif self.state_type == "pos_ori" and self.is_quaternion:
            oris = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X[:, :, 3:])
            self.X = np.concatenate([self.X[:, :, :3], oris], axis=-1)

        self.start = np.mean(self.X[:, 0, :3], axis=0)
        self.goal = np.mean(self.X[:, -1, :3], axis=0)
        if self.goal_centered:
            # Make X goal centered i.e., subtract each trajectory with its goal
            self.X[:, :, :3] = self.X[:, :, :3] - np.expand_dims(self.X[:, -1, :3], axis=1)



        self.dX = np.zeros_like(self.X) # same shape
        self.dX[:, :-1, :3] = (self.X[:, 1:, :3] - self.X[:, :-1, :3]) / self.dt
        
        self.dX[:, -1, :3] = 0
        self.obs = self.X[:,:-1,:] # T = 63
        self.next_obs = self.X[:,1:,:]  # T = 63   
        self.action = self.dX[:,:-1,:] # align with self.obs shape [N,T-1,D]
        self.action = self.dX[:, :-1, :]          # [N,63,D]
        self.next_action = self.dX[:, 1:, :]      # [N,63,D]

        self.joint_pos = self.Q[:, :-1, :]        # [N,63,Dj]
        print(f"joint_pos has shape: {self.joint_pos.shape}")
        self.next_joint_pos = self.Q[:, 1:, :]    # [N,63,Dj]]
        for k,v in self.robot_rgb_obs_data.items():
           if self.robot_rgb_obs_data[k] is not None: 
                self.robot_rgb_obs_data[k] = v[:,:-1,:,:,:]
        if self.state_type == "pos_ori":
            self.Ori = self.X[:, :, 3:]
            self.Ori = torch.from_numpy(self.Ori).type(torch.FloatTensor)
            self.X = self.X[:, :, :3]
        
        # self.error_norm_params = self.normalizer.fit(self.error,mode="gaussian")
        # self.n_error = self.normalizer.normalize(self.error, self.error_norm_params)
        if flatten:
            B, S, L = self.X.shape
            #print(f"self.X shape: {self.X.shape}") 
            self.obs = self.obs.reshape(B * (S-1), L)
            #print(f"self.obs has shape: {self.obs.shape}")
            self.next_obs = self.next_obs.reshape(B * (S-1), L)
            self.action = self.action.reshape(B * (S-1), L)
            self.next_action = self.next_action.reshape(B * (S-1), L)
            self.transition_rewards = self.transition_rewards.reshape(B*(S-1), 1)
            self.done = self.done.reshape(B*(S-1), 1)
            # self.error = self.error.reshape(B*(S-1),L)
        
        if self.action_params is None:
               self.action_params = self.normalizer.fit(self.dX,mode)
        if self.state_params is None:
               self.state_params = self.normalizer.fit(self.X,mode)
        if self.joint_pos_params is None:
            self.joint_pos_params = self.normalizer.fit(self.joint_pos,mode)
        if self.use_visual_obs:
          if self.rgb_obs_params is None:
               #self.rgb_static_obs_params = self.normalizer.fit(self.robot_rgb_obs_data["rgb_static"].astype(np.float32)/255.0,mode)
               self.rgb_gripper_obs_params = self.normalizer.fit(self.robot_rgb_obs_data["rgb_gripper"].astype(np.float32)/255.0,mode)
               #self.rgb_tactile_obs_params = self.normalizer.fit(self.robot_rgb_obs_data["rgb_tactile"].astype(np.float32)/255.0,mode )
               self.rgb_obs_params = {
                   #"rgb_static_obs_params": self.rgb_static_obs_params,
                   "rgb_gripper_obs_params": self.rgb_gripper_obs_params,
                   #"rgb_tactile_obs_params": self.rgb_tactile_obs_params    
                   }
        #   if self.depth_obs_params is None:
            #    self.depth_static_obs_params = self.normalizer.fit(self.robot_depth_obs_data["depth_static_obs"],mode) 
            #    self.depth_gripper_obs_params = self.normalizer.fit(self.robot_depth_obs_data["depth_gripper_obs"],mode) 
            #    self.depth_tactile_obs_params = self.normalizer.fit(self.robot_depth_obs_data["depth_tactile_obs"],mode) 
            #    self.depth_obs_params = { "depth_static_obs_params": self.depth_static_obs_params,
            #                             "depth_gripper_obs_params": self.depth_gripper_obs_params,
            #                             "depth_tactile_obs_params": self.depth_tactile_obs_params
            #                             } 
    def __len__(self):
        return self.obs.shape[0]
        

    def __getitem__(self, idx):
        obs_np = self.obs[idx] #[idx,T,D]
        
        obs_np = torch.from_numpy(obs_np) # tensor
        obs_norm = self.normalizer.normalize(obs_np,self.state_params)
        #print(f"obs_norm has shape {obs_norm.shape}")
        next_obs_np= self.next_obs[idx]
        next_obs_np = torch.from_numpy(next_obs_np) # tensor
        next_obs_norm = self.normalizer.normalize(next_obs_np,self.state_params)
        action_np = self.action[idx] 
        action_np = torch.from_numpy(action_np) # tensor
        x = action_np.clone().float()              # 原始 action（未归一化）
        xn = self.normalizer.normalize(x, self.action_params)
        xr = self.normalizer.unnormalize(xn, self.action_params)
        joint_pos = self.joint_pos[idx]
        next_joint_pos = self.next_joint_pos[idx]
        joint_pos_np = torch.from_numpy(joint_pos)
        joint_pos_norm = self.normalizer.normalize(joint_pos,self.joint_pos_params)
        next_joint_pos_np = torch.from_numpy(next_joint_pos)
        action_norm = self.normalizer.normalize(action_np,self.action_params) 
        next_action_np = self.next_action[idx]
        next_action_np = torch.from_numpy(next_action_np) # tensor
        next_action_norm = self.normalizer.normalize(next_action_np,self.action_params)
        
        reward_np = self.transition_rewards[idx] # get reward for each timestep,state reward,[63 steps]
        reward = torch.from_numpy(reward_np)
        # error = self.error[idx]

        if self.use_visual_obs:
           #rgb_static =self.robot_rgb_obs_data["rgb_static"][idx]
           #rgb_static = rgb_static.astype(np.float32)/255.0
           #rgb_static_norm = self.normalizer.normalize(rgb_static,self.rgb_static_obs_params)
           rgb_gripper = self.robot_rgb_obs_data["rgb_gripper"][idx]
           rgb_gripper = rgb_gripper.astype(np.float32)/255.0
           rgb_gripper_norm = self.normalizer.normalize(rgb_gripper,self.rgb_gripper_obs_params)
           #rgb_tactile = self.robot_rgb_obs_data["rgb_tactile"][idx]
           #rgb_tactile = rgb_tactile.astype(np.float32)/255.0
           #rgb_tactile_norm = self.normalizer.normalize(rgb_tactile,self.rgb_tactile_obs_params)
           #depth_static = self.robot_depth_obs_data["depth_static_obs"][idx]
           #depth_static_norm = self.normalizer.normalize(depth_static,self.depth_static_obs_params)
           #depth_gripper = self.robot_depth_obs_data["depth_gripper_obs"][idx]
           #depth_gripper_norm = self.normalizer.normalize(depth_gripper,self.depth_gripper_obs_params)
           #depth_tactile = self.robot_depth_obs_data["depth_tactile_obs"][idx]
           #depth_tactile_norm = self.normalizer.normalize(depth_tactile,self.depth_tactile_obs_params)
           
           if self.normalized:
              
            return{"obs":obs_norm,
                   "joint_pos":joint_pos_norm, 
                   "next_joint_pos":next_joint_pos_np,
                "action":action_norm,
                "reward":reward,
                "next_obs":next_obs_norm,
                "next_action":next_action_norm,
                "done":self.done[idx],
                #"error":self.n_error,
                "gripper_obs":{
                        
                        #"depth_gripper": depth_gripper,
                        "rgb_gripper": rgb_gripper,
                        
                #     },
                # "static_obs":{
                #         "depth_static": depth_static,
                #         "rgb_static": rgb_static,
                        
                        
                #      },
                # "tactile_obs":{
                #         "depth_tactile": depth_tactile,
                #         "rgb_tactile": rgb_tactile
                #      
                }
                 }# return a dict {} 
           elif not self.normalized:
            return {"obs":obs_np,
                    "joint_pos":joint_pos_np,
                    "next_joint_pos":next_joint_pos_np,
                    "action":action_np,
                    "reward":reward,
                    "next_obs":next_obs_np,
                    "next_action":next_action_np,
                    "done":self.done[idx],
                    # "error":error,
                    "gripper_obs":{   
                                           
                        #"depth_gripper": depth_gripper,
                        "rgb_gripper": rgb_gripper,  

                    },
                    # "static_obs":{
                       
                    #     "depth_static": depth_static,
                    #     "rgb_static": rgb_static,
                        
                    #  },
                    #  "tactile_obs":{

                    #     "depth_tactile": depth_tactile,
                    #     "rgb_tactile": rgb_tactile

                    #  }
                    }
        else:
         if self.normalized:
          return{
                "obs":obs_norm, 
                "action":action_norm,
                "reward":reward,
                "next_obs":next_obs_norm,
                "next_action":next_action_norm,
                "done":self.done[idx],
                 }# return a dict {}
         elif not self.normalized:
          return {
                    "obs":obs_np,
                  "joint_pos":joint_pos_np,
                  "action":action_np,
                  "reward":reward,
                    "next_obs":next_obs_np,
                    "next_action":next_action_np,
                    "done":self.done[idx],
                    }

    def get_valid_columns(self, state_type):
     if "joint" in state_type:
        return np.arange(7, 14)
     elif "pos_ori_grip" in state_type:
        return np.r_[0:7]
     elif "pos_ori" in state_type:
        return np.arange(0, 6)
     elif "pos" in state_type:
        return np.arange(0, 3)
     elif "ori" in state_type:
        return np.arange(3, 6)
     elif "grip" in state_type:
        return np.arange(6, 7)
     else:
        raise ValueError(f"Unknown state_type: {state_type}")

    def rm_rw_data(self, list_indicis):
        self.X = np.load(self.robot_low_dim_obs_data_file)
        new_X = np.delete(self.X, list_indicis, axis=0)
        np.save(self.robot_low_dim_obs_data_file, new_X)
    
    def get_normalize_params(self,param_type:str):
       
       if param_type == "action_params":
          return self.action_params # offline normalize params for normalization
       elif param_type == "state_params":
            return self.state_params
       elif param_type == "rgb_obs_params":
            return self.rgb_obs_params
       elif param_type == "depth_obs_params":
           return self.depth_obs_params
    #    elif param_type == "error_params":
    #       return self.error_norm_params
