# Offline CALVIN dataset for critic training and normalization.
# It loads low-dimensional robot states, joint states, actions, rewards,
# and optional visual observations, then prepares transition-aligned samples.

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
        # Store dataset and task configuration.
        self.skill = skill
        self.goal_centered = goal_centered
        self.demos_dir = Path(demos_dir).expanduser()
        self.state_type = self.skill.state_type
        self.dt = self.skill.dt
        self.normalized = self.skill.normalized

        # Print whether normalized samples will be returned.
        print(f"self.normalized in CalvinCriticOfflineDataset: {self.normalized}") # False

        self.norm_range = [-1, 1]
        self.is_quaternion = is_quaternion

        # Placeholders for dataset statistics and trajectory metadata.
        self.X_mins = None
        self.X_maxs = None
        self.fixed_ori = None
        self.start = None
        self.goal = None
        
        # Normalizer used for state, action, joint, and visual features.
        self.normalizer = Normalizer()

        # self.target_selector = TargetSelector()
        # self.target_selector.scan()
        # self.target_dict = self.target_selector.predict_t(T=64)

        # Store optional visual-observation and normalization settings.
        self.use_visual_obs = use_visual_obs
        self.action_params = action_params
        self.state_params = state_params
        self.joint_pos_params = joint_pos_params
        self.rgb_obs_params = rgb_obs_params
        self.depth_obs_params = depth_obs_params    

        # Verify that the demonstration directory exists.
        assert self.demos_dir.is_dir(), "Demos directory does not exist!"

        # Select training-set file paths.
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
         
        # Select validation-set file paths.
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
    
        # Determine which columns correspond to the requested robot state and joints.
        cols = self.get_valid_columns(self.state_type)
        joint_pos_col = self.get_valid_columns("joint")

        # Load robot state trajectory and joint-position trajectory.
        self.X = np.load(self.robot_low_dim_obs_data_file)[:, :, cols]
        self.Q = np.load(self.robot_low_dim_obs_data_file)[:, :, joint_pos_col]

        print(f"joint pos data has shape: {self.Q.shape}")

        # Load scene observations.
        self.robot_scene_obs = np.load(self.robot_scene_obs_data_file)
        assert isinstance(self.robot_scene_obs, np.ndarray)

        # Load available RGB observations.
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

        # Extract drawer joint state from scene observations.
        self.drawer_joint_state = self.robot_scene_obs[:,:,1:2] #[N,T,1]

        #print(f"self.drawer_joint_state: {self.drawer_joint_state}") # check the drawer joint state values

        # Build state-based offline rewards from drawer opening length.
        max_reward = 3.0
        self.step_reward = -0.1
        max_drawer_joint_state = 0.24

        # Linearly map drawer state into [0, max_reward].
        self.rewards = (self.drawer_joint_state-0.0)/(max_drawer_joint_state-0.0)*max_reward
        self.rewards = np.clip(self.rewards,0.0,max_reward).astype(np.float32) # state rewards

        # Align rewards with transition tuples [obs_t, action_t, next_obs_t].
        self.transition_rewards = self.rewards[:,:-1,:] # align with self.obs shape [N,T-1,1]

        # Apply a constant step penalty.
        self.transition_rewards = self.transition_rewards + self.step_reward

        # Build terminal flags for each demonstration trajectory.
        self.done = np.zeros_like(self.transition_rewards)
        self.done[:,-1,:] = 1
        self.done = torch.from_numpy(self.done)

        self.robot_rgb_gripper_obs_data_file

        # Get the last orientation from the trajectory (this is bad for orientation dependant tasks)
        cols_ori = self.get_valid_columns("ori")
        temp_ori = np.load(self.robot_low_dim_obs_data_file)[:, :, cols_ori]
        self.fixed_ori = temp_ori[0, -1, :]
   
        # Convert Euler orientation to quaternion when requested.
        if self.state_type == "ori" and self.is_quaternion:
            self.X = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X)#p.getQuaternionFromEuler接受三元向量，返回四元数

        elif self.state_type == "pos_ori" and self.is_quaternion:
            oris = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X[:, :, 3:])
            self.X = np.concatenate([self.X[:, :, :3], oris], axis=-1)

        # Estimate average start and final Cartesian positions across demonstrations.
        self.start = np.mean(self.X[:, 0, :3], axis=0)
        self.goal = np.mean(self.X[:, -1, :3], axis=0)

        # Optionally express each trajectory relative to its own final goal.
        if self.goal_centered:
            # Make X goal centered i.e., subtract each trajectory with its goal
            self.X[:, :, :3] = self.X[:, :, :3] - np.expand_dims(self.X[:, -1, :3], axis=1)



        # Build action trajectory from finite differences of robot state.
        self.dX = np.zeros_like(self.X) # same shape

        # Cartesian velocity/action between consecutive states.
        self.dX[:, :-1, :3] = (self.X[:, 1:, :3] - self.X[:, :-1, :3]) / self.dt
        
        # Final action has no next state, so it is set to zero.
        self.dX[:, -1, :3] = 0

        # Build transition-aligned observation pairs.
        self.obs = self.X[:,:-1,:] # T = 63
        self.next_obs = self.X[:,1:,:]  # T = 63   

        # Current and next action sequences.
        self.action = self.dX[:,:-1,:] # align with self.obs shape [N,T-1,D]
        self.action = self.dX[:, :-1, :]          # [N,63,D]
        self.next_action = self.dX[:, 1:, :]      # [N,63,D]

        # Current and next joint-position sequences.
        self.joint_pos = self.Q[:, :-1, :]        # [N,63,Dj]

        print(f"joint_pos has shape: {self.joint_pos.shape}")

        self.next_joint_pos = self.Q[:, 1:, :]    # [N,63,Dj]]

        # Align visual observations with current transition states.
        for k,v in self.robot_rgb_obs_data.items():
           if self.robot_rgb_obs_data[k] is not None: 
                self.robot_rgb_obs_data[k] = v[:,:-1,:,:,:]

        # Separate orientation from position when using pos_ori state type.
        if self.state_type == "pos_ori":
            self.Ori = self.X[:, :, 3:]
            self.Ori = torch.from_numpy(self.Ori).type(torch.FloatTensor)
            self.X = self.X[:, :, :3]
        
        # self.error_norm_params = self.normalizer.fit(self.error,mode="gaussian")
        # self.n_error = self.normalizer.normalize(self.error, self.error_norm_params)

        # Optionally flatten [trajectory, time, feature] into [sample, feature].
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
        
        # Fit action normalization parameters if they were not provided externally.
        if self.action_params is None:
               self.action_params = self.normalizer.fit(self.dX,mode)

        # Fit state normalization parameters.
        if self.state_params is None:
               self.state_params = self.normalizer.fit(self.X,mode)

        # Fit joint-position normalization parameters.
        if self.joint_pos_params is None:
            self.joint_pos_params = self.normalizer.fit(self.joint_pos,mode)

        # Fit normalization parameters for visual observations when enabled.
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

        # Number of available dataset samples.
        return self.obs.shape[0]
        

    def __getitem__(self, idx):

        # Load current observation.
        obs_np = self.obs[idx] #[idx,T,D]
        
        # Convert current observation to torch tensor and normalize it.
        obs_np = torch.from_numpy(obs_np) # tensor
        obs_norm = self.normalizer.normalize(obs_np,self.state_params)

        #print(f"obs_norm has shape {obs_norm.shape}")

        # Load and normalize next observation.
        next_obs_np= self.next_obs[idx]
        next_obs_np = torch.from_numpy(next_obs_np) # tensor
        next_obs_norm = self.normalizer.normalize(next_obs_np,self.state_params)

        # Load current action.
        action_np = self.action[idx] 
        action_np = torch.from_numpy(action_np) # tensor

        # Test normalization and inverse normalization consistency.
        x = action_np.clone().float()              # 原始 action（未归一化）
        xn = self.normalizer.normalize(x, self.action_params)
        xr = self.normalizer.unnormalize(xn, self.action_params)

        # Load current and next joint positions.
        joint_pos = self.joint_pos[idx]
        next_joint_pos = self.next_joint_pos[idx]

        joint_pos_np = torch.from_numpy(joint_pos)

        # Normalize current joint position.
        joint_pos_norm = self.normalizer.normalize(joint_pos,self.joint_pos_params)

        next_joint_pos_np = torch.from_numpy(next_joint_pos)

        # Normalize current action.
        action_norm = self.normalizer.normalize(action_np,self.action_params) 

        # Load and normalize next action.
        next_action_np = self.next_action[idx]
        next_action_np = torch.from_numpy(next_action_np) # tensor
        next_action_norm = self.normalizer.normalize(next_action_np,self.action_params)
        
        # Load transition reward.
        reward_np = self.transition_rewards[idx] # get reward for each timestep,state reward,[63 steps]
        reward = torch.from_numpy(reward_np)

        # error = self.error[idx]

        # Prepare optional visual observations.
        if self.use_visual_obs:

           #rgb_static =self.robot_rgb_obs_data["rgb_static"][idx]
           #rgb_static = rgb_static.astype(np.float32)/255.0
           #rgb_static_norm = self.normalizer.normalize(rgb_static,self.rgb_static_obs_params)

           # Load and scale gripper RGB image into [0,1].
           rgb_gripper = self.robot_rgb_obs_data["rgb_gripper"][idx]
           rgb_gripper = rgb_gripper.astype(np.float32)/255.0

           # Normalize gripper image using fitted visual statistics.
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
           
           # Return normalized transition data when normalization is enabled.
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

           # Return raw transition data when normalization is disabled.
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

        # Dataset configuration without visual observations.
        else:

         # Return normalized low-dimensional transitions.
         if self.normalized:
          return{
                "obs":obs_norm, 
                "action":action_norm,
                "reward":reward,
                "next_obs":next_obs_norm,
                "next_action":next_action_norm,
                "done":self.done[idx],
                 }# return a dict {}

         # Return raw low-dimensional 
