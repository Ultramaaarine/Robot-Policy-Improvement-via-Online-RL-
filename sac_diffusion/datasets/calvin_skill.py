# 这个文件将提取出的demo送到GMM里面去训练
# 2025年11月14日 开始将此dataset改为 online offline 混合dataset 
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

class CalvinSkillDataset(Dataset):
    def __init__(
        self,
        skill: DictConfig,
        goal_centered: bool,
        flatten: bool,
        demos_dir: str,
        data_type:str,
        mode:str,
        is_quaternion:bool,
        as_torch:bool = False
        
    ):
        self.skill = skill
        self.goal_centered = goal_centered
        self.demos_dir = Path(demos_dir).expanduser()
        self.is_quaternion = is_quaternion
        self.state_type = self.skill.state_type
        self.dt = self.skill.dt
        self.normalized = self.skill.normalized
        self.norm_range = [-1, 1]
        self.as_torch = as_torch
        self.X_mins = None
        self.X_maxs = None
        self.fixed_ori = None
        self.start = None
        self.goal = None
        
        self.normalizer = Normalizer()
        assert self.demos_dir.is_dir(), "Demos directory does not exist!"
        if data_type =="training":
         self.data_file = glob.glob(str(self.demos_dir / self.skill.name / "low_dim_obs" / "training.npy"))[0]
         #self.scene_obs_file = glob.glob(str(self.demos_dir / self.skill.name / "scene_obs" / "training.npy"))
        elif data_type == "validation":
         self.data_file = glob.glob(str(self.demos_dir / self.skill.name / "low_dim_obs"/ "validation.npy"))[0]
         #self.scene_obs_file = glob.glob(str(self.demos_dir / self.skill.name / "scene_obs" / "validation.npy"))
        cols = self.get_valid_columns(self.state_type)
        self.X = np.load(self.data_file)[:, :, cols]

        # Get the last orientation from the trajectory (this is bad for orientation dependant tasks)
        cols1 = self.get_valid_columns("ori")
        temp_ori = np.load(self.data_file)[:, :, cols1]
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

        if self.state_type == "pos_ori":
            self.Ori = self.X[:, :, 3:]
            self.Ori = torch.from_numpy(self.Ori).type(torch.FloatTensor)
            self.X = self.X[:, :, :3]

        if flatten:
            B, S, L = self.X.shape
            self.X = self.X.reshape(B * S, L)
            self.dX = self.dX.reshape(B * S, L)
        # self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        # self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)
        if self.normalized:
         self.action_params = self.normalizer.fit(self.dX,mode)
         self.state_params = self.normalizer.fit(self.X,mode)
         self.dX = self.normalizer.normalize(self.dX,self.action_params)
         self.X = self.normalizer.normalize(self.X,self.state_params)
         
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_np = self.X[idx]
        dX_np = self.dX[idx]
        if self.as_torch:
         self.X[idx] = torch.from_numpy(X_np)
         self.dX[idx] = torch.from_numpy(dX_np)
           
        return self.X[idx], self.dX[idx] # return a batch of [T,D]
    
    def get_valid_columns(self, state_type):
     if "joint" in state_type:
        return np.arange(7, 14)
     elif "pos_ori_gripact" in state_type:
        return np.r_[0:6, 14]
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
        self.X = np.load(self.data_file)
        new_X = np.delete(self.X, list_indicis, axis=0)
        np.save(self.data_file, new_X)
    
