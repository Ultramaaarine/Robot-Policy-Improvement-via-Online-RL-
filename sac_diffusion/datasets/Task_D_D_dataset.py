#=== EPISODE KEYS === 
# actions: shape=(64, 7), dtype=float64 
# rel_actions: shape=(64, 7), dtype=float64 
# robot_obs: shape=(64, 15), dtype=float64 
# scene_obs: shape=(64, 24), dtype=float64 
# rgb_static: shape=(64, 200, 200, 3), dtype=uint8 
# rgb_gripper: shape=(64, 84, 84, 3), dtype=uint8 
# rgb_tactile: shape=(64, 160, 120, 6), dtype=uint8 
# depth_static: shape=(64, 200, 200), dtype=float32 
# depth_gripper: shape=(64, 84, 84), dtype=float32 
# depth_tactile: shape=(64, 160, 120, 2), dtype=float32
import os
import re
import logging
from pathlib import Path
import torch
from typing import Union, List, Dict, Tuple
import numpy as np
from sac_diffusion.models.normalizer import Normalizer
from sac_diffusion.datasets.Basic_dataset import BasicDataset
from sac_diffusion.datasets.utils.load_utils import load_npz

logger = logging.getLogger(__name__)

class CalvinDataset(BasicDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.episode_lookup = self.load_file_indicies(self.data_dir,self.skill)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

    def __len__(self):
        self.num_demos = len(self.episode_lookup)
        return  self.num_demos

    def __getitem__(self,idx:Union[int,Tuple[int,int]])->Dict:
       return self.get_sequences(idx)   
    
    def lookup_naming_pattern(self):
         it  = os.scandir(self.data_dir)
         while True:
             filename = Path(next(it))
             if "npz" in filename.suffix:
                break
         aux_naming_pattern = re.split(r"\d+",filename.stem)
         naming_pattern = [filename.parent / aux_naming_pattern[0],filename.suffix]
         n_digits = len(re.findall(r"\d+",filename.stem)[0])
         assert len(naming_pattern) == 2
         assert n_digits > 0
         return naming_pattern, n_digits
    
    def get_episode_name(self,idx:int)->Path:
         return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")
    
    def zip_sequence(self,start_idx:int, end_idx:int)-> Dict[str, np.ndarray]:
         episodes = [load_npz(self.get_episode_name(file_idx)) for file_idx in range(start_idx,end_idx, self.step_len)]
         episode = {key: np.stack([ep[key]for ep in episodes]) for key, _ in episodes[0].items()}
         
         return episode
    
    def get_sequences(self,idx:int)->Dict:
         info_indx = self.episode_lookup[idx]
         start_file_indx = info_indx[0]
         end_file_indx = info_indx[1]
           
         episode = self.zip_sequence(start_file_indx, end_file_indx)
         print("=== EPISODE KEYS ===")
         for k, v in episode.items():
              print(f"{k}: shape={v.shape}, dtype={v.dtype}")
         print("=====================")                     
         # 全提取出来做归一化
         rgb_static_obs =  torch.tensor(episode["rgb_static"])
         rgb_gripper_obs = torch.tensor(episode["rgb_gripper"])
         rgb_tactile_obs = torch.tensor(episode["rgb_tactile"])
         depth_static_obs = torch.tensor(episode["depth_static"])
         depth_gripper_obs = torch.tensor(episode["depth_gripper"])
         depth_tactile_obs = torch.tensor(episode["depth_tactile"])
         action = torch.tensor(episode["actions"])
         #rel_actions = torch.tensor(episode["rel_actions"])
         scene_obs = torch.tensor(episode["scene_obs"])
         robot_obs = [self.transform_robot_obs(obs) for obs in episode["robot_obs"][:, :15]] # transform from ndarray to tensor 改成 0,15吧
         robot_obs = torch.stack(robot_obs) #[D,]-> [T,D]
         
         robot_obs_batch = {"robot_obs": robot_obs}

         depth_obs_batch = {"depth_obs":{
                 "depth_static": depth_static_obs,
                 "depth_gripper": depth_gripper_obs,
                 "depth_tactile": depth_tactile_obs
                 }
         }
         rgb_obs_batch = {"rgb_obs":
                 {
                 "rgb_static": rgb_static_obs,
                 "rgb_gripper": rgb_gripper_obs,
                 "rgb_tactile": rgb_tactile_obs,
                 }            
         }
         scene_obs_batch = {"scene_obs": scene_obs}
         total_obs_batch = {
                 **robot_obs_batch,
                 **rgb_obs_batch,
                 **depth_obs_batch,
                 **scene_obs_batch
}
         return total_obs_batch
    
    def load_file_indicies(self,data_dir:Path, skill:str)->Tuple[List,List]:
     assert data_dir.is_dir(),f"Not a dir: {data_dir}"
     skill_name = skill.split("_",1)[1]
     episode_lookup = []

     file_name = data_dir / "lang_annotations" / "auto_lang_ann.npy"

     data = np.load(file_name, allow_pickle=True).reshape(-1)[0]

     all_eps_idx_part_task = [i for (i,v) in enumerate(data["language"]["task"]) if v == skill_name] # 选出 特定skill数据序列的索引编号 data["language"]["task"]是一个列表 内容是 [skill_name,skill_name,skill_name...]
     all_eps_start_end_part_task = [data["info"]["indx"][i]for i in all_eps_idx_part_task] #选出索引标号对应的数据序列的编号 比如第 i 条数据对应 [0000123, 0000167]

     for i in range(len(all_eps_start_end_part_task)):
            if all_eps_start_end_part_task[i][1] - all_eps_start_end_part_task[i][0] == 64: #选出长度为64的序列
                episode_lookup.append(all_eps_start_end_part_task[i])

     logger.info(f"Found {len(episode_lookup)} demonstrations of skill {skill_name}.")
     return episode_lookup




