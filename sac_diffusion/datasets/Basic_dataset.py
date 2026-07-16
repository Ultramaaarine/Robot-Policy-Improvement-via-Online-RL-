import logging
import os
import torch
from typing import Dict, Union, Tuple
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from sac_diffusion.datasets.utils.load_utils import get_transforms

logger = logging.getLogger(__name__)


class BasicDataset(Dataset):

 def __init__(
            self,
            data_dir:Path,
            skill:Path,
            step_len:int,
            train:bool,
            transforms:Dict
 ):
             self.data_dir = data_dir
             self.skill = skill
             self.step_len = step_len
             self.train = train
             self.transforms = transforms

             self.transform_robot_obs = None
             if "robot_obs" in self.transforms:
                 self.transform_robot_obs = get_transforms(self.transforms.robot_obs)

 def __len__(self):
        raise NotImplementedError
        
 def __getitem__(self, index:Union[Tuple[int,int],int])->Dict:
        raise NotImplementedError