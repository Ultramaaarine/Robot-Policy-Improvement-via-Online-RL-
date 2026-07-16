import os
import sys
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

cwd_path = Path(__file__).absolute().parents[0]
root = cwd_path.parents[0]
sys.path.insert(0, root.as_posix())
logger = logging.getLogger(__name__)
os.chdir(cwd_path)


def load_demostrations(datamodule, mode):
    if mode == "training":
        data_loader = datamodule.train_dataloader()

    elif mode == "validation":
        data_loader = datamodule.val_dataloader()
        #print("len(validation_dataset)=",len(data_loader.dataset))
        #print("val batch_size   =", getattr(data_loader, 'batch_size', None))
    split_iter = iter(data_loader)
    #print(f"length of dataloader is {len(split_iter)}") ##
    demos = []
    for i in range(len(split_iter)): # pl 的dataloader 包装了一层len
       demo = next(split_iter)
       demos += [demo["robot_obs"].numpy()]
    demos = np.concatenate(demos,axis = 0) #沿着第 0 维，把这些 (B, S, D) 的数组拼在一起，变成一个 (N, S, D) 的大数组。
       
    logger.info(f"Dimensions of {mode} demostrations (NxSxD):{demos.shape}.")
    return demos
def load_rgb_obs_data(datamodule,mode):
     if mode == "training":
      data_loader = datamodule.train_dataloader()
     elif mode == "validation":
      data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     rgb_static_datas = []
     rgb_gripper_datas = []
     rgb_tactile_datas = []
     for i in range(len(split_iter)):
         rgb_data = next(split_iter)
         rgb_static_datas.append(rgb_data["rgb_obs"]["rgb_static"].numpy())
         
         rgb_gripper_datas.append(rgb_data["rgb_obs"]["rgb_gripper"].numpy())
         
         rgb_tactile_datas.append(rgb_data["rgb_obs"]["rgb_tactile"].numpy())
     rgb_gripper_datas = np.concatenate(rgb_gripper_datas,axis=0)
     rgb_static_datas = np.concatenate(rgb_static_datas,axis=0)
     rgb_tactile_datas = np.concatenate(rgb_tactile_datas,axis=0)
     rgb_obs_list = [rgb_static_datas, rgb_gripper_datas, rgb_tactile_datas]
     logger.info(f"Dimension of rgb_static normalization data (NxSxD):{rgb_obs_list[0].shape}.")
     logger.info(f"Dimension of rgb_gripper normalization data (NxSxD):{rgb_obs_list[1].shape}.")
     logger.info(f"Dimension of rgb_tactile normalization data (NxSxD):{rgb_obs_list[2].shape}.")
     return rgb_obs_list

def load_depth_obs_data(datamodule,mode):
     if mode == "training":
         data_loader = datamodule.train_dataloader()
     elif mode == "validation":
         data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     depth_static_normalization_datas = []
     depth_gripper_normalization_datas = []
     depth_tactile_normalization_datas = []
     for i in range(len(split_iter)):
         normalization_data = next(split_iter)
         depth_static_normalization_datas.append(normalization_data["depth_obs"]["depth_static"].numpy())
         
         depth_gripper_normalization_datas.append(normalization_data["depth_obs"]["depth_gripper"].numpy())
         
         depth_tactile_normalization_datas.append(normalization_data["depth_obs"]["depth_tactile"].numpy())
             
     depth_static_normalization_datas = np.concatenate(depth_static_normalization_datas,axis=0)
     depth_gripper_normalization_datas = np.concatenate(depth_gripper_normalization_datas,axis=0)
     depth_tactile_normalization_datas = np.concatenate(depth_tactile_normalization_datas,axis=0)
     depth_obs_norm_list = [depth_static_normalization_datas, depth_gripper_normalization_datas, depth_tactile_normalization_datas]
     logger.info(f"Dimension of depth_static normalization data (NxSxD):{depth_obs_norm_list[0].shape}.")
     logger.info(f"Dimension of depth_gripper normalization data (NxSxD):{depth_obs_norm_list[1].shape}.")
     logger.info(f"Dimension of depth_tactile normalization data (NxSxD):{depth_obs_norm_list[2].shape}.")    
     return depth_obs_norm_list
def load_scene_obs_data(datamodule,mode):
     if mode == "training":
         data_loader = datamodule.train_dataloader()
     elif mode == "validation":
         data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     scene_obs_norm_data = []
     for i in range(len(split_iter)):
         normalization_data = next(split_iter)
         scene_obs_norm_data += [normalization_data["scene_obs"].numpy()]
     scene_obs_norm_data = np.concatenate(scene_obs_norm_data,axis=0)   
     return  scene_obs_norm_data    

@hydra.main(version_base="1.1",config_path="../config",config_name="extract_calvin_demos")
def extract_demos(cfg:DictConfig)->None:
    seed_everything(cfg.seed,workers=True)
    cfg.log_dir = Path(cfg.log_dir).expanduser()
    cfg.demos_dir = Path(cfg.demos_dir).expanduser()
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.demos_dir,exist_ok=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule) #实例化数据模块
    datamodule.setup(stage = "fit")#挂载数据集
    
    # 1) Lowdim Obs
    p1 = Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "low_dim_obs")
    p1.mkdir(parents=True,exist_ok=True)
    mode = ["training", "validation"]

    for m in mode:
        demos = load_demostrations(datamodule,m)
        
        save_dir = p1/m # /low_dim_obs/training
        np.save(save_dir,demos)


    # 2) RGB数据

    p2 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "rgb_obs"))
    p2.mkdir(parents=True,exist_ok=True)
    rgb_name = ["rgb_static", "rgb_gripper", "rgb_tactile"]
    for name in rgb_name:
        (p2/name).mkdir(parents=True,exist_ok=True)   
    for m in mode: 
       rgb_obs_list = load_rgb_obs_data(datamodule,m)
       for name, arr in zip(rgb_name,rgb_obs_list):
          np.save(p2/name/m,arr)
      

    # 3) Depth数据
    p3 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "depth_obs"))
    p3.mkdir(parents=True, exist_ok=True)
    depth_name = ["depth_static","depth_gripper","depth_tactile"]
    for name in depth_name:
        (p3/name).mkdir(parents=True,exist_ok=True)   
    for m in mode: 
       depth_obs_list = load_depth_obs_data(datamodule,m)
       for name, arr in zip(depth_name,depth_obs_list):
          np.save(p3/name/m,arr)
      

    # 4) Scene obs数据
    p4 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "scene_obs"))
    p4.mkdir(parents=True,exist_ok=True)
    for m in mode:
      scene_obs_data = load_scene_obs_data(datamodule,m)    
      np.save(p4/m, scene_obs_data)

if __name__ =="__main__":
   extract_demos()
