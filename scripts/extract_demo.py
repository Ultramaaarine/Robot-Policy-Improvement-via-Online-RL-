# 导入文件路径、模块搜索路径、配置管理、日志和数组处理工具。
import os
import sys
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

# 获取当前脚本所在目录，不是启动命令时终端所在的目录。
cwd_path = Path(__file__).absolute().parents[0]
# 将脚本目录的上一级目录作为项目根目录。
root = cwd_path.parents[0]
# 把项目根目录加入模块搜索路径首位，方便导入项目内部模块。
sys.path.insert(0, root.as_posix())
# 创建当前模块的日志记录器。
logger = logging.getLogger(__name__)
# 将进程当前工作目录切换到脚本目录，影响后续相对路径的解析。
os.chdir(cwd_path)


# 读取指定数据划分中的机器人低维观测，并合并所有批次。
# mode 支持 "training" 和 "validation"。
# 返回形状通常为 [N, S, D]：
# N 为实际加载的样本数，S 为序列长度，D 为机器人观测维度。
def load_demostrations(datamodule, mode):
    # 根据模式选择训练或验证数据加载器。
    if mode == "training":
        data_loader = datamodule.train_dataloader()

    elif mode == "validation":
        data_loader = datamodule.val_dataloader()
        #print("len(validation_dataset)=",len(data_loader.dataset))
        #print("val batch_size   =", getattr(data_loader, 'batch_size', None))
    # 创建批次迭代器；传入其他 mode 时，原实现没有设置 data_loader。
    split_iter = iter(data_loader)
    #print(f"length of dataloader is {len(split_iter)}") ##
    # 暂存各批次的机器人观测。
    demos = []
    # 遍历加载器提供的批次数量。
    for i in range(len(split_iter)): # pl 的dataloader 包装了一层len
       demo = next(split_iter)
       # 取出 robot_obs 并转换为 NumPy 数组。
       # 直接调用 numpy() 要求张量位于 CPU，且不需要梯度。
       demos += [demo["robot_obs"].numpy()]
    # 沿样本维拼接，保留每个样本的序列维和特征维。
    # 若加载器没有产生任何批次，空列表无法执行 concatenate。
    demos = np.concatenate(demos,axis = 0) #沿着第 0 维，把这些 (B, S, D) 的数组拼在一起，变成一个 (N, S, D) 的大数组。
       
    # 记录合并后的形状，方便确认样本数量和观测维度。
    logger.info(f"Dimensions of {mode} demostrations (NxSxD):{demos.shape}.")
    return demos

# 分别提取静态相机、夹爪相机和触觉图像。
# 这里只读取并合并数组，不执行像素归一化或维度排列转换。
def load_rgb_obs_data(datamodule,mode):
     # 选择需要导出的数据划分。
     if mode == "training":
      data_loader = datamodule.train_dataloader()
     elif mode == "validation":
      data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     # 三个列表分别保存各模态的批次数据。
     rgb_static_datas = []
     rgb_gripper_datas = []
     rgb_tactile_datas = []
     for i in range(len(split_iter)):
         rgb_data = next(split_iter)
         # 读取静态相机图像；要求对应字段存在。
         rgb_static_datas.append(rgb_data["rgb_obs"]["rgb_static"].numpy())
         
         # 读取夹爪相机图像。
         rgb_gripper_datas.append(rgb_data["rgb_obs"]["rgb_gripper"].numpy())
         
         # 读取触觉传感器的图像数据。
         rgb_tactile_datas.append(rgb_data["rgb_obs"]["rgb_tactile"].numpy())
     # 各模态分别沿批次维合并，其他维度保持数据加载器输出的布局。
     rgb_gripper_datas = np.concatenate(rgb_gripper_datas,axis=0)
     rgb_static_datas = np.concatenate(rgb_static_datas,axis=0)
     rgb_tactile_datas = np.concatenate(rgb_tactile_datas,axis=0)
     # 返回顺序固定为 static、gripper、tactile，与保存时的名称列表对应。
     rgb_obs_list = [rgb_static_datas, rgb_gripper_datas, rgb_tactile_datas]
     # 日志中的 NxSxD 是原有描述，实际图像通常还包含通道和空间维度。
     logger.info(f"Dimension of rgb_static normalization data (NxSxD):{rgb_obs_list[0].shape}.")
     logger.info(f"Dimension of rgb_gripper normalization data (NxSxD):{rgb_obs_list[1].shape}.")
     logger.info(f"Dimension of rgb_tactile normalization data (NxSxD):{rgb_obs_list[2].shape}.")
     return rgb_obs_list

# 提取三个传感器对应的深度观测，并按样本维合并。
# 变量名虽然包含 normalization，但本函数没有执行归一化运算。
def load_depth_obs_data(datamodule,mode):
     # 根据模式选择加载器。
     if mode == "training":
         data_loader = datamodule.train_dataloader()
     elif mode == "validation":
         data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     # 分别暂存静态相机、夹爪相机和触觉深度批次。
     depth_static_normalization_datas = []
     depth_gripper_normalization_datas = []
     depth_tactile_normalization_datas = []
     for i in range(len(split_iter)):
         normalization_data = next(split_iter)
         # 提取静态相机深度数据。
         depth_static_normalization_datas.append(normalization_data["depth_obs"]["depth_static"].numpy())
         
         # 提取夹爪相机深度数据。
         depth_gripper_normalization_datas.append(normalization_data["depth_obs"]["depth_gripper"].numpy())
         
         # 提取触觉深度数据。
         depth_tactile_normalization_datas.append(normalization_data["depth_obs"]["depth_tactile"].numpy())
             
     # 沿第 0 维拼接全部批次，不改变深度值、空间尺寸或通道布局。
     depth_static_normalization_datas = np.concatenate(depth_static_normalization_datas,axis=0)
     depth_gripper_normalization_datas = np.concatenate(depth_gripper_normalization_datas,axis=0)
     depth_tactile_normalization_datas = np.concatenate(depth_tactile_normalization_datas,axis=0)
     # 顺序与后面 depth_name 中的名称一致。
     depth_obs_norm_list = [depth_static_normalization_datas, depth_gripper_normalization_datas, depth_tactile_normalization_datas]
     logger.info(f"Dimension of depth_static normalization data (NxSxD):{depth_obs_norm_list[0].shape}.")
     logger.info(f"Dimension of depth_gripper normalization data (NxSxD):{depth_obs_norm_list[1].shape}.")
     logger.info(f"Dimension of depth_tactile normalization data (NxSxD):{depth_obs_norm_list[2].shape}.")    
     return depth_obs_norm_list

# 提取场景状态，例如场景物体或可动部件的状态。
# scene_obs 各维的具体含义由数据集定义。
def load_scene_obs_data(datamodule,mode):
     if mode == "training":
         data_loader = datamodule.train_dataloader()
     elif mode == "validation":
         data_loader = datamodule.val_dataloader()
     split_iter = iter(data_loader)
     # 暂存各批次场景观测。
     scene_obs_norm_data = []
     for i in range(len(split_iter)):
         normalization_data = next(split_iter)
         scene_obs_norm_data += [normalization_data["scene_obs"].numpy()]
     # 合并为完整的场景观测数组；此处同样没有执行归一化。
     scene_obs_norm_data = np.concatenate(scene_obs_norm_data,axis=0)   
     return  scene_obs_norm_data    

# 使用 Hydra 加载 ../config 下名为 extract_calvin_demos 的配置。
# 配置会作为 DictConfig 对象传入 extract_demos。
@hydra.main(version_base="1.1",config_path="../config",config_name="extract_calvin_demos")
def extract_demos(cfg:DictConfig)->None:
    # 设置随机种子，并启用数据加载工作进程相关的种子设置。
    # 这不意味着对同一加载器的多次遍历一定产生相同的样本顺序。
    seed_everything(cfg.seed,workers=True)
    # 将路径转换为 Path，并展开用户目录符号 ~。
    cfg.log_dir = Path(cfg.log_dir).expanduser()
    cfg.demos_dir = Path(cfg.demos_dir).expanduser()
    # 创建日志和演示输出目录；已存在时不报错。
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.demos_dir,exist_ok=True)
    # 按配置中的 _target_ 等信息构造数据模块。
    datamodule = hydra.utils.instantiate(cfg.datamodule) #实例化数据模块
    # 准备训练和验证阶段需要的数据集。
    datamodule.setup(stage = "fit")#挂载数据集
    
    # 1) Lowdim Obs
    # 创建当前技能对应的低维观测输出目录。
    p1 = Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "low_dim_obs")
    p1.mkdir(parents=True,exist_ok=True)
    # 分别导出训练集和验证集。
    mode = ["training", "validation"]

    for m in mode:
        # 加载该划分中的所有机器人观测批次。
        demos = load_demostrations(datamodule,m)
        
        save_dir = p1/m # /low_dim_obs/training
        # np.save 会自动添加 .npy 后缀，生成 training.npy 或 validation.npy。
        np.save(save_dir,demos)


    # 2) RGB数据

    # RGB 数据按技能名和传感器类型分目录保存。
    p2 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "rgb_obs"))
    p2.mkdir(parents=True,exist_ok=True)
    rgb_name = ["rgb_static", "rgb_gripper", "rgb_tactile"]
    for name in rgb_name:
        (p2/name).mkdir(parents=True,exist_ok=True)   
    for m in mode: 
       # 重新遍历加载器，获取三个传感器的 RGB 数组。
       rgb_obs_list = load_rgb_obs_data(datamodule,m)
       # zip 将名称与对应数组一一配对。
       for name, arr in zip(rgb_name,rgb_obs_list):
          # 例如保存到 rgb_obs/rgb_gripper/training.npy。
          np.save(p2/name/m,arr)
      

    # 3) Depth数据
    # 深度数据采用与 RGB 相同的分目录方式。
    p3 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "depth_obs"))
    p3.mkdir(parents=True, exist_ok=True)
    depth_name = ["depth_static","depth_gripper","depth_tactile"]
    for name in depth_name:
        (p3/name).mkdir(parents=True,exist_ok=True)   
    for m in mode: 
       depth_obs_list = load_depth_obs_data(datamodule,m)
       for name, arr in zip(depth_name,depth_obs_list):
          # 例如保存到 depth_obs/depth_static/validation.npy。
          np.save(p3/name/m,arr)
      

    # 4) Scene obs数据
    # 场景状态单独保存在 scene_obs 目录下。
    p4 = Path(Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name / "scene_obs"))
    p4.mkdir(parents=True,exist_ok=True)
    for m in mode:
      scene_obs_data = load_scene_obs_data(datamodule,m)    
      np.save(p4/m, scene_obs_data)

    # 各模态是在不同的加载器遍历中导出的。
    # 如果加载器启用 shuffle，或数据集每次随机采样窗口，
    # 不同文件中相同的第 0 维索引不一定对应同一条样本。
    # 本代码没有额外保存样本 ID 来校验跨模态对齐。

# 直接运行本脚本时启动导出；被其他模块导入时不执行此入口。
if __name__ =="__main__":
   extract_demos()
