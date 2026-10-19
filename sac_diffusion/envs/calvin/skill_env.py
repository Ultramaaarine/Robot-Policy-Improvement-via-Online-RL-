# 导入日志、配置实例化、资源回收、仿真环境和数值计算所需的模块。
import logging
import hydra
import gc
import sys
import os
import pkgutil
import gym
import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np
import math
from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv

# 创建当前模块的日志记录器。
logger = logging.getLogger(__name__)

# here are the indices of data in the observation vector
# here [0,1,2] - gripper position; [3,4,5] - gripper orientation, [14] - gripper action, [6] - gripper width
# 机器人观测向量中的索引：末端位置 0～2、姿态 3～5、关节角 7～13。
CALVIN_POSITION_INDICES = np.array([0, 1, 2])
CALVIN_ORIENTATION_INDICES = np.array([3, 4, 5])
CALVIN_JOINTS_INDICES = np.array([7, 8, 9, 10, 11, 12, 13])
# 拼接位置与姿态索引，得到末端六维位姿对应的索引。
CALVIN_6D_INDICES = np.concatenate([CALVIN_POSITION_INDICES, CALVIN_ORIENTATION_INDICES])

from gym.envs.registration import register

# 向 Gym 注册环境；通过 Gym 创建时，注册的时间上限为 64 步。
register(id="skill-env", entry_point="sac_diffusion.envs.calvin:CalvinSkillEnv", max_episode_steps=64)


# CALVIN 技能环境：在桌面仿真环境上增加技能配置、观测筛选和奖励逻辑。
class CalvinSkillEnv(PlayTableSimEnv):
    # 初始化技能环境；cfg 为整体配置，skill 为技能配置，init_pos 为可选末端初始位置。
    def __init__(self, cfg, skill, init_pos):
        print("===Initializing CalvinSkillEnv ===")
        # 保存技能配置，并将环境配置的顶层内容展开为字典。
        self.skill = skill
        pt_cfg = {**cfg.calvin_env.env}
        # 移除 Hydra 的控制字段，避免将它们作为普通参数传给父类。
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)

        # 根据技能调整场景初始状态，例如执行关抽屉技能前先打开抽屉。
        self.set_scene_cfg(pt_cfg["scene_cfg"])
        
        # 在父类初始化前保存机器人和相机配置，供 load() 使用。
        self.robot_cfg = pt_cfg["robot_cfg"]
        self.cameras_c = pt_cfg["cameras"]
        # 调用父类完成基础环境初始化。
        super(CalvinSkillEnv, self).__init__(**pt_cfg) # still env config

        # 记录机器人基座位姿及当前末端目标位姿，供后续初始化或采样使用。
        self.init_base_pos, self.init_base_orn = self.p.getBasePositionAndOrientation(self.robot.robot_uid)
        self.init_gripper_pos = self.robot.target_pos
        self.init_gripper_orn = self.robot.target_orn
        # 末端位置采样噪声的标准差；obs_space 指定需要输出哪些观测。
        self.ee_noise = np.array(cfg.ee_noise)
        self.obs_space = cfg.obs_space

        self.init_pos = init_pos

        # 保存技能步数上限；本类 step() 没有直接根据该值判断超时。
        self.max_episode_steps = self.skill.max_steps
        # 根据 Hydra 配置创建任务判定器，用来判断技能是否成功。
        self.tasks = hydra.utils.instantiate(cfg.calvin_env.tasks)

        # 保存源环境标记；真实任务关键点在 reset() 中更新。
        self.is_source = cfg.is_source
        self.gt_keypoint = None
        print("===CalvinSkillEnv is initialized===")

    # 清空物理世界，设置重力，再加载机器人、场景及相机。
    def load(self):
        logger.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        logger.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        # 将物理客户端 ID 传给机器人，并将客户端和随机数生成器传给场景。
        self.robot = hydra.utils.instantiate(self.robot_cfg, cid=self.cid)
        self.scene = hydra.utils.instantiate(self.scene_cfg, p=self.p, cid=self.cid, np_random=self.np_random)

        self.robot.load()
        self.scene.load()

        # init cameras after scene is loaded to have robot id available
        # 逐个实例化相机；此时机器人和场景已加载，可以提供对象 ID。
        self.cameras = [
            hydra.utils.instantiate(
                self.cameras_c[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in self.cameras_c
        ]
        #print(f"calvin skill env initialized") no respose, calvin skill is not instantialized
    # 初始化 PyBullet 连接，支持 VR 共享内存、GUI、EGL 和普通无界面模式。
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        # 保存时间步相关配置；下方实际设置的仿真时间步长为 1 / self.bts。
        self.bts = bullet_time_step
        self.rhw = (render_height, render_width)
        # 仅在尚未连接时创建客户端。
        if not self.isConnected():
            # 标记当前环境拥有该客户端，关闭时由本环境负责断开。
            self.ownsPhysicsClient = True
            # VR 模式：连接已有的共享内存物理服务，并开启实时仿真。
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client
                if cid < 0:
                    logger.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            # GUI 模式：创建带可视化窗口的物理客户端。
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    logger.error("Failed to connect to GUI.")
            # EGL 模式：创建无界面客户端，并加载 EGL 插件用于离屏渲染。
            elif self.use_egl:
                options = f"--width={self.rhw[1]} --height={self.rhw[0]}"
                self.p = bc.BulletClient(p.DIRECT, options=options)
                # 设置确定性的重叠物体配对选项。
                self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
                cid = self.p._client
                # 关闭分割、深度和 RGB 调试预览。
                self.p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
