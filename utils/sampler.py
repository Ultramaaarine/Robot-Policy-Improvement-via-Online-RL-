#重要： RL 微调阶段应该只用 ReplayBuffer sampler 
#因为 RL 的 critic / V / advantage 不需要序列，反而不能用序列，会让 TD 错乱。
#SequenceSampler = offline 模仿（BC）用的
#ReplayBuffer sampler = RL（critic / V / policy 微调）用的 二者没有冲突，但用途完全不同。 
# 在 diffusion policy 论文中 建模目标是连续序列
# 可保留此sampler做扩展 训练一个生成连续序列的模型
import torch
from sac_diffusion.models.replay_buffer import LowdimReplayBuffer
import numpy as np

class Sampler(object):
  
  def __init__(self,batch_size:int):
    self.replaybuffer =  LowdimReplayBuffer()
    self.batch_size = batch_size
  def sampling(self):
    batch  = self.replaybuffer.load_transitions(self.batch_size)
    samples = dict()
    idx = np.random.randint()
    return samples