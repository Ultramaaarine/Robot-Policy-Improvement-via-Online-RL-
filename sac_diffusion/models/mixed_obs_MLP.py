# Encode mixed low-dimensional and optional visual observations together with
# the diffusion timestep, then project them into the diffusion condition space.

#1. step-wise policy, not sequential policy like "diffusion policy visuomotor policy" algorithm
#2. use MLP as network
#3. input dim = [B*T,D] flattened [B,T,D]
#4. for each interaction step use sampling to generate an action
#5. obs are first normalized

# 本模块将观测特征与扩散时间步嵌入拼接，再通过 MLP 得到条件特征。
# 注意：当前实现直接处理 [B, T, D]，没有显式展平为 [B*T, D]。
# nn.Linear 作用于最后一个维度，因此同一套 MLP 会分别处理各个时刻。
# 这里的 MLP 不在时间维上进行信息交互；整个策略是否建模时序，还取决于后续网络。
# 本模块没有执行观测归一化，输入是否已归一化由调用方负责。

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict
from typing import List, Optional
from sac_diffusion.utils.robot_mixed_obs_encoder import Robot_Mixed_Obs_Encoder
from sac_diffusion.utils.positional_embedder import SinusoidalEmbbeder
from sac_diffusion.utils.obs_wrapper import OBSWrapper 

# 混合观测条件编码模块：
# 1. 编码低维观测和可选视觉观测；
# 2. 将扩散时间步转换为向量；
# 3. 拼接两类特征，并映射到指定的输出维度。
class Mixed_Obs_MLP(nn.Module):
 def __init__(self,           
              # 扩散时间步嵌入的维度；虽然参数名包含 pos，这里用于时间步编码。
              pos_emb_dim,

              # 每个时刻的低维条件输入维度。
              cond_dim: Optional[int],

              
              # 是否启用视觉观测。
              enable_visual_obs,
              # 低维观测经过编码后的特征维度。
              low_dim_obs_encoder_out_dim,
              # 视觉观测经过编码后的特征维度。
              visual_obs_feature_dim:Optional[int],
              # 本模块最终输出的条件特征维度。
              diffusion_input_dim,
              # 是否只使用视觉观测，具体处理由 Robot_Mixed_Obs_Encoder 实现。
              use_visual_obs_only:bool, 
              # 是否组合观测特征，具体组合方式由观测编码器实现。
              combine_obs:bool, 
              # 后续融合 MLP 的隐藏层宽度，默认 256。
              middle_dim = 256
              ):   # input: tensor [B*T,D] as action cond with cond action
  # 初始化父类，使子网络和参数能够被 PyTorch 注册和管理。
  super().__init__()

  # Dimension of the low-dimensional condition input.
  # 保存低维条件的原始输入维度。
  self.cond_dim = cond_dim # dim of lowdim_condition

  # Dimension of the diffusion timestep embedding.
  # 保存扩散时间步编码后的向量长度。
  self.t_emb_dim = pos_emb_dim

  # Output dimension of the low-dimensional observation encoder.
  # 保存低维观测编码器的输出维度。
  self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim # dim of encoded lowdim_obs

  # Whether visual observations are enabled.
  # 保存视觉输入开关。
  self.enable_visual_obs = enable_visual_obs

  # Feature dimension of encoded visual observations.
  # 开启视觉时使用配置的视觉特征维度；关闭视觉时设为 0。
  # 这样构建融合 MLP 时，不会为未启用的视觉特征预留输入维度。
  if self.enable_visual_obs:
     self.visual_obs_feature_dim = visual_obs_feature_dim # dim of encoded visual_obs
  elif not self.enable_visual_obs:
     self.visual_obs_feature_dim = 0

  # Final feature dimension passed into the diffusion model.
  # 输出是供后续网络使用的条件特征，不是最终动作。
  self.diffusion_input_dim = diffusion_input_dim

  # Whether low-dimensional and visual observation features are combined.
  # 将组合观测的设置传递给底层观测编码器。
  self.combine_obs = combine_obs

  # Whether only visual observations are used as condition input.
  # 将纯视觉模式的设置传递给底层观测编码器。
  self.use_visual_obs_only = use_visual_obs_only

  # Mixed observation encoder for low-dimensional and optional visual inputs.
  # 创建观测编码器，负责处理低维信息及可选图像信息。
  # forward() 要求该编码器返回三维张量 [B, T, D_obs]。
  self.obs_encoder = Robot_Mixed_Obs_Encoder(        
              self.use_visual_obs_only, 
              self.cond_dim, # dim of lowdim_cond
              self.low_dim_obs_encoder_out_dim,
              self.visual_obs_feature_dim,
              self.enable_visual_obs, 
              self.combine_obs, 
              )

  # Sinusoidal embedding for the diffusion timestep.
  # 将每个样本的扩散时间步编码为 self.t_emb_dim 维向量。
  # 扩散时间步表示加噪/去噪进度，与观测序列中的时刻编号不同。
  self.timestep_embedder = SinusoidalEmbbeder(self.t_emb_dim)
  
  # Fuse observation features with diffusion timestep embedding
  # and project them into the diffusion condition dimension.
  # 融合网络的输入维度：
  # 低维观测特征维度 + 视觉特征维度 + 扩散时间步嵌入维度。
  # 因此观测编码器实际输出的末尾维度必须与前两项之和一致；
  # 使用纯视觉或其他组合模式时，也需要满足这一维度约定。
  self.mlp = nn.Sequential(
   # 第一层将拼接后的特征映射到隐藏层。
   nn.Linear(self.low_dim_obs_encoder_out_dim + self.visual_obs_feature_dim + self.t_emb_dim, middle_dim),
   # SiLU 非线性激活：SiLU(x) = x * sigmoid(x)。
   nn.SiLU(),
   # 第二层继续在隐藏特征空间中进行变换。
   nn.Linear(middle_dim,middle_dim),
   nn.SiLU(),
   # 第三层输出指定维度的扩散条件特征。
   nn.Linear(middle_dim,self.diffusion_input_dim), # diffusion input_dim is 256
   # 最后一层仍使用 SiLU，因此返回的是激活后的特征。
   nn.SiLU()
  )
  
 # 前向输入：
 # visual_obs：可选视觉观测，具体输入格式由观测编码器决定。
 # low_dim_obs：低维观测，通常为 [B, T, cond_dim]。
 # time_emb：每个样本的扩散时间步，预期为 [B]，此时尚未转换成嵌入向量。
 # 返回：条件特征 [B, T, diffusion_input_dim]。
 def forward(self,visual_obs,low_dim_obs,time_emb): # x is noisy action obs in this nn are normalized, time_emb is [B,] timestep tensor

        # Encode low-dimensional and optional visual observations.
        # 先调用观测编码器，得到每个观测时刻的特征表示。
        # 本函数没有接收带噪动作 x，处理的只是观测条件和扩散时间步。
        encoded_cond = self.obs_encoder.forward(visual_obs=visual_obs,low_dim_obs=low_dim_obs)

        #print(f"encoded cond has shape:{encoded_cond.shape}") # [1,64,128]

        # Extract batch size and temporal length.
        # B 为批大小，T 为观测序列长度；_ 对应观测编码后的特征维度。
        B, T, _ = encoded_cond.shape

        #print(f"time emb has shape: {time_emb.shape}")

        # Convert scalar diffusion timesteps into embedding vectors.
        # 将 [B] 的扩散时间步转换为 [B, t_emb_dim] 的嵌入向量。
        timestep_emb = self.timestep_embedder(time_emb)

        # Broadcast the timestep embedding across all observation timesteps.
        # 先添加时间维：[B, t_emb_dim] -> [B, 1, t_emb_dim]。
        # 再广播为 [B, T, t_emb_dim]，同一样本的所有观测时刻共享同一个扩散步。
        # expand 通常通过视图广播，不会像 repeat 一样直接复制全部数据。
        timestep_emb = timestep_emb[:, None, :].expand(B, T, -1)

        #print(f"timestep_emb vec has shape {timestep_emb.shape}")
        #print(f"encoded cond has shape {encoded_cond.shape}")

        # Concatenate observation features and timestep embedding.
        # 沿最后一个特征维拼接，保留批次维 B 和时间维 T。
        # h 的形状为 [B, T, D_obs + t_emb_dim]。
        h = torch.cat([encoded_cond,timestep_emb],dim = -1)

        # Project the fused condition into diffusion input feature space.
        # MLP 对每个时刻的特征独立应用相同的参数，不改变序列长度。
        # 输出形状为 [B, T, diffusion_input_dim]。
        pred = self.mlp(h)

        #print(f"ObsMLP output has shape: {pred.shape}")

        # 返回融合了观测信息与扩散时间步信息的条件特征。
        return pred
