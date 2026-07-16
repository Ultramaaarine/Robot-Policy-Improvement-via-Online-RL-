# low_dim_cond encoder
# cond are normalized
# pos : [B,] 1D tensor
# emb: [B,emb_dim]: 2D tensor 
# pos is a tensor which can be defined in policy as a 
# torch.randint(0,self.training_timesteps,(action.shape[0],),device=action.device)
import math
import torch
import torch.nn as nn
from typing import Optional

# embedder for diffusion step: random timestep input pos
class SinusoidalEmbbeder(nn.Module):
 def __init__(self,emb_dim):
  super().__init__()
  assert emb_dim % 2 ==0
  self.half_dim = emb_dim // 2
  self.i = torch.arange(0,self.half_dim)
  self.frequencies = torch.exp(-math.log(10000)*self.i/self.half_dim)
  
 def forward(self,pos:torch.Tensor):#pos 是 diffusion model 的 timestep
  pos = pos.to(dtype=torch.get_default_dtype())
  frequencies = self.frequencies.to(dtype=pos.dtype,device=pos.device)
  args = pos[:,None]*frequencies[None,:]
  emb = torch.concatenate([torch.sin(args),torch.cos(args)],dim=-1)

  return emb
 
# embedder for transformer input: sequential input x
class PositionalEmbeder(nn.Module):
 def __init__(self, emb_dim, max_len=512):
        super().__init__()
        assert emb_dim % 2 == 0

        half_dim = emb_dim // 2
        i = torch.arange(0, half_dim)

        frequencies = torch.exp(-math.log(10000) * i / half_dim)

        # 生成位置
        pos = torch.arange(0, max_len).float()[:, None]   # [T,1] 0 - 512 random
        args = pos * frequencies[None, :]                 # [T,half_dim]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [T,D]
        emb = emb.unsqueeze(0)   # [1,T,D]

        self.register_buffer("emb", emb)

 def forward(self, x):
        # x: [B, T, D]
        T = x.shape[1]
        return x + self.emb[:, :T, :]

