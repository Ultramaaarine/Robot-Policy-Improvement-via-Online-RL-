import torch.nn as nn
import torch
from typing import Optional
from torch.nn import functional as F


class DownSamplingBlock(nn.Module):
 def __init__(self, down_input_dim, 
              down_output_dim,
              cond_dim:Optional[int],
              enable_cond:Optional[bool],
              
              kernel_size):
  
  super().__init__()
  self.enable_cond = enable_cond
  
  self.downsampling_block1 = nn.Sequential(
   nn.Conv1d(down_input_dim,down_output_dim,kernel_size,stride=2,padding=1),
   nn.GroupNorm(num_groups=8,num_channels=down_output_dim),
   nn.ReLU())
  self.downsampling_block2 = nn.Sequential(
   nn.Conv1d(down_output_dim,down_output_dim,kernel_size,stride = 1,padding = 1),
   nn.GroupNorm(num_groups=8,num_channels=down_output_dim), # is pooling necessary?
   nn.ReLU()
  )

   
  if self.enable_cond:
   assert cond_dim is not None 
   self.cond_dim = cond_dim
   self.cond_proj_layer = nn.Sequential(
    nn.Mish(),
    nn.Linear(cond_dim,down_input_dim) # cond is a encoded tensor down input dim : 32,64,128
   )
  else: self.cond_proj_layer, self.cond_dim = None, None

 def forward(self,x:torch.Tensor,cond:Optional[torch.Tensor]): # cond is a tensor
  
  #print(f"downblock input x has shape: {x.shape}") #[32,32,64]
  if self.cond_proj_layer and cond is not None:
   cond = self.cond_proj_layer(cond) # this cond is the timestep_cond_emb tensor from TimeCond class
    #经过 projection layer后，cond在每一个downblock会变成：[32,64,32] ,[32,64,64],[32,64,128] [32,64,256]
   cond  = cond.permute(0,2,1).contiguous()#[32,32,64] ,[32,64,64],[32,12,648] [32,256,64]
   if cond.size(-1) != x.size(-1):
    cond = F.interpolate(cond,size=x.size(-1),mode="nearest") #[32,32,64],[32,64,32],[32,128,16],[32,256,8] 与action相同
    #print(f"cond has dim {cond.shape}")
   x = x + cond
   #print(f"out has shape: {out.shape}")
  out = self.downsampling_block1(x)
  out = self.downsampling_block2(out) 
 
  return out
  
  
class MiddleBlock(nn.Module):
 def __init__(self, 
              mid_input_dim,
              mid_output_dim,
              
              cond_dim:Optional[int],
              enable_cond:Optional[bool] = False,
              kernel_size = 3,stride = 1,padding:Optional[int] = None):
  super().__init__()
  if padding is None:
   padding = (kernel_size-1)//2
  self.enable_cond = enable_cond
  self.mid_block1 = nn.Sequential(
   nn.Conv1d(mid_input_dim,mid_output_dim,kernel_size,stride,padding),
   nn.GroupNorm(num_groups=8,num_channels=mid_output_dim),
   nn.ReLU(),
   
  )
  self.mid_block2 = nn.Sequential(
   nn.Conv1d(mid_output_dim,mid_output_dim,kernel_size,stride,padding),
   nn.GroupNorm(num_groups=8,num_channels=mid_output_dim),
   nn.ReLU()
  )
  if self.enable_cond:
   assert cond_dim is not None
   self.cond_proj_layer = nn.Sequential(
    nn.Mish(),
    nn.Linear(cond_dim, mid_output_dim)
  )
  else: self.cond_proj_layer = None

 def forward(self,x,cond:Optional[torch.Tensor]):
 # print(f"input of middleblock has shape: {x.shape}")
  out = self.mid_block1(x)
  if self.cond_proj_layer is not None and cond is not None:
   
   cond = self.cond_proj_layer(cond) #residual modulation FiLM  is also possible
   cond = cond.permute(0,2,1).contiguous()
   if cond.size(-1) != out.size(-1):
    cond = F.interpolate(cond,size=out.size(-1),mode="nearest")
   out = out + cond
 # print(f"output of Unet middle block has shpae {out.shape}")#[32,256,8]
  out = self.mid_block2(out)
  return out


class UpSamplingBlock(nn.Module):
 def __init__(self, 
              up_input_dim, 
              up_output_dim,
              cond_dim:Optional[int],
              
              enable_cond:Optional[bool] = False,
              kernel_size=3):
  super().__init__()
  self.enable_cond = enable_cond
  self.upsampling_block1 = nn.Sequential(
   nn.ConvTranspose1d(up_input_dim,
                      up_output_dim,
                      kernel_size,stride=2,padding=1,output_padding=1,bias=False),
   nn.GroupNorm(num_groups=8,num_channels=up_output_dim),
   nn.ReLU()
   
  )
  if self.enable_cond:
   assert cond_dim is not None
   self.cond_proj_layer = nn.Sequential(
    nn.Mish(),
    nn.Linear(cond_dim,up_input_dim)
   )
  else:
   self.cond_proj_layer = None
  
  self.upsampling_block2 = nn.Sequential(
   nn.Conv1d(up_output_dim,up_output_dim,kernel_size,stride=1,padding=1),
   nn.GroupNorm(num_groups=8,num_channels=up_output_dim),
   nn.ReLU()
  )

 def forward(self,x:torch.Tensor,cond:Optional[torch.Tensor]):
  #print(f"input of Upsamplingblock has shape: {x.shape}")
  if self.cond_proj_layer and cond is not None:
   cond = self.cond_proj_layer(cond)
   cond = cond.permute(0,2,1).contiguous()
   if cond.size(-1) != x.size(-1):
    cond = F.interpolate(cond, x.size(-1))
   x = cond + x

  out = self.upsampling_block1(x)
   
  out = self.upsampling_block2(out)
  #print(f"output of Upsampling block has shape {out.shape}")
  return out
