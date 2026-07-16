#一、 三种归一化：1.最大值 最小值归一化 2.高斯归一化 3.零中心数据的没用offset的归一化
#二、 三种归一化表达方式为：
# 1 scale = output_max-output_min/input_max-input_min offset = output_mim- input_min*(scale)
# 2 scale = 1/std offset = -mean/std
# 3 scale = min(abs(output_max),abs(output_min)) / max(abs(input_max),abs(input_min)) offset = zero
#三、要注意细节 比如当输入的最大值和最小值相差不大时 比如相差值小于 1e-4 这时应该把  input_max-input_min 
# 改为 output_max - output_min 以避免除数过小 offset 改为 offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
#四、归一化模块继承自 DictofTensorMixin 或许我的normalizer可以不继承自这个类
#
import torch
import numpy as np
from typing import Dict, Union


def _to_float_tensor(x,*,dtype = None, device = None):
  """
  convert numpy array to tensor
  
  :param x: input data 
  :param dtype: Description
  :param device: Description
  """
  if isinstance(x,np.ndarray):
   x = torch.from_numpy(x)
  if not torch.is_floating_point(x):
   x = x.float()
  if dtype is not None:
   x = x.to(dtype=dtype)
  if device is not None:
   x = x.to(device)
  return x


class Normalizer():
 def __init__(self):
  pass
 
 @torch.no_grad()
 def fit(self,data:Union[np.ndarray,torch.Tensor],
         mode:str,
         last_n_dim=1,
         thresh_hole = 1e-4,
         output_max = 1,
         output_min = -1,
         fit_offset = True,
         dtype:torch.dtype = torch.float32,
         device: Union[str,torch.device, None]=None)->Dict[str, torch.Tensor]:
  
  x0 = _to_float_tensor(data,dtype=dtype,device=device) # convert numpy to tensor
 
  assert last_n_dim >= 0
  if last_n_dim ==0:
   dim = 1
  else:
   dim = np.prod(x0.shape[-last_n_dim:])
  
  x = x0.reshape(-1,dim)
  
  mean = x.mean(dim=0) 
  std = x.std(dim=0,unbiased=False)
  input_max = torch.amax(x,dim=0,keepdim=False)
  input_min = torch.amin(x,dim=0,keepdim=False)
  #input_max = data.max(dim=0)
  #input_min = data.min(dim=0) 请注意 最大值和最小值并非统计量 直接使用 data.max(dim = 0)会出现返回值为 （value,idx）的元组
  if not torch.all(input_max >= input_min):
   raise ValueError("input_max < input_min detected")

  input_range = input_max-input_min
  ignore_dim = input_range < thresh_hole
  input_range[ignore_dim] = output_max-output_min
  
  if mode == "min_max":
   scale = (output_max-output_min)/input_range
   offset = output_min-input_min*scale 
   offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
  
  elif mode == "gaussian":
   ignore_dim = std.abs()< thresh_hole
   safe_std = std.clone()
   safe_std[ignore_dim] = 1.0
   scale = 1/safe_std
   scale[ignore_dim] = 1
   if fit_offset:
    offset = -mean/safe_std
    offset[ignore_dim] = 0.0
   else:
    offset = torch.zeros_like(mean)

  scale  = scale.to(dtype=x0.dtype, device=x0.device)
  offset = offset.to(dtype=x0.dtype, device=x0.device)
  input_min  = input_min.to(dtype=x0.dtype, device=x0.device)
  input_max  = input_max.to(dtype=x0.dtype, device=x0.device)
  mean = mean.to(dtype=x0.dtype, device=x0.device)
  std  = std.to(dtype=x0.dtype, device=x0.device)

  params = {
   "scale":scale,
   "offset": offset,
   "input_stats":{
     "min":input_min,
     "max":input_max,
     "std":std,
     "mean":mean
   },
   "dim":dim,
   "last_n_dim":last_n_dim
  }
  
  return params
 
 def normalize(self,x:Union[torch.Tensor,np.ndarray],params:Dict):
  assert 'scale' in  params and 'offset' in params 
  
  scale = params['scale']
  offset = params['offset']
  dim = int(params.get('dim',scale.shape[0]))
  last_n_dim = int(params.get('last_n_dim',1))
  x = _to_float_tensor(x,dtype=scale.dtype,device = scale.device)
  src_shape = x.shape
  if last_n_dim ==0:
   x = x.reshape(-1,1)
  else:
   x = x.reshape(-1,dim)
  x  = x*scale + offset

  if last_n_dim == 0:
   x = x.reshape(src_shape)
  else:
   x = x.reshape(*src_shape[:-last_n_dim],*src_shape[-last_n_dim:])
  return x
 
 
 def unnormalize(self, x: Union[torch.Tensor, np.ndarray], params: Dict):
    assert 'scale' in params and 'offset' in params
    scale  = params['scale']
    offset = params['offset']
    dim = int(params.get('dim', scale.shape[0]))
    last_n_dim = int(params.get('last_n_dim', 1))

    x = _to_float_tensor(x, dtype=scale.dtype, device=scale.device)
    src_shape = x.shape

    if last_n_dim == 0:
        x = x.reshape(-1, 1)
    else:
        x = x.reshape(-1, dim)

    # 避免除 0（理论上不会为 0，这里仅作健壮性防护）
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x = (x - offset) / safe_scale

    if last_n_dim == 0:
        x = x.reshape(src_shape)
    else:
        x = x.reshape(*src_shape[:-last_n_dim], *src_shape[-last_n_dim:])
    return x
  
 def online_normalize(self,x:Union[np.ndarray,torch.Tensor]):
    pass