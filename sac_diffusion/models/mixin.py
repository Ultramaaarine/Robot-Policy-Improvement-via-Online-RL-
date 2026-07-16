import torch
import torch.nn as nn
class Mixin(nn.Module):
 def __init__(self, *args, **kwargs):
     super().__init__(*args, **kwargs)
     self.register_buffer("_anchor",torch.tensor(0), persistent=False)

 @property
 def device(self):
  return self._anchor.device
 
 @property
 def dtype(self):
   return self._anchor.dtype
 def to_(self, *args, **kwargs):
   super().to(*args, **kwargs)
   return self
 
 def to_(self,*args,**kwargs):
   super().to(*args,**kwargs)
   return self
 def assert_uniform_device_dtype(self):
   devs,dtypes = set(),set()
   for p in self.parameters(recurse = True):
     devs.add(p.device)
     dtypes.add(p.dtype)
   for b in self.buffers(recurse=True):
     devs.add(b.device)
     dtypes.add(b.dtype)
   if len(devs) > 1 or len(dtypes) >1:
     raise RuntimeError(f"inconsistency device/dtype in the module: device={devs},dtypes={dtypes}")
   return True
