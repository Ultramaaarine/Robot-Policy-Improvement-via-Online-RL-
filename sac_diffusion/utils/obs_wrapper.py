import torch
from collections import deque


class OBSWrapper():
    def __init__(self,context_len:int,obs_dim:int):
        self.context_len = context_len
        self.obs_dim = obs_dim
        # 核心：固定长度的队列，当满了之后，新元素进入，最旧元素自动弹出
        self.history_deque = deque(maxlen=self.context_len)

    def reset(self,first_obs:torch.Tensor): 
         
         if not isinstance(first_obs,torch.Tensor):
              first_obs = torch.from_numpy(first_obs)
              
         assert first_obs.shape == (self.obs_dim,)  # first obs from env, not from demo
         self.history_deque.clear()
         for _ in range(self.context_len):
             self.history_deque.append(first_obs.clone()) # [D,] tensor 

    def get_first_input(self):
         """
         convert first obs list to [1,T,D]
         call reset() before call this
         self.history_deque: a list with T first obs
         """ 
         first_input = torch.stack(list(self.history_deque),dim=0)
         first_input = first_input.unsqueeze(0) #[1,T,D]
         
         return first_input

    
    def update_obs(self,new_obs): # new obs from env.step(action), normalized, encoded
          
          assert isinstance(new_obs,torch.Tensor)
          self.history_deque.append(new_obs.clone())
          obs_seq_tensor = torch.stack(list(self.history_deque),dim=0) #[T,D]
          obs_seq_tensor_final = obs_seq_tensor.unsqueeze(0)  # [1,T,D]

          return obs_seq_tensor_final
    