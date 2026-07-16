import torch
from sac_diffusion.models.base_replay_buffer import BaseReplayBuffer


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, obs_shape, action_dim, device="cpu"):
        super().__init__()
        #convert dictconfig to int
        if  isinstance(obs_shape,int):
            obs_shape = (int(obs_shape),) # tuple
        else: 
            obs_shape = tuple(int(x) for x in obs_shape)
        if isinstance( action_dim, (list,tuple)):
           assert len(action_dim) == 1, f"action_dim should be int, got {action_dim}"
           action_dim = int(action_dim[0])
        else:
            action_dim = int(action_dim)

        self.size = 0
        self.ptr = 0
        self.capacity = int(capacity)
        self.device = device

        # single-step buffer
        self.obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32) # *obs_shape 是解包运算符，为了让代码能自动适配不同维度的 observation shape。 (6,) 才是 torch.zeros 最喜欢的格式。
        self.next_obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32)
        self.act_buf = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        self.rew_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.done_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)
        self.next_joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)
        # sequence buffer
        self.seq_capacity = 20000
        self.seq_ptr = 0
        self.seq_len = 63
        self.seq_size = 0

        self.seq_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )
        self.seq_joint_pos_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 7),
            dtype=torch.float32
        )
        self.seq_next_joint_pos_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,7),
            dtype=torch.float32
        )
        self.seq_act_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, action_dim),
            dtype=torch.float32
        )
        self.seq_visual_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 3,84,84),
            dtype=torch.uint8
        )
        self.seq_reward_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,1),
            dtype=torch.float32
        )
        self.seq_next_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )
        self.seq_done_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,1),
            dtype=torch.float32    
        )
    def save_transitions(self, transition): 
        """
        transition: {
                "obs": obs, 
                "joint_pos"
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "done": done,
        }
        """
        obs = torch.as_tensor(transition["obs"]["position"], dtype=torch.float32) # pos obs from env
        joint_pos = torch.as_tensor(transition["obs"]["joints"],dtype= torch.float32)
        next_obs = torch.as_tensor(transition["next_obs"]["position"], dtype=torch.float32)
        next_joint_pos = torch.as_tensor(transition["next_joint"]["joints"])
        action = torch.as_tensor(transition["action"], dtype=torch.float32)
        reward = torch.as_tensor(transition["reward"], dtype=torch.float32).view(1)
        done = torch.as_tensor(transition["done"], dtype=torch.float32).view(1)

        self.obs_buf[self.ptr] = obs
        self.joint_pos_buf[self.ptr] = joint_pos
        self.next_obs_buf[self.ptr] = next_obs
        self.next_joint_pos_buf[self.ptr] = next_joint_pos
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def load_transitions(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            "obs": self.obs_buf[idx].to(self.device),
            "actions": self.act_buf[idx].to(self.device),
            "rewards": self.rew_buf[idx].to(self.device),
            "next_obs": self.seq_next_obs_buf[idx].to(self.device),
            "dones": self.seq_done_buf[idx].to(self.device),
        }

    def save_sequence(self, seq): # transition seq a list of 63 len
        """
        seq: transition seq a list of 63 len
        """
        assert len(seq) == self.seq_len, f"Expected seq_len={self.seq_len}, got {len(seq)}"

    # ===== 1. 检查 visual 是否完整 =====
        has_visual = True
        for t in seq:
            obs = t.get("obs", {})
            if "rgb_gripper" not in obs or obs["rgb_gripper"] is None:
                has_visual = False
                break

    # ===== 2. 存 pos =====
        states = [torch.as_tensor(t["obs"]["position"], dtype=torch.float32) for t in seq]
        seq_states = torch.stack(states, dim=0)
        self.seq_obs_buf[self.seq_ptr] = seq_states

    # ===== 3. 存 action =====
        actions = [torch.as_tensor(t["action"], dtype=torch.float32) for t in seq]
        seq_actions = torch.stack(actions, dim=0)
        self.seq_act_buf[self.seq_ptr] = seq_actions
    # =====4. 存 joint pos =====
        joints = [torch.as_tensor(t["obs"]["joints"],dtype=torch.float32)]
        seq_joints = torch.stack(joints,dim = 0)
        self.seq_joint_pos_buf[self.ptr] = seq_joints
  
    # ===== 5. 存 visual（如果有）=====
        if has_visual:
            visual_obs = [t["obs"]["rgb_gripper"] for t in seq]

            visual_obs_stack = torch.stack(
                [torch.from_numpy(v) for v in visual_obs], dim=0
            )  # [T,C,H,W]

           
            visual_obs_stack = visual_obs_stack.to(torch.uint8)

            self.seq_visual_obs_buf[self.seq_ptr] = visual_obs_stack

    # ===== 6.更新 reward =====
        rewards = [torch.as_tensor(t["reward"], dtype=torch.float32).view(1) for t in seq]
        self.seq_reward_buf[self.seq_ptr] = torch.stack(rewards, dim=0)

    # ===== 7.更新 next_obs =====   
        next_states = [torch.as_tensor(t["next_obs"]["position"], dtype=torch.float32) for t in seq]
        self.seq_next_obs_buf[self.seq_ptr] = torch.stack(next_states, dim=0)

    # ===== 8. 更新 next_joint_pos =====
        next_joint_pos = [torch.as_tensor(t["next_obs"]["joints"], dtype=torch.float32) for t in seq]
        self.seq_next_joint_pos_buf[self.seq_ptr] = torch.stack(next_joint_pos)

    # ===== 9. 更新done =====
        dones = [torch.as_tensor(t["done"], dtype=torch.float32).view(1) for t in seq]
        self.seq_done_buf[self.seq_ptr] = torch.stack(dones, dim=0)

    # ===== 10. 更新指针 =====
        self.seq_ptr = (self.seq_ptr + 1) % self.seq_capacity
        self.seq_size = min(self.seq_size + 1, self.seq_capacity)
        #print(f"[save_sequence] seq_ptr={self.seq_ptr}, seq_size={self.seq_size}")

    def _assemble_all_sequence_batch(self,batch_size)->dict:
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"
        idx = torch.randint(0,self.seq_size,(batch_size,))
        batch = {
            "obs": self.seq_obs_buf[idx],
            "joint_pos": self.seq_joint_pos_buf[idx],
            "action":self.seq_act_buf[idx],
            "next_obs": self.seq_next_obs_buf[idx],
            "next_joint_pos":self.seq_next_joint_pos_buf[idx],
            "reward": self.seq_reward_buf[idx],
            "done": self.seq_done_buf[idx],
            "gripper_obs":{"rgb_gripper":self.seq_visual_obs_buf[idx]}
        }
        return batch

    def _assemble_state_sequence_batch(self, return_action: bool, batch_size)-> torch.Tensor: 
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"
        idx = torch.randint(0, self.seq_size, (batch_size,))

        state_batch = self.seq_obs_buf[idx].to(self.device)  # [B, T, obs_dim]

        if return_action:
            action_batch = self.seq_act_buf[idx].to(self.device)  # [B, T, action_dim]
            return torch.cat([state_batch, action_batch], dim=-1)

        return state_batch

    def _assemble_visual_sequence_batch(self, batch_size):
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"
        idx = torch.randint(0, self.seq_size, (batch_size,))
        batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0  # [B, T, 3, H, W]
        return batch

    def _assemble_visual_state_sequence_batch(self, batch_size):
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"
        idx = torch.randint(0, self.seq_size, (batch_size,))

        visual_batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0
        state_batch = self.seq_obs_buf[idx].to(self.device)

        return {"visual": visual_batch, "state": state_batch}

    def load_sequence(self, seq_type: str, batch_size):
        """
        seq_type:visual,state,visual_state,state-action
        """
        if seq_type == "visual":
            return self._assemble_visual_sequence_batch(batch_size)
        elif seq_type == "state":
            return self._assemble_state_sequence_batch(False, batch_size)
        elif seq_type == "visual_state":
            return self._assemble_visual_state_sequence_batch(batch_size)
        elif seq_type == "state-action":
         
            return self._assemble_state_sequence_batch(True, batch_size)
        elif seq_type =="all":
            return self._assemble_all_sequence_batch(batch_size)
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return {
            "obs": self.obs_buf[idx].to(self.device),
            "actions": self.act_buf[idx].to(self.device),
            "rewards": self.rew_buf[idx].to(self.device),
            "next_obs": self.next_obs_buf[idx].to(self.device),
            "dones": self.done_buf[idx].to(self.device),
        }