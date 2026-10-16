# Replay buffer for storing both single-step transitions and fixed-length rollout sequences.
# It supports state, joint, action, reward, done, and optional visual observations,
# and provides different batch formats for training.

import torch
from sac_diffusion.models.base_replay_buffer import BaseReplayBuffer


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, obs_shape, action_dim, device="cpu"):
        super().__init__()

        # Convert configuration values into standard Python int / tuple formats.
        #convert dictconfig to int
        if  isinstance(obs_shape,int):
            obs_shape = (int(obs_shape),) # tuple
        else: 
            obs_shape = tuple(int(x) for x in obs_shape)

        # Ensure action_dim is represented as a single integer.
        if isinstance( action_dim, (list,tuple)):
           assert len(action_dim) == 1, f"action_dim should be int, got {action_dim}"
           action_dim = int(action_dim[0])
        else:
            action_dim = int(action_dim)

        # Current number of stored transitions and write pointer.
        self.size = 0
        self.ptr = 0

        # Maximum capacity of the single-step buffer.
        self.capacity = int(capacity)

        # Device used when sampled data is returned.
        self.device = device

        # single-step buffer
        # Store individual transitions for standard replay-buffer sampling.
        self.obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32) # *obs_shape 是解包运算符，为了让代码能自动适配不同维度的 observation shape。 (6,) 才是 torch.zeros 最喜欢的格式。
        self.next_obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32)
        self.act_buf = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        self.rew_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.done_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)
        self.next_joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)

        # sequence buffer
        # Separate buffer for complete fixed-length rollout trajectories.
        self.seq_capacity = 20000
        self.seq_ptr = 0
        self.seq_len = 63
        self.seq_size = 0

        # Observation sequence buffer: [N, T, obs_dim].
        self.seq_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )

        # Joint-position sequence buffer.
        self.seq_joint_pos_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 7),
            dtype=torch.float32
        )

        # Next joint-position sequence buffer.
        self.seq_next_joint_pos_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,7),
            dtype=torch.float32
        )

        # Action sequence buffer.
        self.seq_act_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, action_dim),
            dtype=torch.float32
        )

        # Gripper-camera image sequence buffer.
        self.seq_visual_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 3,84,84),
            dtype=torch.uint8
        )

        # Reward sequence buffer.
        self.seq_reward_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,1),
            dtype=torch.float32
        )

        # Next observation sequence buffer.
        self.seq_next_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )

        # Done flag sequence buffer.
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

        # Convert one environment transition into torch tensors.
        obs = torch.as_tensor(transition["obs"]["position"], dtype=torch.float32) # pos obs from env
        joint_pos = torch.as_tensor(transition["obs"]["joints"],dtype= torch.float32)
        next_obs = torch.as_tensor(transition["next_obs"]["position"], dtype=torch.float32)
        next_joint_pos = torch.as_tensor(transition["next_joint"]["joints"])
        action = torch.as_tensor(transition["action"], dtype=torch.float32)
        reward = torch.as_tensor(transition["reward"], dtype=torch.float32).view(1)
        done = torch.as_tensor(transition["done"], dtype=torch.float32).view(1)

        # Store the transition at the current circular-buffer pointer.
        self.obs_buf[self.ptr] = obs
        self.joint_pos_buf[self.ptr] = joint_pos
        self.next_obs_buf[self.ptr] = next_obs
        self.next_joint_pos_buf[self.ptr] = next_joint_pos
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        # Advance the circular write pointer.
        self.ptr = (self.ptr + 1) % self.capacity

        # Increase current buffer size until capacity is reached.
        self.size = min(self.size + 1, self.capacity)


    def load_transitions(self, batch_size):

        # Randomly sample individual transitions.
        idx = torch.randint(0, self.size, (batch_size,))

        # Move sampled tensors to the configured device.
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

        # Sequence replay requires exactly seq_len transitions.
        assert len(seq) == self.seq_len, f"Expected seq_len={self.seq_len}, got {len(seq)}"

    # ===== 1. 检查 visual 是否完整 =====
        # Check whether every transition contains a valid visual observation.
        has_visual = True
        for t in seq:
            obs = t.get("obs", {})
            if "rgb_gripper" not in obs or obs["rgb_gripper"] is None:
                has_visual = False
                break

    # ===== 2. 存 pos =====
        # Stack all position observations into one trajectory tensor.
        states = [torch.as_tensor(t["obs"]["position"], dtype=torch.float32) for t in seq]
        seq_states = torch.stack(states, dim=0)
        self.seq_obs_buf[self.seq_ptr] = seq_states

    # ===== 3. 存 action =====
        # Stack all actions into one action trajectory.
        actions = [torch.as_tensor(t["action"], dtype=torch.float32) for t in seq]
        seq_actions = torch.stack(actions, dim=0)
        self.seq_act_buf[self.seq_ptr] = seq_actions

    # =====4. 存 joint pos =====
        # Store joint-position information.
        joints = [torch.as_tensor(t["obs"]["joints"],dtype=torch.float32)]
        seq_joints = torch.stack(joints,dim = 0)
        self.seq_joint_pos_buf[self.ptr] = seq_joints
  
    # ===== 5. 存 visual（如果有）=====
        # Store the complete visual trajectory only if all frames are available.
        if has_visual:
            visual_obs = [t["obs"]["rgb_gripper"] for t in seq]

            visual_obs_stack = torch.stack(
                [torch.from_numpy(v) for v in visual_obs], dim=0
            )  # [T,C,H,W]

            # Store images compactly as uint8.
            visual_obs_stack = visual_obs_stack.to(torch.uint8)

            self.seq_visual_obs_buf[self.seq_ptr] = visual_obs_stack

    # ===== 6.更新 reward =====
        # Store reward sequence.
        rewards = [torch.as_tensor(t["reward"], dtype=torch.float32).view(1) for t in seq]
        self.seq_reward_buf[self.seq_ptr] = torch.stack(rewards, dim=0)

    # ===== 7.更新 next_obs =====   
        # Store next-position sequence.
        next_states = [torch.as_tensor(t["next_obs"]["position"], dtype=torch.float32) for t in seq]
        self.seq_next_obs_buf[self.seq_ptr] = torch.stack(next_states, dim=0)

    # ===== 8. 更新 next_joint_pos =====
        # Store next joint-position sequence.
        next_joint_pos = [torch.as_tensor(t["next_obs"]["joints"], dtype=torch.float32) for t in seq]
        self.seq_next_joint_pos_buf[self.seq_ptr] = torch.stack(next_joint_pos)

    # ===== 9. 更新done =====
        # Store episode termination flags for all timesteps.
        dones = [torch.as_tensor(t["done"], dtype=torch.float32).view(1) for t in seq]
        self.seq_done_buf[self.seq_ptr] = torch.stack(dones, dim=0)

    # ===== 10. 更新指针 =====
        # Advance the sequence circular-buffer pointer.
        self.seq_ptr = (self.seq_ptr + 1) % self.seq_capacity

        # Increase number of stored sequences until sequence capacity is reached.
        self.seq_size = min(self.seq_size + 1, self.seq_capacity)

        #print(f"[save_sequence] seq_ptr={self.seq_ptr}, seq_size={self.seq_size}")


    def _assemble_all_sequence_batch(self,batch_size)->dict:

        # Ensure enough complete trajectories exist before sampling.
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample sequence indices.
        idx = torch.randint(0,self.seq_size,(batch_size,))

        # Return all available sequence modalities in one dictionary.
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

        # Ensure enough complete sequences are available.
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample trajectory indices.
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # State trajectory: [B, T, obs_dim].
        state_batch = self.seq_obs_buf[idx].to(self.device)  # [B, T, obs_dim]

        # Optionally concatenate state and action features.
        if return_action:
            action_batch = self.seq_act_buf[idx].to(self.device)  # [B, T, action_dim]
            return torch.cat([state_batch, action_batch], dim=-1)

        return state_batch


    def _assemble_visual_sequence_batch(self, batch_size):

        # Ensure enough complete sequences are available.
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample visual trajectories.
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # Convert uint8 visual observations into normalized float images.
        batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0  # [B, T, 3, H, W]

        return batch


    def _assemble_visual_state_sequence_batch(self, batch_size):

        # Ensure enough complete sequences are available.
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Use the same sampled indices for visual and state trajectories.
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # Normalize visual observations.
        visual_batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0

        # Load corresponding state trajectories.
        state_batch = self.seq_obs_buf[idx].to(self.device)

        return {"visual": visual_batch, "state": state_batch}


    def load_sequence(self, seq_type: str, batch_size):
        """
        seq_type:visual,state,visual_state,state-action
        """

        # Select which type of sequence batch should be returned.
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

        # Length refers to the number of stored single-step transitions.
        return self.size


    def __getitem__(self, idx):

        # Convert tensor index into a Python integer if necessary.
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Return one single-step transition.
        return {
            "obs": self.obs_buf[idx].to(self.device),
            "actions": self.act_buf[idx].to(self.device),
            "rewards": self.rew_buf[idx].to(self.device),
            "next_obs": self.next_obs_buf[idx].to(self.device),
            "dones": self.done_buf[idx].to(self.device),
        }
