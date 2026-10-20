# Replay buffer for storing both single-step transitions and fixed-length rollout sequences.
# It supports state, joint, action, reward, done, and optional visual observations,
# and provides different batch formats for training.

# 使用 PyTorch 张量预分配存储空间，并通过索引抽样组成训练批次。
import torch
from sac_diffusion.models.base_replay_buffer import BaseReplayBuffer


# 经验回放同时维护两套独立缓存：单步转移缓存和固定长度轨迹缓存。
# 两套缓存的容量、写入指针和有效数量分别维护，保存序列不会自动增加单步缓存数量。
class ReplayBuffer(BaseReplayBuffer):
    # capacity 为单步容量，obs_shape 为每步观测形状，action_dim 为动作维度。
    # device 指定部分采样接口的返回设备；下面预分配张量未显式指定 device，通常创建在 CPU。
    def __init__(self, capacity, obs_shape, action_dim, device="cpu"):
        super().__init__()

        # Convert configuration values into standard Python int / tuple formats.
        #convert dictconfig to int
        # 将整数观测维度转换为单元素元组；可迭代配置则逐项转换成整数。
        if  isinstance(obs_shape,int):
            obs_shape = (int(obs_shape),) # tuple
        else: 
            obs_shape = tuple(int(x) for x in obs_shape)

        # Ensure action_dim is represented as a single integer.
        # 动作维度可传整数或单元素列表/元组，最终统一为整数。
        if isinstance( action_dim, (list,tuple)):
           assert len(action_dim) == 1, f"action_dim should be int, got {action_dim}"
           action_dim = int(action_dim[0])
        else:
            action_dim = int(action_dim)

        # Current number of stored transitions and write pointer.
        # size 为单步有效数量，ptr 为下一次单步写入位置。
        self.size = 0
        self.ptr = 0

        # Maximum capacity of the single-step buffer.
        # 单步写满后循环覆盖旧数据，最多保留 capacity 条转移。
        self.capacity = int(capacity)

        # Device used when sampled data is returned.
        # 记录目标返回设备；all 序列采样接口没有执行设备迁移。
        self.device = device

        # single-step buffer
        # Store individual transitions for standard replay-buffer sampling.
        # 预分配当前观测、下一观测、动作、奖励、结束标志及七维关节状态的单步缓存。
        self.obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32) # *obs_shape 是解包运算符，为了让代码能自动适配不同维度的 observation shape。 (6,) 才是 torch.zeros 最喜欢的格式。
        self.next_obs_buf = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32)
        self.act_buf = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        # 奖励与 done 每步保留一个分量，形状均为 [capacity, 1]。
        self.rew_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.done_buf = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)
        self.next_joint_pos_buf = torch.zeros((self.capacity,7),dtype = torch.float32)

        # sequence buffer
        # Separate buffer for complete fixed-length rollout trajectories.
        # 序列容量固定为 20000 条，独立于构造参数 capacity。
        self.seq_capacity = 20000
        # 下一条完整轨迹的写入位置。
        self.seq_ptr = 0
        # 每条轨迹固定包含 63 步；保存时要求长度完全一致。
        self.seq_len = 63
        # 当前已保存的有效完整轨迹数量。
        self.seq_size = 0

        # Observation sequence buffer: [N, T, obs_dim].
        # 序列观测缓存形状为 [轨迹容量, 轨迹长度, *obs_shape]。
        self.seq_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )

        # Joint-position sequence buffer.
        # 当前关节状态缓存，每个时刻固定为 7 维。
        self.seq_joint_pos_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 7),
            dtype=torch.float32
        )

        # Next joint-position sequence buffer.
        # 下一关节状态缓存，与当前关节序列按转移时刻对应。
        self.seq_next_joint_pos_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,7),
            dtype=torch.float32
        )

        # Action sequence buffer.
        # 动作序列缓存形状为 [20000, 63, action_dim]。
        self.seq_act_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, action_dim),
            dtype=torch.float32
        )

        # Gripper-camera image sequence buffer.
        # 预分配所有轨迹的夹爪图像，按 [N, T, C, H, W] 保存为 uint8。
        # 此数组包含 20000×63×3×84×84 个字节，约 26.7 GB；即使没有视觉输入也会分配。
        self.seq_visual_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, 3,84,84),
            dtype=torch.uint8
        )

        # Reward sequence buffer.
        # 为每条轨迹的每个时刻保存一个奖励值。
        self.seq_reward_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,1),
            dtype=torch.float32
        )

        # Next observation sequence buffer.
        # 保存与各转移对应的下一观测，而不是在当前观测缓存中临时移动索引。
        self.seq_next_obs_buf = torch.zeros(
            (self.seq_capacity, self.seq_len, *obs_shape),
            dtype=torch.float32
        )

        # Done flag sequence buffer.
        # 结束标志用 float32 保存，通常 False 对应 0、True 对应 1。
        self.seq_done_buf = torch.zeros(
            (self.seq_capacity,self.seq_len,1),
            dtype=torch.float32    
        )


    # 保存一个单步转移；obs 和 next_obs 按包含 position、joints 的字典读取。
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
        # 读取当前末端位置并转换为 float32，要求形状与 obs_shape 兼容。
        obs = torch.as_tensor(transition["obs"]["position"], dtype=torch.float32) # pos obs from env
        joint_pos = torch.as_tensor(transition["obs"]["joints"],dtype= torch.float32)
        next_obs = torch.as_tensor(transition["next_obs"]["position"], dtype=torch.float32)
        # 注意原实现从独立的 next_joint 字段读取关节，而非 next_obs；保留原逻辑。
        next_joint_pos = torch.as_tensor(transition["next_joint"]["joints"])
        action = torch.as_tensor(transition["action"], dtype=torch.float32)
        # 将标量奖励和结束标志整理为长度 1 的张量，以匹配单步缓存形状。
        reward = torch.as_tensor(transition["reward"], dtype=torch.float32).view(1)
        done = torch.as_tensor(transition["done"], dtype=torch.float32).view(1)

        # Store the transition at the current circular-buffer pointer.
        # 把当前转移的各字段写入相同单步索引，保证对应关系。
        self.obs_buf[self.ptr] = obs
        self.joint_pos_buf[self.ptr] = joint_pos
        self.next_obs_buf[self.ptr] = next_obs
        self.next_joint_pos_buf[self.ptr] = next_joint_pos
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        # Advance the circular write pointer.
        # 通过取模让写入位置在 [0, capacity - 1] 内循环，满容量后覆盖旧转移。
        self.ptr = (self.ptr + 1) % self.capacity

        # Increase current buffer size until capacity is reached.
        # 有效数量随写入增加，但不超过单步容量。
        self.size = min(self.size + 1, self.capacity)


    # 随机抽取单步批次；调用前需要 self.size 大于 0。
    def load_transitions(self, batch_size):

        # Randomly sample individual transitions.
        # 有放回抽样，同一条转移可能在一个批次中重复出现。
        idx = torch.randint(0, self.size, (batch_size,))

        # Move sampled tensors to the configured device.
        return {
            "obs": self.obs_buf[idx].to(self.device),
            "actions": self.act_buf[idx].to(self.device),
            "rewards": self.rew_buf[idx].to(self.device),
            # 注意：本接口原实现读取的是序列缓存，返回维度和来源与单步 obs 不一致；此处仅注释。
            "next_obs": self.seq_next_obs_buf[idx].to(self.device),
            # 这里同样读取序列结束标志，而非单步 done_buf。
            "dones": self.seq_done_buf[idx].to(self.device),
        }


    # 保存一条完整轨迹；seq 为按时间排序的转移字典列表。
    def save_sequence(self, seq): # transition seq a list of 63 len
        """
        seq: transition seq a list of 63 len
        """

        # Sequence replay requires exactly seq_len transitions.
        # 只接受恰好 63 步的轨迹，不在这里截断或补齐。
        assert len(seq) == self.seq_len, f"Expected seq_len={self.seq_len}, got {len(seq)}"

    # ===== 1. 检查 visual 是否完整 =====
        # Check whether every transition contains a valid visual observation.
        # 只有所有转移都含非 None 的 rgb_gripper 时，才写入该轨迹图像。
        has_visual = True
        # 逐步检查视觉字段；循环变量 t 在循环结束或提前退出后仍保留最后检查的转移。
        for t in seq:
            obs = t.get("obs", {})
            if "rgb_gripper" not in obs or obs["rgb_gripper"] is None:
                has_visual = False
                break

    # ===== 2. 存 pos =====
        # Stack all position observations into one trajectory tensor.
        # 将各步末端位置转换成张量，再堆叠为 [T, *obs_shape]。
        states = [torch.as_tensor(t["obs"]["position"], dtype=torch.float32) for t in seq]
        seq_states = torch.stack(states, dim=0)
        # 将当前完整轨迹的位置序列写入序列缓存指针位置。
        self.seq_obs_buf[self.seq_ptr] = seq_states

    # ===== 3. 存 action =====
        # Stack all actions into one action trajectory.
        # 逐步转换动作并堆叠为 [T, action_dim]。
        actions = [torch.as_tensor(t["action"], dtype=torch.float32) for t in seq]
        seq_actions = torch.stack(actions, dim=0)
        self.seq_act_buf[self.seq_ptr] = seq_actions

    # =====4. 存 joint pos =====
        # Store joint-position information.
        # 注意：这里没有 for t in seq，只读取上方视觉检查循环遗留的一个 t。
        # 因此得到的是单个关节状态 [1, 7]，并非完整的逐时刻关节序列。
        joints = [torch.as_tensor(t["obs"]["joints"],dtype=torch.float32)]
        seq_joints = torch.stack(joints,dim = 0)
        # 注意：这里使用单步 ptr 而非 seq_ptr；[1, 7] 赋值时会广播到该行所有时刻。
        # 保留原实现，未修改指针或关节采集逻辑。
        self.seq_joint_pos_buf[self.ptr] = seq_joints
  
    # ===== 5. 存 visual（如果有）=====
        # Store the complete visual trajectory only if all frames are available.
        # 仅视觉完整时写图像；否则未清空对应槽位，初始为零，循环覆盖后可能保留旧图像。
        if has_visual:
            # 收集每步图像，下面 torch.from_numpy 要求每个元素为 NumPy 数组。
            visual_obs = [t["obs"]["rgb_gripper"] for t in seq]

            # 沿时间维堆叠为 [T, C, H, W]，需要图像形状与预分配的 [3, 84, 84] 一致。
            visual_obs_stack = torch.stack(
                [torch.from_numpy(v) for v in visual_obs], dim=0
            )  # [T,C,H,W]

            # Store images compactly as uint8.
            # 这里只做类型转换，不乘以 255；若输入已经归一化到 [0, 1]，会损失大部分像素信息。
            visual_obs_stack = visual_obs_stack.to(torch.uint8)

            self.seq_visual_obs_buf[self.seq_ptr] = visual_obs_stack

    # ===== 6.更新 reward =====
        # Store reward sequence.
        # 将每步奖励整理为 [1]，堆叠后的奖励序列为 [T, 1]。
        rewards = [torch.as_tensor(t["reward"], dtype=torch.float32).view(1) for t in seq]
        self.seq_reward_buf[self.seq_ptr] = torch.stack(rewards, dim=0)

    # ===== 7.更新 next_obs =====   
        # Store next-position sequence.
        # 逐步读取 next_obs 中的位置，保存与动作逐项对应的下一状态序列。
        next_states = [torch.as_tensor(t["next_obs"]["position"], dtype=torch.float32) for t in seq]
        self.seq_next_obs_buf[self.seq_ptr] = torch.stack(next_states, dim=0)

    # ===== 8. 更新 next_joint_pos =====
        # Store next joint-position sequence.
        # 此处正确地遍历全部转移，从 next_obs 读取每步下一关节状态。
        next_joint_pos = [torch.as_tensor(t["next_obs"]["joints"], dtype=torch.float32) for t in seq]
        self.seq_next_joint_pos_buf[self.seq_ptr] = torch.stack(next_joint_pos)

    # ===== 9. 更新done =====
        # Store episode termination flags for all timesteps.
        # 逐步读取结束标志，堆叠为 [T, 1]。
        dones = [torch.as_tensor(t["done"], dtype=torch.float32).view(1) for t in seq]
        self.seq_done_buf[self.seq_ptr] = torch.stack(dones, dim=0)

    # ===== 10. 更新指针 =====
        # Advance the sequence circular-buffer pointer.
        # 序列指针独立循环，写满后覆盖旧轨迹。
        self.seq_ptr = (self.seq_ptr + 1) % self.seq_capacity

        # Increase number of stored sequences until sequence capacity is reached.
        # 更新有效轨迹数量，但不超过序列容量。
        self.seq_size = min(self.seq_size + 1, self.seq_capacity)

        #print(f"[save_sequence] seq_ptr={self.seq_ptr}, seq_size={self.seq_size}")


    # 抽取包含所有模态的轨迹字典；该接口保持缓存原始设备和图像 uint8 类型。
    def _assemble_all_sequence_batch(self,batch_size)->dict:

        # Ensure enough complete trajectories exist before sampling.
        # 接口要求有效轨迹数不少于 batch_size；后续抽样仍然是有放回的。
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample sequence indices.
        # 在有效序列范围内有放回抽样，所有字段共用同一组轨迹索引。
        idx = torch.randint(0,self.seq_size,(batch_size,))

        # Return all available sequence modalities in one dictionary.
        # 返回位置、关节、动作、下一状态、奖励、done 和嵌套图像字典。
        batch = {
            "obs": self.seq_obs_buf[idx],
            "joint_pos": self.seq_joint_pos_buf[idx],
            "action":self.seq_act_buf[idx],
            # 注意：本接口原实现读取的是序列缓存，返回维度和来源与单步 obs 不一致；此处仅注释。
            "next_obs": self.seq_next_obs_buf[idx],
            "next_joint_pos":self.seq_next_joint_pos_buf[idx],
            "reward": self.seq_reward_buf[idx],
            "done": self.seq_done_buf[idx],
            # 此处返回原始 uint8 图像，没有除以 255，也没有标记哪些轨迹缺失视觉。
            "gripper_obs":{"rgb_gripper":self.seq_visual_obs_buf[idx]}
        }

        return batch


    # 抽取位置状态序列，可选择在最后一个维度拼接动作特征。
    def _assemble_state_sequence_batch(self, return_action: bool, batch_size)-> torch.Tensor: 

        # Ensure enough complete sequences are available.
        # 接口要求有效轨迹数不少于 batch_size；后续抽样仍然是有放回的。
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample trajectory indices.
        # 有放回地抽取轨迹索引，批次中可能出现相同轨迹。
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # State trajectory: [B, T, obs_dim].
        # 将状态序列移动到目标设备，通常形状为 [B, T, obs_dim]。
        state_batch = self.seq_obs_buf[idx].to(self.device)  # [B, T, obs_dim]

        # Optionally concatenate state and action features.
        # 启用时返回状态与动作拼接结果，否则仅返回状态。
        if return_action:
            action_batch = self.seq_act_buf[idx].to(self.device)  # [B, T, action_dim]
            # 沿特征维拼接，普通向量观测下输出为 [B, T, obs_dim + action_dim]。
            return torch.cat([state_batch, action_batch], dim=-1)

        return state_batch


    # 仅抽样视觉轨迹，并将图像转换为目标设备上的归一化浮点张量。
    def _assemble_visual_sequence_batch(self, batch_size):

        # Ensure enough complete sequences are available.
        # 接口要求有效轨迹数不少于 batch_size；后续抽样仍然是有放回的。
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Randomly sample visual trajectories.
        # 有放回地抽取轨迹索引，批次中可能出现相同轨迹。
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # Convert uint8 visual observations into normalized float images.
        # uint8 图像转换为 float 后除以 255，输出形状为 [B, T, 3, 84, 84]。
        batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0  # [B, T, 3, H, W]

        return batch


    # 用相同的轨迹索引同时读取图像和状态，使两种模态保持对齐。
    def _assemble_visual_state_sequence_batch(self, batch_size):

        # Ensure enough complete sequences are available.
        # 接口要求有效轨迹数不少于 batch_size；后续抽样仍然是有放回的。
        assert self.seq_size >= batch_size, f"Not enough sequences: have {self.seq_size}, need {batch_size}"

        # Use the same sampled indices for visual and state trajectories.
        # 有放回地抽取轨迹索引，批次中可能出现相同轨迹。
        idx = torch.randint(0, self.seq_size, (batch_size,))

        # Normalize visual observations.
        # 视觉分支进行 float 转换和 [0, 1] 缩放，状态分支只迁移设备。
        visual_batch = self.seq_visual_obs_buf[idx].to(self.device).float() / 255.0

        # Load corresponding state trajectories.
        # 将状态序列移动到目标设备，通常形状为 [B, T, obs_dim]。
        state_batch = self.seq_obs_buf[idx].to(self.device)

        # 返回 visual/state 两个字段，不同于 all 接口中的嵌套 gripper_obs 格式。
        return {"visual": visual_batch, "state": state_batch}


    # 按 seq_type 选择采样结果格式；支持 visual、state、visual_state、state-action 和 all。
    def load_sequence(self, seq_type: str, batch_size):
        """
        seq_type:visual,state,visual_state,state-action
        """

        # Select which type of sequence batch should be returned.
        # 仅返回归一化图像序列。
        if seq_type == "visual":
            return self._assemble_visual_sequence_batch(batch_size)

        # 仅返回位置状态序列，不包含动作。
        elif seq_type == "state":
            return self._assemble_state_sequence_batch(False, batch_size)

        # 返回图像与状态字典。
        elif seq_type == "visual_state":
            return self._assemble_visual_state_sequence_batch(batch_size)

        # 返回沿特征维拼接的状态动作序列。
        elif seq_type == "state-action":
         
            return self._assemble_state_sequence_batch(True, batch_size)

        # 返回所有序列字段；注意其设备、图像类型及字典布局与其他模式不同。
        elif seq_type =="all":
            return self._assemble_all_sequence_batch(batch_size)

        else:
            # 对不支持的序列类型显式报错。
            raise ValueError(f"Unknown sequence type: {seq_type}")


    # len(buffer) 仅返回单步数量 self.size，不返回已存轨迹数 self.seq_size。
    def __len__(self):

        # Length refers to the number of stored single-step transitions.
        return self.size


    # 通过 buffer[idx] 读取指定单步转移，支持 Python 索引或单元素张量索引。
    def __getitem__(self, idx):

        # Convert tensor index into a Python integer if necessary.
        # 张量索引通过 item() 转成 Python 标量，要求张量只有一个元素。
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Return one single-step transition.
        return {
            "obs": self.obs_buf[idx].to(self.device),
            "actions": self.act_buf[idx].to(self.device),
            "rewards": self.rew_buf[idx].to(self.device),
            # 此处读取单步 next_obs_buf，与 load_transitions 中使用的序列缓存不同。
            "next_obs": self.next_obs_buf[idx].to(self.device),
            "dones": self.done_buf[idx].to(self.device),
        }
