
# 使用 PyTorch 构造时序窗口，NumPy 负责随机采样和标量截断，DictConfig 表示配置对象。
import torch
from typing import Optional, Union
import numpy as np
from omegaconf import DictConfig


# 从整批轨迹中提取“历史观测 → 未来动作”训练窗口，并构建阶段、目标和掩码信息。
# B 为批大小，T 为完整轨迹长度，To 为观测窗口长度，Ta 为动作窗口长度，Da 为动作维度。
# 本实现整批样本共享同一个采样时刻 t_i，不为每条轨迹单独采样。
def build_sliding_window_batch(
    # 配置提供窗口长度、阶段边界、时间采样方式和目标偏移等参数。
    cfg: DictConfig,
    # 输入包含观测、下一观测和动作；奖励、结束标志、关节和视觉信息可选。
    batch: dict[str, torch.Tensor],
    # 环境提供的接近/把手目标，支持单个目标或按批次提供目标。
    env_goal_pos,
    # 可传入整数或仅含一个元素的列表/数组/张量；为 None 时随机采样时刻。
    t: Optional[Union[int, list[int], np.ndarray, torch.Tensor]] = None,
    # 可显式覆盖动作窗口长度 Ta，不影响观测长度 To。
    horizon: Optional[int] = None,
    
):
    """
    Build history-observation -> future-action window.

    obs window:
        [t-To+1, ..., t]

    action window:
        [t, ..., t+Ta-1]
    """

    # Observation horizon To and action horizon Ta.
    # The config supports both the newer and legacy parameter names.
    # 优先使用 obs_hor_len，兼容旧配置 obs_horizon，两者均缺省时取 4。
    To = int(getattr(cfg.training, "obs_hor_len", getattr(cfg.training, "obs_horizon", 4)))
    # 动作长度依次取 horizon、act_hor_len、action_horizon，最终默认 16。
    Ta = int(horizon if horizon is not None else getattr(cfg.training, "act_hor_len", getattr(cfg.training, "action_horizon", 16)))

    # Stage boundaries:
    # stage 0: reaching
    # stage 1: alignment
    # stage 2: pulling
    # 阶段边界按轨迹索引划分：接近阶段 0、对齐阶段 1、拉动阶段 2。
    reach_end = int(getattr(cfg.training, "reach_end_t", 32))
    align_end = int(getattr(cfg.training, "align_end_t", 36))

    # Future offsets used to construct stage-dependent goals.
    # 拉动阶段的局部目标向未来偏移 future_offset，再叠加 pull_extra_offset。
    future_offset = int(getattr(cfg.training, "future_offset", Ta))
    pull_extra_offset = int(getattr(cfg.training, "pull_extra_offset", 4))
    # 拉动方向估计的起点，默认等于对齐阶段结束时刻。
    pull_start_t_cfg = int(getattr(cfg.training, "pull_start_t", align_end))

    # Read current and next observations.
    # 兼容两套数据字段名：优先 state/next_state，否则使用 obs/next_obs。
    obs = batch["state"] if "state" in batch else batch["obs"]
    next_obs = batch["next_state"] if "next_state" in batch else batch["next_obs"]

    # Optional robot joint states.
    # 可选关节数据缺失时为 None；后面虽提取窗口，但当前没有写入返回条件。
    joint_pos = batch.get("joint_pos", None)
    next_joint_pos = batch.get("next_joint_pos", None)

    # Optional wrist/gripper RGB observation.
    # 默认不使用视觉；存在嵌套 gripper_obs 中的 rgb_gripper 时读取其序列。
    rgb_gripper_obs = None
    if "gripper_obs" in batch and batch["gripper_obs"].get("rgb_gripper", None) is not None:
        rgb_gripper_obs = batch["gripper_obs"]["rgb_gripper"]

    # Action and transition information.
    # 动作是必需字段；其余转移信息允许缺失并保持 None。
    action = batch["action"]
    reward = batch.get("reward", None)
    done = batch.get("done", None)
    next_action = batch.get("next_action", None)

    # B: batch size, T: trajectory length, D: observation dimension.
    # 观测必须为 [B, T, D]，当前仅支持 D 为 3 或 6；后续索引在观测设备上创建。
    B, T, D = obs.shape
    device = obs.device

        # -------- normalize env_goal_pos to [B, 1, 3] --------
    # Convert the environment goal to a float tensor on the same device.
    # 将环境目标统一成 float32 张量，并放到观测所在设备。
    if not isinstance(env_goal_pos, torch.Tensor):
        env_goal_pos = torch.as_tensor(env_goal_pos, device=device, dtype=torch.float32)
    else:
        env_goal_pos = env_goal_pos.to(device=device, dtype=torch.float32)

    # Handle a single XYZ goal: [3] -> [B,1,3].
    # 单个三维坐标 [3] 先变为 [1, 1, 3]，再沿批次维广播到 [B, 1, 3]。
    if env_goal_pos.ndim == 1:
        env_goal_pos = env_goal_pos.view(1, 1, 3).expand(B, 1, 3)

    # 二维输入按 [1, 3] 或 [B, 3] 处理，为每条轨迹保留一个目标。
    elif env_goal_pos.ndim == 2:
        # [1,3] or [B,3]
        # Broadcast a single goal across the whole batch if necessary.
        # 只提供一份目标时，让同一批的所有轨迹共用该目标。
        if env_goal_pos.shape[0] == 1 and B > 1:
            env_goal_pos = env_goal_pos.expand(B, -1)

        # 不能广播成当前批大小时显式报错。
        if env_goal_pos.shape[0] != B:
            raise ValueError(f"env_goal_pos batch {env_goal_pos.shape[0]} != B {B}")

        env_goal_pos = env_goal_pos[:, None, :]  # [B,1,3]

    # 三维输入允许携带目标序列，但当前实现只保留其第一个时间位置。
    elif env_goal_pos.ndim == 3:
        # [B,1,3] or [B,T,3]
        # Broadcast the batch dimension if only one goal sequence is given.
        # 只提供一份目标时，让同一批的所有轨迹共用该目标。
        if env_goal_pos.shape[0] == 1 and B > 1:
            env_goal_pos = env_goal_pos.expand(B, -1, -1)

        # 不能广播成当前批大小时显式报错。
        if env_goal_pos.shape[0] != B:
            raise ValueError(f"env_goal_pos batch {env_goal_pos.shape[0]} != B {B}")

        # Keep only one goal position per trajectory.
        # 即使传入 [B, T, 3]，后续也使用每条轨迹的首个目标，而不按时间切片目标序列。
        env_goal_pos = env_goal_pos[:, :1, :]  # [B,1,3]

    else:
        raise ValueError(f"Bad env_goal_pos shape: {env_goal_pos.shape}")

    # Keep only XYZ coordinates.
    # 只保留前三个坐标分量，并转换为内存连续的张量。
    env_goal_pos = env_goal_pos[..., :3].contiguous()  # [B,1,3]

    # Split observations into position and orientation components.
    # 六维观测解释为前三维位置、后三维姿态；这里不额外变换姿态表示。
    if D == 6:
        pos = obs[:, :, :3]
        ori = obs[:, :, 3:6]
        next_pos = next_obs[:, :, :3]
        next_ori = next_obs[:, :, 3:6]
    # 三维观测仅包含位置，当前和下一姿态均设为 None。
    elif D == 3:
        pos = obs
        ori = None
        next_pos = next_obs
        next_ori = None
    else:
        raise ValueError(f"Expected obs dim = 3 or 6, got {D}")

    # -------- choose t --------
    # t is the current control step.
    # allow t from 0 to T-1. obs left side pads, action right side masks.
    # 当前控制时刻允许覆盖整条轨迹；窗口越界部分后续通过边界复制和掩码处理。
    t_min = 0
    t_max = T - 1

    # If t is not provided, sample one control step according to sample_mode.
    # 未指定 t 时，从配置选择随机采样方式。
    if t is None:
        sample_mode = getattr(cfg.training, "sample_mode", "uniform")

        # Uniformly sample t across the complete trajectory.
        # 均匀模式从闭区间 [0, T - 1] 中采样一个整数，供整批共用。
        if sample_mode == "uniform":
            t_i = int(np.random.randint(t_min, t_max + 1))

        # 偏置模式按 50% 过渡区域、30% 拉动阶段、20% 接近阶段选择采样区间。
        # 这些区间可能重叠，因此不代表三个实际阶段最终出现的精确比例。
        elif sample_mode == "biased_50_30_20":
            # Helper for sampling from a closed integer interval.
            # 从包含上下边界的整数区间采样；空区间返回 None，由调用处回退处理。
            def _randint_closed(lo, hi):
                lo = int(lo)
                hi = int(hi)
                if lo > hi:
                    return None
                return int(np.random.randint(lo, hi + 1))

            # Reach-stage interval.
            # 接近区间截至 reach_end - 1，并受轨迹长度限制。
            reach_lo = t_min
            reach_hi = min(t_max, reach_end - 1)

            # Pull-stage interval.
            # 拉动区间从 align_end 开始，一直延伸到轨迹末尾。
            pull_lo = max(t_min, align_end)
            pull_hi = t_max

            # transition around reach -> align
            # Include a wider temporal region around the reach/align transition.
            # 过渡区间覆盖接近结束前 To 步及之后 Ta 步附近，以增加阶段切换样本。
            trans01_lo = max(t_min, reach_end - To)
            trans01_hi = min(t_max, reach_end + Ta - 1)

            # Sampling distribution:
            # 50% transition region, 30% pull stage, 20% reach stage.
            # 先随机选择采样区间，再在对应区间内均匀采样；空区间回退到整条轨迹。
            u = np.random.rand()
            if u < 0.50:
                t_candidate = _randint_closed(trans01_lo, trans01_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)
            elif u < 0.80:
                t_candidate = _randint_closed(pull_lo, pull_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)
            else:
                t_candidate = _randint_closed(reach_lo, reach_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)

            t_i = int(t_candidate)
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

    # Convert different input types into one integer timestep.
    # 显式传入整数时直接使用；其他支持的容器类型必须只含一个元素。
    elif isinstance(t, int):
        t_i = int(t)
    elif isinstance(t, torch.Tensor):
        # 张量先脱离计算图并移到 CPU，再转换成 Python 整数列表。
        vals = [int(x) for x in t.detach().cpu().view(-1).tolist()]
        # 拒绝按样本分别传入多个时刻，当前实现仅允许单个共享时刻。
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]
    elif isinstance(t, np.ndarray):
        vals = [int(x) for x in t.reshape(-1).tolist()]
        # 拒绝按样本分别传入多个时刻，当前实现仅允许单个共享时刻。
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]
    else:
        vals = [int(x) for x in t]
        # 拒绝按样本分别传入多个时刻，当前实现仅允许单个共享时刻。
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]

    # Ensure the selected timestep stays inside the trajectory.
    # 将指定或采样的时刻限制在合法轨迹范围内；要求 T 至少为 1。
    t_i = int(np.clip(t_i, t_min, t_max))

    # -------- history obs index + future action index --------
    # Observation history: [t-To+1, ..., t].
    # 构造包含当前时刻的历史索引 [t - To + 1, ..., t]，起始位置可能小于 0。
    obs_raw_idx = torch.arange(t_i - To + 1, t_i + 1, device=device)   # [To]

    # Future action horizon: [t, ..., t+Ta-1].
    # 构造未来动作索引 [t, ..., t + Ta - 1]，末尾可能超过轨迹长度。
    act_raw_idx = torch.arange(t_i, t_i + Ta, device=device)           # [Ta]

    # Masks indicate which raw indices correspond to valid trajectory steps.
    # 先依据未截断索引生成有效性标记：1 为有效，0 为填充。
    obs_mask_1d = ((obs_raw_idx >= 0) & (obs_raw_idx < T)).float()
    action_mask_1d = ((act_raw_idx >= 0) & (act_raw_idx < T)).float()

    # Clamp invalid indices so tensor indexing remains valid.
    # The masks above are later used to identify padded elements.
    # 再将索引截断到 [0, T - 1]：历史左侧复制首帧，动作右侧复制最后一步。
    # 复制出来的数据不是有效训练位置，需要下游按掩码忽略。
    obs_idx = obs_raw_idx.clamp(0, T - 1)
    act_idx = act_raw_idx.clamp(0, T - 1)

    # Build one-step-ahead indices.
    # 以已截断索引加 1，生成下一时刻阶段与目标索引；不是从原始负索引加 1 后截断。
    next_obs_idx = (obs_idx + 1).clamp(0, T - 1)
    # 生成下一动作索引；当前函数后续没有使用这个变量。
    next_act_idx = (act_idx + 1).clamp(0, T - 1)

    # Expand validity masks across the batch dimension.
    # 将同一组有效性掩码广播到 B 条轨迹，保持所有样本的窗口时刻一致。
    obs_mask = obs_mask_1d.view(1, To).expand(B, -1)          # [B,To]
    action_mask = action_mask_1d.view(1, Ta).expand(B, -1)    # [B,Ta]

    # -------- gather obs windows --------
    # Extract current and next observation windows.
    # 按 obs_idx 提取当前位置和可选姿态，输出时间长度为 To。
    pos_window = pos[:, obs_idx, :]                                    # [B,To,3]
    ori_window = ori[:, obs_idx, :] if ori is not None else None
    # next_obs 已经存储每条转移的下一状态，因此用同一 obs_idx 读取，不再额外平移一次。
    next_pos_window = next_pos[:, obs_idx, :]                          # [B,To,3]
    next_ori_window = next_ori[:, obs_idx, :] if next_ori is not None else None

    # Optional joint-state windows.
    # 关节窗口沿用观测索引，数据缺失时保持 None。
    joint_pos_window = joint_pos[:, obs_idx, :] if joint_pos is not None else None
    next_joint_window = next_joint_pos[:, obs_idx, :] if next_joint_pos is not None else None

    # Optional wrist/gripper RGB history window.
    # 视觉输入预期为 [B, T, C, H, W]，沿时间维取得历史窗口；此处不归一化像素。
    rgb_gripper_obs_window = (
        rgb_gripper_obs[:, obs_idx, :, :, :] if rgb_gripper_obs is not None else None
    )

    # -------- gather future action windows --------
    # Extract future action trajectory from the selected timestep.
    # 动作窗口为 [B, Ta, Da]；越界末尾的数据会复制最后一个动作。
    act_window = action[:, act_idx, :]                                 # [B,Ta,Da]
    # next_action 已表示下一动作，故仍按 act_idx 读取；没有该字段则返回 None。
    next_action_window = next_action[:, act_idx, :] if next_action is not None else None

    # Corresponding reward and termination windows.
    # 奖励和 done 与未来动作按相同索引对齐，预期有末尾特征维，例如 [B, T, 1]。
    reward_window = reward[:, act_idx, :] if reward is not None else None
    done_window = done[:, act_idx, :] if done is not None else None

    # -------- stage/progress for obs side --------
    # Encode task stage for each observation timestep:
    # 0 = reach, 1 = align, 2 = pull.
    # 按每个历史索引生成阶段标签，0 为接近、1 为对齐、2 为拉动。
    stage_ids_obs = torch.zeros(To, device=device, dtype=torch.long)
    stage_ids_obs[(obs_idx >= reach_end) & (obs_idx < align_end)] = 1
    stage_ids_obs[obs_idx >= align_end] = 2
    # 阶段扩展成 [B, To, 1] 浮点张量，后续可作为条件或再转成嵌入索引。
    stage_seq = stage_ids_obs.view(1, To, 1).expand(B, -1, -1).float()

    # Stage sequence for the next observation state.
    # 下一观测阶段使用 next_obs_idx，阶段边界处可能与当前阶段不同。
    next_stage_ids_obs = torch.zeros(To, device=device, dtype=torch.long)
    next_stage_ids_obs[(next_obs_idx >= reach_end) & (next_obs_idx < align_end)] = 1
    next_stage_ids_obs[next_obs_idx >= align_end] = 2
    next_stage_seq = next_stage_ids_obs.view(1, To, 1).expand(B, -1, -1).float()

    # Normalized temporal progress in [0, 1].
    # 将索引除以 T - 1 得到时间进度；T = 1 时分母使用 1，避免除零。
    # 当前 progress 字段在返回字典中被注释，因此这些值虽计算但未输出。
    progress_seq = (obs_idx.float() / max(T - 1, 1)).view(1, To, 1).expand(B, -1, -1)
    next_progress_seq = (next_obs_idx.float() / max(T - 1, 1)).view(1, To, 1).expand(B, -1, -1)

    # -------- stage/progress for action side --------
    # Stage labels for each future action step.
    # 未来动作阶段按 act_idx 单独计算，其长度 Ta 不一定等于观测长度 To。
    action_stage_ids = torch.zeros(Ta, device=device, dtype=torch.long)
    action_stage_ids[(act_idx >= reach_end) & (act_idx < align_end)] = 1
    action_stage_ids[act_idx >= align_end] = 2
    # 生成 [B, Ta, 1] 的动作阶段序列，供动作解码器按未来时刻使用。
    action_stage_seq = action_stage_ids.view(1, Ta, 1).expand(B, -1, -1).float()

    # -------- stage-aware goal positions on obs side --------
    # Stage 0 goal: next demonstrated position at the corresponding timestep.
    # 局部目标体系：接近阶段用对应转移的下一演示位置作为目标。
    reach_goal_seq = next_pos[:, obs_idx, :]                           # [B,To,3]

    # Stage 1 goal: fixed position near the end of the alignment phase.
    # 对齐阶段固定参考 align_end - 1 对应的 next_pos，并限制锚点在有效范围内。
    align_anchor_idx = min(max(align_end - 1, 0), T - 1)
    align_goal = next_pos[:, align_anchor_idx:align_anchor_idx + 1, :]
    # 将同一个对齐锚点复制到全部 To 个历史时刻。
    align_goal_seq = align_goal.expand(-1, To, -1)

    # Stage 2 goal: a future demonstrated position ahead of the current timestep.
    # 拉动阶段使用比历史索引更靠后的参考位置，越过轨迹末尾时固定使用最后位置。
    pull_goal_idx = torch.clamp(obs_idx + future_offset + pull_extra_offset, 0, T - 1)
    pull_goal_seq = next_pos[:, pull_goal_idx, :]

    # Select the appropriate goal according to the task stage.
    # 把阶段标签沿 XYZ 三个坐标分量广播，供 torch.where 逐元素选择目标。
    stage_ids_expand = stage_ids_obs.view(1, To, 1).expand(B, -1, 3)
    # 按历史时刻的阶段，在接近、对齐、拉动局部目标之间选择，得到 [B, To, 3]。
    stage_goal_pos_seq = torch.where(
        # 把阶段标签沿 XYZ 三个坐标分量广播，供 torch.where 逐元素选择目标。
        stage_ids_expand == 0,
        reach_goal_seq,
        torch.where(stage_ids_expand == 1, align_goal_seq, pull_goal_seq),
    )

    # next stage-aware goal positions
    # Construct equivalent stage-dependent goals for next observations.
    # 下一状态的局部目标按 next_obs_idx 查 next_pos；与 next_pos_window 的读取索引不同。
    next_reach_goal_seq = next_pos[:, next_obs_idx, :]
    next_align_goal_seq = align_goal.expand(-1, To, -1)

    # 为下一状态构造更靠后的拉动参考目标。
    next_pull_goal_idx = torch.clamp(
        next_obs_idx + future_offset + pull_extra_offset, 0, T - 1
    )
    next_pull_goal_seq = next_pos[:, next_pull_goal_idx, :]

    next_stage_ids_expand = next_stage_ids_obs.view(1, To, 1).expand(B, -1, 3)
    # 按下一状态阶段选择其对应的局部目标。
    next_stage_goal_pos_seq = torch.where(
        next_stage_ids_expand == 0,
        next_reach_goal_seq,
        torch.where(next_stage_ids_expand == 1, next_align_goal_seq, next_pull_goal_seq),
    )

    # -------- errors --------
    # Cartesian error between the current position and stage-dependent goal.
    # 目标减实际位置，得到指向局部目标的三维误差向量。
    goal_error = stage_goal_pos_seq - pos_window
    next_goal_error = next_stage_goal_pos_seq - next_pos_window

    # Squared distance, Euclidean distance, and normalized goal direction.
    # 沿 XYZ 维求平方和并保留单维，形状为 [B, To, 1]。
    goal_dist2 = (goal_error ** 2).sum(dim=-1, keepdim=True)
    # 距离加入 1e-6 稳定平方根；零误差时该距离为 0.001，而不是严格为零。
    goal_dist = torch.sqrt(goal_dist2 + 1e-6)
    # 用误差除以带小常数的距离，得到稳定的近似单位方向。
    goal_dir = goal_error / (goal_dist + 1e-6)

    # 用相同方式计算下一状态相对其局部目标的平方距离、距离和方向。
    next_goal_dist2 = (next_goal_error ** 2).sum(dim=-1, keepdim=True)
    next_goal_dist = torch.sqrt(next_goal_dist2 + 1e-6)
    next_goal_dir = next_goal_error / (next_goal_dist + 1e-6)

    # -------- global subgoals --------
    # Define the start of the pull phase and the final trajectory timestep.
    # 将拉动方向估计起点限制到轨迹内，最终时刻固定为 T - 1。
    pull_start_t = min(max(pull_start_t_cfg, 0), T - 1)
    final_t = T - 1

    # stage 0/1: env goal / handle goal
    # Use the environment-provided goal for reach/alignment.
    # 全局子目标体系与前面的局部目标不同：接近/对齐阶段使用外部环境关键点。
    subgoal_reach = env_goal_pos  # [B,1,3]

    # stage 2: demo pull goal
    # Use the final demonstrated position as the pull subgoal.
    # 全局拉动子目标取 next_pos 序列最后一个位置。
    subgoal_pull = next_pos[:, final_t:final_t + 1, :]  # [B,1,3]

    # Expand fixed subgoals across the observation horizon.
    # 将每条轨迹的固定子目标广播到所有历史时刻。
    subgoal_reach_seq = subgoal_reach.expand(-1, To, -1)  # [B,To,3]
    subgoal_pull_seq = subgoal_pull.expand(-1, To, -1)    # [B,To,3]

    # next side
    # The global subgoal itself is unchanged for the next-state representation.
    # 下一状态复用同一组全局子目标位置，只改变相对位置误差及阶段选择。
    next_subgoal_reach_seq = subgoal_reach_seq
    next_subgoal_pull_seq = subgoal_pull_seq

    # Compute errors to both reach and pull subgoals.
    # 分别计算相对于环境接近目标和演示最终拉动目标的位置误差。
    subgoal_reach_error = subgoal_reach_seq - pos_window
    subgoal_pull_error = subgoal_pull_seq - pos_window

    # 对下一状态位置计算同样的两类子目标误差。
    next_subgoal_reach_error = next_subgoal_reach_seq - next_pos_window
    next_subgoal_pull_error = next_subgoal_pull_seq - next_pos_window

    # Select the currently active subgoal according to the task stage.
    # Reach/alignment use env_goal_pos; pull uses the final demonstration position.
    # 阶段 0、1 选择环境目标，阶段 2 选择演示最终位置，得到当前有效全局子目标。
    active_subgoal_pos = torch.where(
        stage_seq.expand(-1, -1, 3) < 2.0,
        subgoal_reach_seq,
        subgoal_pull_seq,
    )
    # 按相同阶段规则选择对应误差；单维阶段条件会广播到 XYZ 三维。
    active_subgoal_error = torch.where(
        stage_seq < 2.0,
        subgoal_reach_error,
        subgoal_pull_error,
    )

    # Active subgoal representation for the next state.
    # 根据下一状态阶段确定其有效全局子目标和误差。
    # 当前这些 active_subgoal 字段在 cond 和顶层输出中都被注释，计算结果未返回。
    next_active_subgoal_pos = torch.where(
        next_stage_seq.expand(-1, -1, 3) < 2.0,
        next_subgoal_reach_seq,
        next_subgoal_pull_seq,
    )
    next_active_subgoal_error = torch.where(
        next_stage_seq < 2.0,
        next_subgoal_reach_error,
        next_subgoal_pull_error,
    )

    # -------- pull direction --------
    # Estimate the demonstrated pulling direction using the displacement
    # from pull_start_t to the final demonstration position.
    # 用 next_pos 中拉动段终点减起点，按批次估计演示拉动方向。
    raw_pull_vec = next_pos[:, final_t, :] - next_pos[:, pull_start_t, :]
    # 计算每条轨迹方向向量的欧氏范数，形状为 [B, 1]。
    raw_pull_norm = torch.norm(raw_pull_vec, dim=-1, keepdim=True)

    # If possible, also estimate the average local displacement
    # along the demonstrated pull segment.
    # 拉动段至少有两个位置时，另计算相邻位移的均值作为备选方向。
    if pull_start_t < final_t:
        pull_seg = next_pos[:, pull_start_t:final_t + 1, :]
        local_deltas = pull_seg[:, 1:, :] - pull_seg[:, :-1, :]
        # 沿时间维平均相邻位移，得到每条轨迹的平均位移向量。
        mean_local_delta = local_deltas.mean(dim=1)
        mean_local_norm = torch.norm(mean_local_delta, dim=-1, keepdim=True)
    else:
        # 只有一个位置时没有可用位移，用零向量表示。
        mean_local_delta = torch.zeros_like(raw_pull_vec)
        mean_local_norm = torch.zeros_like(raw_pull_norm)

    # Fall back to the mean local motion if the global displacement is nearly zero.
    # 端点位移过小时，按样本切换到平均局部位移；其余样本保持端点位移。
    use_local = (raw_pull_norm < 1e-6).float()
    pull_vec = (1.0 - use_local) * raw_pull_vec + use_local * mean_local_delta
    pull_norm = (1.0 - use_local) * raw_pull_norm + use_local * mean_local_norm

    # Final fallback direction when no valid motion direction can be estimated.
    # 若备选位移也接近零，使用固定的负 x 方向作为最终回退。
    default_pull_dir = torch.tensor(
        [-1.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).view(1, 3).expand(B, -1)

    # Normalize the demonstrated pulling direction.
    # 将选中的位移向量归一化，随后按样本替换无法估计的方向。
    demo_pull_dir = pull_vec / (pull_norm + 1e-8)
    use_default = (pull_norm < 1e-6).float()
    demo_pull_dir = (1.0 - use_default) * demo_pull_dir + use_default * default_pull_dir

    # Broadcast the pull direction over the observation horizon.
    # 将每条演示的拉动方向广播到 To 个时刻，并 clone 成独立张量。
    base_pull_dir_seq = demo_pull_dir.unsqueeze(1).expand(-1, To, -1).clone()

    # Pull direction is only active during stage 2.
    # 只有阶段 2 的方向保留，其余阶段方向设为零。
    pull_mask = (stage_seq == 2.0).float()
    next_pull_mask = (next_stage_seq == 2.0).float()

    # 生成当前和下一状态的阶段门控拉动方向；当前这些方向字段没有写入返回条件。
    pull_dir_seq = base_pull_dir_seq * pull_mask
    next_pull_dir_seq = base_pull_dir_seq * next_pull_mask

    # Build the conditioning dictionary passed to the policy / critic.
    # 组装当前/下一状态条件与掩码；本函数主要切片并构建特征，不执行模型输入归一化。
    # 保留所有原有注释状态：关节、进度、有效子目标和拉动方向等字段目前没有加入 cond。
    cond = {
        "pos": pos_window,
        "ori": ori_window,
        "next_pos": next_pos_window,
        "next_ori": next_ori_window,

        # 之前去掉 joint_pos，这里继续默认不放进 cond。
        # 如果之后要加回来，取消注释即可。
        #"joint_pos": joint_pos_window,
        #"next_joint_pos": next_joint_window,

        # 此字段的时间长度为 Ta，区别于大多数观测条件字段的 To。
        "next_action": next_action_window,
        "rgb_gripper": rgb_gripper_obs_window,

        # Stage information for current and next observations.
        "stage": stage_seq,
        "next_stage": next_stage_seq,
        #"progress": progress_seq,
        #"next_progress": next_progress_seq,

        # Stage-dependent local target positions.
        # 保存按阶段变化的局部目标，不等同于下面固定的全局子目标。
        "stage_goal_pos": stage_goal_pos_seq,
        "next_stage_goal_pos": next_stage_goal_pos_seq,

        #"goal_error": goal_error,
        #"next_goal_error": next_goal_error,

        # Goal-distance features used as conditioning information.
        "goal_dist2": goal_dist2,
        "goal_dist": goal_dist,
        "goal_dir": goal_dir,

        "next_goal_dist2": next_goal_dist2,
        "next_goal_dist": next_goal_dist,
        "next_goal_dir": next_goal_dir,

        # Global reach and pull subgoals.
        "subgoal_reach_pos": subgoal_reach_seq,
        "subgoal_pull_pos": subgoal_pull_seq,

        "next_subgoal_reach_pos": next_subgoal_reach_seq,
        "next_subgoal_pull_pos": next_subgoal_pull_seq,

        #"active_subgoal_pos": active_subgoal_pos,
        #"active_subgoal_error": active_subgoal_error,

        #"next_active_subgoal_pos": next_active_subgoal_pos,
        #"next_active_subgoal_error": next_active_subgoal_error,

        #"pull_dir": pull_dir_seq,
        #"next_pull_dir": next_pull_dir_seq,

        # Transformer 用这个，True = ignore padding
        # Observation masks for valid and padded timesteps.
        # 有效掩码中 True 表示真实观测；padding 掩码中的 True 则表示需要忽略。
        "obs_valid_mask": obs_mask.bool(),
        "obs_padding_mask": obs_mask == 0,

        # Action masks for valid, padded, and loss-bearing timesteps.
        # 动作有效、填充和损失掩码使用相同时间范围，但布尔含义或数值类型不同。
        "action_valid_mask": action_mask.bool(),
        "action_padding_mask": action_mask == 0,
        "action_loss_mask": action_mask.float()

    }

    # Return the future action window together with all conditioning metadata.
    return {
        # "pos_window": pos_window,
        # "ori_window": ori_window,
        # "next_pos_window": next_pos_window,
        # "next_ori_window": next_ori_window,

        # "joint_pos_window": joint_pos_window,
        # "next_joint_window": next_joint_window,

         # 返回未来动作及对应转移窗口；奖励、done、next_action 缺失时对应值为 None。
         "act_window": act_window,
         "next_action_window": next_action_window,
         "reward_window": reward_window,
         "done_window": done_window,

        # "obs_mask": obs_mask,
        # "action_mask": action_mask,

        # "stage_goal_pos_seq": stage_goal_pos_seq,
        # "next_stage_goal_pos_seq": next_stage_goal_pos_seq,

        # "goal_error": goal_error,
        # "next_goal_error": next_goal_error,
        # "goal_dist2": goal_dist2,
        # "goal_dist": goal_dist,
        # "goal_dir": goal_dir,

        # "subgoal_reach_pos_seq": subgoal_reach_seq,
        # "subgoal_pull_pos_seq": subgoal_pull_seq,
        # "subgoal_reach_error": subgoal_reach_error,
        # "subgoal_pull_error": subgoal_pull_error,
        # "active_subgoal_pos": active_subgoal_pos,
        # "active_subgoal_error": active_subgoal_error,

        # "next_subgoal_reach_pos_seq": next_subgoal_reach_seq,
        # "next_subgoal_pull_pos_seq": next_subgoal_pull_seq,
        # "next_active_subgoal_pos": next_active_subgoal_pos,
        # "next_active_subgoal_error": next_active_subgoal_error,

        # "progress_seq": progress_seq,
        # "next_progress_seq": next_progress_seq,
        # "stage_seq": stage_seq,
        # "next_stage_seq": next_stage_seq,

        # "action_stage_seq": action_stage_seq,

        # "pull_dir_seq": pull_dir_seq,
        # "next_pull_dir_seq": next_pull_dir_seq,
        # "demo_pull_dir": demo_pull_dir,

        # cond_t stores the complete conditioning package for this sampled timestep.
        # cond_t 是完整条件包：特征字典、索引、兼容掩码、共享时刻及阶段信息。
        "cond_t": {
            "cond": cond,
            # 输出的是已截断索引，判断哪些位置是填充还需同时查看掩码。
            "obs_idx": obs_idx.unsqueeze(0).expand(B, -1),      # [B,To]
            "act_idx": act_idx.unsqueeze(0).expand(B, -1),      # [B,Ta]
            "obs_mask": obs_mask,                              # [B,To]
            "action_mask": action_mask,                        # [B,Ta]
            # 兼容 Transformer 的旧掩码名称，True 表示需要屏蔽的观测位置。
            "src_key_padding_mask": obs_mask == 0,              # [B,To]
            # 整批的 t 都是同一个 t_i；这是轨迹时刻，不是扩散去噪时间步。
            "t": torch.full((B,), t_i, device=device, dtype=torch.long),
            "stage": stage_seq,
            # 把未来动作阶段独立传出，以便与 Ta 个动作 token 对齐。
            "action_stage": action_stage_seq,
            "batch_size": B,
        }
    }
