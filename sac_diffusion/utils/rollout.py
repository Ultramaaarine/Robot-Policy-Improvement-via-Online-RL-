# cond: total 20 dim: pos:3, joints: 7, subgoal_active_error: 3, pull_dir: 3, progress: 1, goal: 3 (read from env, not from offline dataset)
# is a torch.tensor 
# obs,pos seq has only one batch size, not like offline dataset
# copy multiple time when rollout, generate multiple action sequences for selection 
# 环境交互与评估工具：构造历史条件、生成候选动作、执行回合、保存轨迹及统计。
# 文件顶部的 20 维说明是原有描述；当前实际条件字段应以 build_rollout_cond 和 assemble_conditions 为准。
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import json
from sac_diffusion.models.normalizer import Normalizer
from sac_diffusion.utils.condition_builder import assemble_conditions
from sac_diffusion.utils.action_selector import select_action
from sac_diffusion.utils.common_funcs import extract_obs
from sac_diffusion.utils.save_array_stats import save_array_stats
from sac_diffusion.utils.common_funcs import compute_shaped_reward

# 从一条离线演示提取位置轨迹、末端子目标及拉动方向，供在线 rollout 使用。
def prepare_rollout_targets(training_dataset, cfg): # reach goal is acquired from env move to build_cond, here only offline data
    # 读取对齐阶段结束时刻，默认第 36 步；阶段边界按环境步数划分。
    align_end = int(getattr(cfg.training, "align_end_t", 36))
    # 读取演示拉动段起点，未配置时使用对齐结束时刻。
    pull_start_t = int(getattr(cfg.training, "pull_start_t", align_end))

    # 当前固定选取第 0 条演示，后面的随机选择代码仅为注释。
    demo_idx = 0 # np.random.random_integers(0,7)
    sample = training_dataset[demo_idx] # single dict

    # 读取演示观测序列，预期布局为 [T, D]，其中前三列是末端位置。
    traj = sample["obs"]
    # 若为张量则先脱离计算图并移到 CPU，统一转换成 float32 NumPy 数组。
    if isinstance(traj, torch.Tensor):
        traj_np = traj.detach().cpu().numpy().astype(np.float32)
    else:
        traj_np = np.asarray(traj, dtype=np.float32)

    # 检查演示是否为二维时序数据。
    if traj_np.ndim != 2:
        raise ValueError(f"Expected traj_np shape [T,D], got {traj_np.shape}")

    # 提取位置轨迹 [T, 3]，并记录轨迹长度。
    traj_pos_np = traj_np[:, :3].astype(np.float32)
    traj_T = traj_pos_np.shape[0]

    # 将拉动起点限制在轨迹范围内；此处假定演示至少包含一个时刻。
    pull_start_t = min(max(pull_start_t, 0), traj_T - 1)
    final_t = traj_T - 1

    
    # 使用演示最后一个位置作为拉动阶段的末端目标。
    subgoal_pull_np = traj_pos_np[final_t].copy()

    # 用拉动段终点减起点，估计拉动方向向量。
    raw_pull_vec = traj_pos_np[final_t] - traj_pos_np[pull_start_t]
    raw_pull_norm = float(np.linalg.norm(raw_pull_vec))

    # 若方向向量过小，第一次尝试用局部位移均值；仍过小时使用默认方向。
    if raw_pull_norm < 1e-6:
        pull_seg = traj_pos_np[pull_start_t:final_t + 1]
        if pull_seg.shape[0] >= 2:
            # 计算拉动段相邻位置之差，再取平均位移。
            local_deltas = pull_seg[1:] - pull_seg[:-1]
            raw_pull_vec = local_deltas.mean(axis=0).astype(np.float32)
            raw_pull_norm = float(np.linalg.norm(raw_pull_vec))

    # 若方向向量过小，第一次尝试用局部位移均值；仍过小时使用默认方向。
    if raw_pull_norm < 1e-6:
        # 方向无法可靠估计时，默认采用世界坐标负 x 方向。
        demo_pull_dir_np = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
        # 将方向向量归一化，小常数用于稳定除法。
        demo_pull_dir_np = (raw_pull_vec / (raw_pull_norm + 1e-8)).astype(np.float32)

    print(
        f"[rollout] estimated demo_pull_dir = {demo_pull_dir_np}, "
        f"pull_start_t={pull_start_t}, final_t={final_t}",
        flush=True,
    )

    return {
        "traj_pos_np": traj_pos_np,
        "traj_T": traj_T,
        "pull_start_t": pull_start_t,
        "final_t": final_t,
        "subgoal_pull_np": subgoal_pull_np,
        "demo_pull_dir_np": demo_pull_dir_np,
    }

# 递归处理字典中的张量，将张量第 0 维重复 K 次；其他对象保持原样。
# 本调用场景假定原始批大小为 1，且张量具有批次维。
def repeat_tensor_tree(x, K: int):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        # 只重复批次维，其他维度保持不变；repeat 会生成重复后的张量。
        reps = [K] + [1] * (x.dim() - 1)
        return x.repeat(*reps)

    # 遍历嵌套字典，递归复制其中的张量。
    if isinstance(x, dict):
        return {k: repeat_tensor_tree(v, K) for k, v in x.items()}

    return x


# 把单样本条件复制成 K 个相同条件，用于生成多个动作候选。
def repeat_cond_t_batch(cond_t: dict, K: int):
    cond_t_k = {}
    for key, value in cond_t.items():
        # 批大小是元数据，直接设置为 K，不按张量处理。
        if key == "batch_size":
            cond_t_k[key] = K
        else:
            cond_t_k[key] = repeat_tensor_tree(value, K)

    cond_t_k["batch_size"] = K
    return cond_t_k

# 根据当前环境步数、观测历史和演示目标，构建策略输入及诊断信息。
# To 为历史观测长度，Ta 为未来动作长度；两者可以不同。
# eef_pos 参数在本函数体中未直接使用；传入的 progress_idx 会按 t 重新计算。
def build_rollout_cond(
    cfg,
    device,
    replaybuffer,
    t,
    eef_pos,
    obs_hist,
    joint_hist,
    gripper_hist,
    demo_info,
    env_goal,
    progress_idx,
):
    # 读取模型需要的历史观测窗口长度。
    To = int(cfg.training.obs_hor_len)
    # 读取需要预测的未来动作序列长度。
    Ta = int(cfg.training.act_hor_len)

    # 目标窗口相对当前步向未来偏移，默认偏移一个动作预测长度。
    future_offset = int(getattr(cfg.training, "future_offset", Ta))
    # 初始目标窗口长度默认等于观测窗口长度，后续还会统一到 To。
    goal_window_len = int(getattr(cfg.training, "goal_window_len", To))

    # 读取接近阶段结束步数，默认 32；之后进入对齐阶段。
    reach_end = int(getattr(cfg.training, "reach_end_t", 32))
    # 读取对齐阶段结束时刻，默认第 36 步；阶段边界按环境步数划分。
    align_end = int(getattr(cfg.training, "align_end_t", 36))

    traj_pos_np = demo_info["traj_pos_np"]
    traj_T = int(demo_info["traj_T"])

    # 优先使用经验回放要求的序列长度，缺少该属性时使用演示长度。
    max_rollout_len = int(getattr(replaybuffer, "seq_len", traj_T))

    # 接近/对齐目标取自环境关键点；拉动目标和方向取自离线演示。
    subgoal_reach_np = np.asarray(env_goal, dtype=np.float32).reshape(-1)[:3]
    # 使用演示最后一个位置作为拉动阶段的末端目标。
    subgoal_pull_np = np.asarray(demo_info["subgoal_pull_np"], dtype=np.float32).reshape(-1)[:3]
    demo_pull_dir_np = np.asarray(demo_info["demo_pull_dir_np"], dtype=np.float32).reshape(-1)[:3]

    # ------------------------------------------------------------
    # 1. rollout history window: [t-To+1, ..., t]
    # deque 里已经是 left-padded 后的长度 To
    # ------------------------------------------------------------
    # 把历史队列堆叠为 [To, 3] 的末端位置序列；关节历史同步堆叠。
    rollout_obs_window = np.stack(list(obs_hist), axis=0).astype(np.float32)
    rollout_joint_window = np.stack(list(joint_hist), axis=0).astype(np.float32)

    # 检查历史队列长度，确保符合模型输入要求。
    if rollout_obs_window.shape[0] != To:
        raise ValueError(
            f"rollout_obs_window length {rollout_obs_window.shape[0]} != To {To}"
        )

    # 关节历史也必须恰好包含 To 个时刻。
    if rollout_joint_window.shape[0] != To:
        raise ValueError(
            f"rollout_joint_window length {rollout_joint_window.shape[0]} != To {To}"
        )

    rollout_gripper_window = None
    # 有夹爪图像历史时，构建并检查对应视觉窗口。
    if gripper_hist is not None:
        rollout_gripper_window = np.stack(list(gripper_hist), axis=0).astype(np.float32)
        if rollout_gripper_window.shape[0] != To:
            raise ValueError(
                f"rollout_gripper_window length {rollout_gripper_window.shape[0]} != To {To}"
            )

    # ------------------------------------------------------------
    # 2. progress index
    # Do NOT use nearest-demo progress anymore.
    # Keep a step-based progress_idx only for logging / compatibility.
    # It should NOT be used as model condition.
    # ------------------------------------------------------------
    # 按实际 rollout 步数计算兼容性进度索引，不再用最近演示点匹配进度。
    progress_idx = int(np.clip(t, 0, traj_T - 1))

    # ------------------------------------------------------------
    # 3. padding masks
    #
    # obs_padding_mask:
    #   [1, To], True = padding / ignore
    #
    # action_padding_mask:
    #   [1, Ta], True = padding / ignore
    #
    # action_loss_mask:
    #   [1, Ta], 1 = valid
    #   rollout 里一般不用 loss，但保留统一接口
    # ------------------------------------------------------------
    # 历史观测索引包括当前时刻；未来动作索引从当前时刻开始。
    obs_raw_idx = np.arange(t - To + 1, t + 1)
    act_raw_idx = np.arange(t, t + Ta)

    # 负的历史索引属于起始填充；超过回合长度的未来索引也属于填充。
    obs_valid_np = (obs_raw_idx >= 0) & (obs_raw_idx < max_rollout_len)
    action_valid_np = (act_raw_idx >= 0) & (act_raw_idx < max_rollout_len)

    # padding 掩码与 valid 掩码取反：True 分别表示忽略位置和有效位置。
    obs_padding_np = ~obs_valid_np
    action_padding_np = ~action_valid_np

    # 为单回合添加批次维，将掩码转换为目标设备上的布尔张量。
    obs_valid_mask = torch.from_numpy(obs_valid_np[None, :]).to(
        device=device,
        dtype=torch.bool,
    )  # [1, To]

    obs_padding_mask = torch.from_numpy(obs_padding_np[None, :]).to(
        device=device,
        dtype=torch.bool,
    )  # [1, To]

    action_valid_mask = torch.from_numpy(action_valid_np[None, :]).to(
        device=device,
        dtype=torch.bool,
    )  # [1, Ta]

    action_padding_mask = torch.from_numpy(action_padding_np[None, :]).to(
        device=device,
        dtype=torch.bool,
    )  # [1, Ta]

    # 将动作有效性转成浮点数，供统一的损失接口使用：有效为 1，填充为 0。
    action_loss_mask = action_valid_mask.float()  # [1, Ta]

    # ------------------------------------------------------------
    # 4. rollout-step based stage, independent of progress_idx
    #
    # obs-side stage uses rollout obs step:
    #   [t-To+1, ..., t]
    #
    # action-side stage uses rollout future action step:
    #   [t, ..., t+Ta-1]
    # ------------------------------------------------------------
    obs_step_raw_idx = np.arange(t - To + 1, t + 1)
    # 阶段查询使用截断后的步数，保证索引落在回合范围内。
    obs_step_idx = np.clip(obs_step_raw_idx, 0, max_rollout_len - 1)

    action_step_raw_idx = np.arange(t, t + Ta)
    action_step_idx = np.clip(action_step_raw_idx, 0, max_rollout_len - 1)

    # logging only, not used as model condition
    # 将历史步数归一化为进度值，仅返回用于日志，不加入模型条件特征。
    progress_seq_np = (
        obs_step_idx.astype(np.float32) / max(max_rollout_len - 1, 1)
    ).reshape(To, 1)

    # obs-side stage: [To, 1]
    # 观测阶段：0 表示接近，1 表示对齐，2 表示拉动，完全由步数阈值决定。
    stage_seq_np = np.zeros((To, 1), dtype=np.float32)
    stage_seq_np[(obs_step_idx >= reach_end) & (obs_step_idx < align_end)] = 1.0
    stage_seq_np[obs_step_idx >= align_end] = 2.0

    # action-side stage: [Ta, 1]
    # 未来动作按各自执行时刻分配阶段；同一预测序列可以跨越阶段边界。
    action_stage_seq_np = np.zeros((Ta, 1), dtype=np.float32)
    action_stage_seq_np[
        (action_step_idx >= reach_end) & (action_step_idx < align_end)
    ] = 1.0
    action_stage_seq_np[action_step_idx >= align_end] = 2.0

    # 未来动作阶段转换为 [1, Ta, 1]，与动作解码器时刻对齐。
    action_stage_seq = torch.from_numpy(action_stage_seq_np).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )  # [1, Ta, 1]

    # ------------------------------------------------------------
    # 5. goal window
    # use rollout step, not progress_idx
    # ------------------------------------------------------------
    # 当前环境步数用于定位演示目标窗口，超出演示长度时固定在末端。
    goal_ref_idx = int(np.clip(t, 0, traj_T - 1))

    # 向未来偏移获得目标窗口起点，并将范围限制在演示轨迹内。
    start_idx = min(goal_ref_idx + future_offset, traj_T - 1)
    end_idx = min(start_idx + goal_window_len, traj_T)

    # 取演示中的未来位置片段作为当前观测窗口对应的参考目标。
    goal_window = traj_pos_np[start_idx:end_idx].astype(np.float32)

    # 空窗口时重复演示末端位置；此分支仍依赖演示轨迹非空。
    if goal_window.shape[0] == 0:
        goal_window = np.repeat(
            traj_pos_np[-1:],
            goal_window_len,
            axis=0,
        ).astype(np.float32)

    # 演示尾部长度不足时，用最后一个目标位置在右侧补齐。
    elif goal_window.shape[0] < goal_window_len:
        pad_len = goal_window_len - goal_window.shape[0]
        pad_block = np.repeat(goal_window[-1:], pad_len, axis=0)
        goal_window = np.concatenate([goal_window, pad_block], axis=0)

    # 目标窗口不足 To 时继续补齐，超过 To 时截断，使其与观测窗口逐项对应。
    if goal_window_len < To:
        pad_len = To - goal_window_len
        pad_block = np.repeat(goal_window[-1:], pad_len, axis=0)
        goal_window = np.concatenate([goal_window, pad_block], axis=0)

    elif goal_window_len > To:
        goal_window = goal_window[:To]

    goal_pos_window = goal_window[:, :3].astype(np.float32)
    # 取最终目标窗口的最后一个位置作为本步记录的 goal_pos。
    goal_pos = goal_pos_window[-1].copy()

    # ------------------------------------------------------------
    # 6. goal errors
    # ------------------------------------------------------------
    # 逐位置计算参考目标减实际历史位置，得到 [To, 3] 误差。
    goal_error_np = goal_pos_window - rollout_obs_window
    # 先计算平方距离，再计算带小常数的距离及近似单位方向。
    goal_dist2_np = np.sum(goal_error_np ** 2, axis=-1, keepdims=True).astype(np.float32)
    goal_dist_np = np.sqrt(goal_dist2_np + 1e-6).astype(np.float32)
    goal_dir_np = goal_error_np / (goal_dist_np + 1e-6)

    # ------------------------------------------------------------
    # 7. subgoals
    # ------------------------------------------------------------
    # 把固定的环境接近目标复制 To 次，便于逐历史时刻计算误差。
    subgoal_reach_seq_np = np.repeat(
        subgoal_reach_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

    # 把演示末端的拉动目标复制 To 次。
    subgoal_pull_seq_np = np.repeat(
        subgoal_pull_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

    # 分别计算接近目标和拉动目标相对历史位置的误差。
    subgoal_reach_error_np = subgoal_reach_seq_np - rollout_obs_window
    subgoal_pull_error_np = subgoal_pull_seq_np - rollout_obs_window

    # 阶段 0、1 使用环境接近目标；阶段 2 使用演示末端的拉动目标。
    active_subgoal_pos_np = np.where(
        stage_seq_np < 2.0,
        subgoal_reach_seq_np,
        subgoal_pull_seq_np,
    ).astype(np.float32)

    # 按相同阶段规则选择当前有效子目标对应的误差。
    active_subgoal_error_np = np.where(
        stage_seq_np < 2.0,
        subgoal_reach_error_np,
        subgoal_pull_error_np,
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 8. pull direction
    # 暂时不改，仍然使用 demo_pull_dir_np
    # ------------------------------------------------------------
    # 将演示估计的拉动方向扩展为历史序列。
    pull_dir_np = np.repeat(
        demo_pull_dir_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

    # 只在阶段 2 保留拉动方向，其余阶段置零；当前 cond 中对应字段仍被注释。
    pull_dir_np *= (stage_seq_np == 2.0).astype(np.float32)

    # ------------------------------------------------------------
    # 9. cond dict
    #
    # 注意：
    # - progress_seq_np 只用于 logging，不放进 cond。
    # - mask 不作为普通特征拼接，只和 cond 并列放进 cond_t。
    # ------------------------------------------------------------
    print(
        f"[rollout cond] t={t}, step_progress_idx={progress_idx}, "
        f"stage_seq={stage_seq_np.squeeze().tolist()}, "
        f"action_stage={action_stage_seq_np.squeeze().tolist()}, "
        f"pull_dir_last={pull_dir_np[-1].tolist()}, "
        f"goal_error_last={goal_error_np[-1].tolist()}",
        flush=True,
    )

    # 组装单回合条件张量，常见形状为 [1, To, D]。
    # 关节、进度、active_subgoal_error 和 pull_dir 的写入仍按原代码保持注释状态。
    # 本实现同时在 cond 和外层 cond_t 中保存部分掩码，不把它们当作普通数值特征。
    cond = {
        "pos": torch.from_numpy(rollout_obs_window).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        # 如果你的 assemble_conditions 需要 joint_pos，就打开这个。
        # "joint_pos": torch.from_numpy(rollout_joint_window).unsqueeze(0).to(
        #     device=device,
        #     dtype=torch.float32,
        # ),

        "stage": torch.from_numpy(stage_seq_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        # progress 已经从模型 cond 里移除。
        # 如果 assemble_conditions 里还读 cond["progress"]，需要同步删掉。
        # "progress": torch.from_numpy(progress_seq_np).unsqueeze(0).to(
        #     device=device,
        #     dtype=torch.float32,
        # ),

        # 这是演示未来目标窗口的位置，与按阶段切换的 active_subgoal_pos 是不同字段。
        "stage_goal_pos": torch.from_numpy(goal_pos_window).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        # "goal_error": torch.from_numpy(goal_error_np).unsqueeze(0).to(
        #     device=device,
        #     dtype=torch.float32,
        # ),

        "goal_dist2": torch.from_numpy(goal_dist2_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        "goal_dist": torch.from_numpy(goal_dist_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        "goal_dir": torch.from_numpy(goal_dir_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        "subgoal_pull_pos": torch.from_numpy(subgoal_pull_seq_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        "subgoal_pull_error": torch.from_numpy(subgoal_pull_error_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        "active_subgoal_pos": torch.from_numpy(active_subgoal_pos_np).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),

        # 如果你的 20-dim cond 真的用的是 active_subgoal_error，
        # 那就把这个打开，并在 assemble_conditions 里使用它。
        # "active_subgoal_error": torch.from_numpy(active_subgoal_error_np).unsqueeze(0).to(
        #     device=device,
        #      dtype=torch.float32,
        #  ),

        # pull_dir 暂时不改；如果 assemble_conditions 使用它，需要打开。
        # "pull_dir": torch.from_numpy(pull_dir_np).unsqueeze(0).to(
        #     device=device,
        #     dtype=torch.float32,
        # ),

        "obs_padding_mask": obs_padding_mask,
        "action_padding_mask": action_padding_mask,
        "action_loss_mask": action_loss_mask,
    }

    # 有图像时添加批次维并写入条件，无图像时显式设为 None。
    if rollout_gripper_window is not None:
        cond["rgb_gripper"] = torch.from_numpy(rollout_gripper_window).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
    else:
        cond["rgb_gripper"] = None

    return {
        "progress_idx": progress_idx,
        "goal_pos": goal_pos,
        "goal_idx": start_idx,

        "goal_error_np": goal_error_np,
        "stage_seq_np": stage_seq_np,
        "action_stage_seq_np": action_stage_seq_np,
        "progress_seq_np": progress_seq_np,

        "rollout_obs_window": rollout_obs_window,
        "rollout_joint_window": rollout_joint_window,
        "active_subgoal_error_np": active_subgoal_error_np,

        # 提供策略需要的完整条件结构；外层其他字段主要用于执行记录和奖励计算。
        "cond_t": {
            "cond": cond,

            # obs-side masks
            "obs_valid_mask": obs_valid_mask,             # [1, To], True = valid
            "obs_padding_mask": obs_padding_mask,         # [1, To], True = padding

            # action-side masks
            "action_valid_mask": action_valid_mask,       # [1, Ta], True = valid
            "action_padding_mask": action_padding_mask,   # [1, Ta], True = padding
            "action_loss_mask": action_loss_mask,         # [1, Ta], 1 = valid

            # backward compatible aliases
            "obs_mask": obs_valid_mask.float(),           # [1, To], 1 = valid
            "action_mask": action_loss_mask,              # [1, Ta], 1 = valid
            "src_key_padding_mask": obs_padding_mask,     # legacy name; means obs padding mask

            # future action stage, aligned with decoder/action tokens
            # 未来阶段长度为 Ta，观测阶段长度为 To，不能将二者混用。
            "action_stage": action_stage_seq,             # [1, Ta, 1]

            # obs-side stage
            "stage": cond["stage"],                       # [1, To, 1]

            # 记录当前环境步数；这不是扩散去噪过程的时间步。
            "t": torch.full((1,), int(t), device=device, dtype=torch.long),

            # compatibility only; not model condition
            "progress_idx": torch.full(
                (1,),
                int(progress_idx),
                device=device,
                dtype=torch.long,
            ),

            "batch_size": 1,
        },
    }

# 从选中的动作序列取第一步动作，转成可交给环境的 float32 NumPy 向量。
def _first_action_np(action_candidate: torch.Tensor) -> np.ndarray:
    """
    Convert selected action candidate to one executable action [Da].
    Accepts:
        [1, Ta, Da]
        [Ta, Da]
        [Da]
    """
    # 三维输入取第一个批次的第一步；二维输入取第一步；一维输入直接使用。
    if action_candidate.ndim == 3:
        action = action_candidate[0, 0, :]
    elif action_candidate.ndim == 2:
        action = action_candidate[0, :]
    elif action_candidate.ndim == 1:
        action = action_candidate
    else:
        raise ValueError(f"Unexpected action_candidate shape: {action_candidate.shape}")

    # 脱离梯度图并移到 CPU，以便与 NumPy 环境接口交互。
    return action.detach().cpu().numpy().astype(np.float32)


# 为相同条件生成多个动作序列，可通过 critic 选择候选，最终只执行所选序列第一步。
def select_rollout_action(
    select_with_critic,
    model,
    critic_network,
    cond_norm_params,
    build_out,
    action_norm_params,
    rollout_step,
    device,
):
    # 每次固定生成 8 条候选动作序列，即使未启用 critic 选择也会生成这些候选。
    K = 8

    cond_t = build_out["cond_t"]
    # 复制同一个环境条件，供策略在一个批次中生成多个候选。
    cond_t_k = repeat_cond_t_batch(cond_t, K)

    # 调用扩散策略生成位置动作序列；具体动作尺度和反归一化逻辑由策略方法实现。
    action_seq = model.predict_action(
        cond=cond_t_k,
        action_mode="pos",
        action_params=action_norm_params,
        cond_norm_params=cond_norm_params,
        rollout_step=rollout_step,

    )

    action_seq = action_seq.to(device)  # [K, Ta, Da]

    # 启用时构造归一化条件，再将候选序列交给动作选择器评分和选择。
    if select_with_critic:
        # 复用训练时的条件组装工具，使用传入的归一化参数。
        n_cond, next_n_cond, visual_cond, action_mask, stage_out, next_stage_out = assemble_conditions(
            cond_t=cond_t_k,
            cond_norm_params=cond_norm_params,
            batch_size=K,
            device=device,
            error_scale=1000,
            pull_start_t=36,
            build_action_mask=False,
        )

        # If critic still expects cond/action same temporal length,
        # use last obs-cond and expand to action horizon.
        # 若观测与动作长度不同，用最后一个观测条件沿动作时间维重复，满足 critic 接口。
        if n_cond.ndim == 3 and n_cond.shape[1] != action_seq.shape[1]:
            n_cond = n_cond[:, -1:, :].expand(
                -1,
                action_seq.shape[1],
                -1,
            ).contiguous()

        # 视觉条件长度不匹配时，同样重复最后一帧的条件特征。
        if visual_cond is not None and visual_cond.ndim >= 3 and visual_cond.shape[1] != action_seq.shape[1]:
            visual_cond = visual_cond[:, -1:].expand(
                -1,
                action_seq.shape[1],
                *visual_cond.shape[2:],
            ).contiguous()

        # 具体评分汇总与候选选择方式由 select_action 实现，返回候选及诊断信息。
        action_candidate, selected_idx, scores, probs = select_action(
            critic_network,
            action_seq=action_seq,
            n_cond=n_cond,
            visual_obs_seq=visual_cond,
        )

        # 只提取所选动作序列的第一步，下一环境步将重新预测。
        v_policy = _first_action_np(action_candidate)

        return {
            "v_policy": v_policy,
            # 当前执行动作是策略动作的副本，没有在这里额外修正。
            "v_exec": v_policy.copy(),
            "scores": scores,
            "probs": probs,
            "selected_idx": selected_idx,
        }

    else:
        # 未启用 critic 时固定使用第一条候选的第一步动作。
        v_policy = action_seq[0, 0, :].detach().cpu().numpy().astype(np.float32)

        return {
            "v_policy": v_policy,
            # 当前执行动作是策略动作的副本，没有在这里额外修正。
            "v_exec": v_policy.copy(),
            "scores": None,
            "probs": None,
            "selected_idx": None,
        }


# 保存单回合动作、位置、误差、扩散过程及最终抽屉开度等诊断文件。
def save_rollout_result(
    model,
    result,
    roll_dir:Path,
):
    # 创建该回合的输出目录。
    roll_dir.mkdir(parents=True, exist_ok=True)

    # 读取策略记录的初始噪声统计；记录内容依赖模型中的日志实现。
    noise_stats = model.noise_stats_log
    if len(noise_stats) > 0:
        noise_arr = np.array(
            [[s["mean"], s["std"], s["min"], s["max"]] for s in noise_stats],
            dtype=np.float32,
        )
        save_array_stats(noise_arr, roll_dir, prefix="init_noise_stats")

    # 读取去噪轨迹；外层只在整次 rollout 开始时清空日志，因此多回合间可能累计。
    trace = model.diffusion_trace_log
    if len(trace) > 0:
        trace_arr = np.array(
            [
                [
                    s.get("rollout_step", -1),
                    s["step"],
                    s["timestep"],
                    s["x_mean"],
                    s["x_std"],
                    s["x_min"],
                    s["x_max"],
                    s["x0_mean"],
                    s["x0_std"],
                    s["x0_min"],
                    s["x0_max"],
                ]
                for s in trace
            ],
            dtype=np.float32,
        )
        # 按照环境步数分组保存去噪轨迹，每条记录含 11 列。
        _save_trace_by_rollout_step(trace_arr, roll_dir, prefix="diffusion_trace")

    # 分别保存预测动作、实际执行动作、环境控制量、位置和误差等统计。
    save_array_stats(result["pred_actions"], roll_dir, prefix="pred_actions")
    save_array_stats(result["exec_actions"], roll_dir, prefix="exec_actions")
    save_array_stats(result["controls"], roll_dir, prefix="env_controls")
    save_array_stats(result["errors"], roll_dir, prefix="errors")
    save_array_stats(result["positions"], roll_dir, prefix="positions")
    save_array_stats(result["next_position"], roll_dir, prefix="next_positions")
    save_array_stats(result["goal_positions"], roll_dir, prefix="goal_positions")
    save_array_stats(result["goal_indices"].astype(np.float32), roll_dir, prefix="goal_indices")
    save_array_stats(result["stage_seq"].reshape(-1, 1), roll_dir, prefix="stage_seq")
    save_array_stats(result["progress_seq"].reshape(-1, 1), roll_dir, prefix="progress_seq")
    

    # 保存每步误差和动作的范数，便于比较随时间的变化。
    np.savez(
        roll_dir / "norms.npz",
        error_norms=result["error_norms"],
        action_norms=result["action_norms"],
    )
    # 最终开度同时保存为二进制数组和文本；此处要求回合已实际执行过步骤。
    final_open_len = float(result["final_open_len"])
    np.save(
        roll_dir / "final_open_length.npy",
        np.asarray([final_open_len], dtype=np.float32),
    )
    with open(roll_dir/"final_open_len.txt","w") as f:
        f.write(f"{final_open_len:.8f}\n")

# 按 rollout_step 对去噪记录分组并输出统计及 CSV。
# 下方原有文档字符串写了 7 列，但当前断言和 CSV 实际使用 11 列。
def _save_trace_by_rollout_step(trace_arr: np.ndarray, out_dir: Path, prefix: str):
        """
        trace_arr: [N, 7] columns =
        [rollout_step, denoise_step_i, timestep, mean, std, min, max]
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        # 11 列依次为环境步、去噪迭代、扩散时间步，以及 x、x0 各自的均值/标准差/最小值/最大值。
        assert trace_arr.ndim == 2 and trace_arr.shape[1] == 11, f"bad trace_arr shape: {trace_arr.shape}"

        # 取得日志中出现的所有环境步数，逐步筛选对应去噪记录。
        rollout_steps = np.unique(trace_arr[:, 0]).astype(int)
        for rs in rollout_steps:
            part = trace_arr[trace_arr[:, 0] == rs]  # [Mi,7]
        # 你想要“看 denoise”，最好不要把第0~2列也算进 stats（它们不是随机变量）
        # 所以这里只保存第3~6列 stats（mean/std/min/max），但原始数据仍然完整保存在 npz 里
            # 当前实际将完整的 11 列传给统计函数，并没有按上方旧注释仅截取统计量列。
            save_array_stats(part, out_dir, prefix=f"{prefix}_rs{rs:03d}")

        # 另外再存一个更可读的 txt/csv（可选）
            csv_path = out_dir / f"{prefix}_rs{rs:03d}.csv"
            np.savetxt(
                csv_path,
                part,
                delimiter=",",
                header="rollout_step,denoise_i,timestep,x_mean,x_std,x_min,x_max,x0_mean,x0_std,x0_min,x0_max",
                comments=""
            )
# 执行一个环境回合：重置 → 构建条件 → 生成并执行动作 → 计算奖励 → 保存轨迹。
def rollout_one_episode(
    model,
    critic_network,
    select_with_critic:bool,
    cond_norm_params,
    cfg,
    device,
    replaybuffer,
    rollout_env, # env after reset,get obs
    action_norm_params,
    demo_info,
    log_dir,
    epoch,
    rollout_i,
    phase="rollout",
):
    # 读取模型需要的历史观测窗口长度。
    To = int(cfg.training.obs_hor_len)
    # 读取时间间隔用于抽屉速度估计；当前动作没有乘以 dt 转换。
    dt = float(cfg.skill.dt)
    # 读取机器人相对位移尺度；当前相关动作换算语句已注释，此变量未用于实际控制。
    max_rel_pos = float(rollout_env.robot.max_rel_pos)
    # 回合执行步数上限使用经验回放序列长度。
    max_step = replaybuffer.seq_len
    #rollout_env.manual_base_bias = np.array([0.005, 0.0, 0.0], dtype=np.float32)
    # 重置环境后读取观测和本回合任务关键点。
    env_obs = rollout_env.reset() # get scene obs (target pos) then send to build_cond func
    # print(f"obs has keys: {env_obs.keys()}")
    # print("eef position:", env_obs["position"])
    # print("eef orientation:", env_obs.get("orientation", None))
    # print("view_mtx:", env_obs.get("view_mtx", None))
    env_goal = rollout_env.gt_keypoint
    # 提取经验回放所需的观测结构，作为第一个转移的当前状态。
    out = extract_obs(obs=env_obs) # [position,joint,gripper]
    

    eef_pos = env_obs["position"].astype(np.float32)
    joint_pos = env_obs["joints"].astype(np.float32)

    # 用初始位置重复填满长度 To 的历史队列，起始虚拟历史由 padding 掩码标记。
    obs_hist = deque([eef_pos.copy()] * To, maxlen=To) # use this to compute reward 
    
    # 关节历史使用相同长度及初始填充方式；队列满后自动移除最旧元素。
    joint_hist = deque([joint_pos.copy()] * To, maxlen=To)

    gripper_hist = None
    if env_obs.get("rgb_gripper", None) is not None:
        # 若有夹爪图像，转成 float32 并除以 255，作为视觉历史的初始帧。
        gripper_obs = env_obs["rgb_gripper"].astype(np.float32) / 255.0
        gripper_hist = deque([gripper_obs.copy()] * To, maxlen=To)

    # 初始化轨迹记录列表；total_r 累计环境原始奖励，回放保存的则是塑形奖励。
    total_r = 0.0
    pred_actions = []
    exec_actions = []
    controls = []
    errors = []
    positions = []
    joint_positions = []
    next_position = []
    next_joint_positions = []
    err_norm = []
    action_norm = []
    goal_indices = []
    goal_positions = []
    transition_seq = []
    rollout_stage_seq = []
    rollout_progress_seq = []
    final_open_len = None
    done = False
    progress_idx = 0

    # 每个环境步重新构建条件并预测动作，采用每次只执行第一步的滚动方式。
    for t in range(max_step):
        # 根据最新观测队列、环境目标、演示信息和当前步数构造策略输入。
        build_out = build_rollout_cond(
            cfg=cfg,
            device=device,
            replaybuffer=replaybuffer,
            t=t,
            eef_pos=eef_pos,
            obs_hist=obs_hist,
            joint_hist=joint_hist,
            gripper_hist=gripper_hist,
            demo_info=demo_info,
            env_goal = env_goal,
            progress_idx=progress_idx,
        )
 


        #cond_t = build_out["cond_t"]
        progress_idx = build_out["progress_idx"]

        # 记录动作执行前的最新位置，以及参考目标、阶段和历史误差窗口。
        positions.append(build_out["rollout_obs_window"][-1].copy())
        joint_positions.append(build_out["rollout_joint_window"][-1].copy())
        goal_positions.append(build_out["goal_pos"].copy())
        goal_indices.append(build_out["goal_idx"])
        rollout_stage_seq.append(build_out["stage_seq_np"].copy())
        rollout_progress_seq.append(build_out["progress_seq_np"].copy())
        errors.append(build_out["goal_error_np"].copy())
        err_norm.append(float(np.linalg.norm(build_out["goal_error_np"][-1])))
       
        # 预测多个候选序列，并按配置决定是否由 critic 参与选择。
        act_out = select_rollout_action(
            select_with_critic = select_with_critic,
            model=model,
            critic_network=critic_network,
            cond_norm_params=cond_norm_params,
            build_out=build_out,
            action_norm_params=action_norm_params,
            rollout_step=t,
            device=device
            )
        
            

        v_policy = act_out["v_policy"]
        v_exec = act_out["v_exec"]
       
        # 分别保存策略动作和执行动作，并记录执行动作的欧氏范数。
        pred_actions.append(v_policy.copy())
        exec_actions.append(v_exec.copy())
        action_norm.append(float(np.linalg.norm(v_exec)))

        #u = np.clip((v_exec * dt) / max_rel_pos, -1.0, 1.0).astype(np.float32)
        # 当前直接将策略输出作为环境控制量，不进行上方注释中的缩放与裁剪。
        u = v_exec
        controls.append(u.copy())

        #time.sleep(0.05)
        env_obs = rollout_env.get_obs()
        # 读取执行前的抽屉开度；此处依赖 state 第 16 项代表抽屉位移。
        drawer_open_length = env_obs["state"][16]
        # 向环境执行一步，获得下一观测、原始奖励与结束标志。
        next_obs, reward, done, info = rollout_env.step(u)
        next_drawer_open_length =  next_obs["state"][16]
        # 每一步更新最终开度，回合结束时保留最后一次观测值。
        final_open_len = float(next_drawer_open_length)
        # 用相邻开度差除以配置 dt 估计速度，要求 dt 非零。
        drawer_open_velocity = (next_drawer_open_length- drawer_open_length)/dt
        print(f"drawer_open_velocity is {drawer_open_velocity}")
        next_obs_out = extract_obs(next_obs)
        next_eef_pos = next_obs_out["position"].astype(np.float32)
        # 复制当前历史并追加新位置，构造奖励计算所需的下一观测窗口。
        next_obs_hist = deque(obs_hist, maxlen=To)
        next_obs_hist.append(next_eef_pos.copy())
        next_rollout_obs_window = np.stack(list(next_obs_hist), axis=0).astype(np.float32)
        curr_active_err_window = build_out["active_subgoal_error_np"]   # [To, 3]
        # 取当前历史窗口末尾的阶段，决定执行后距离应相对于哪个子目标计算。
        curr_stage = float(build_out["stage_seq_np"][-1, 0])

        # 接近和对齐阶段使用环境关键点，拉动阶段使用演示末端目标。
        if curr_stage < 2.0:
            active_goal = env_goal.astype(np.float32)
        else:
            active_goal = demo_info["subgoal_pull_np"].astype(np.float32)

        # 计算执行动作后的新子目标误差，并与移位后的历史误差组成新窗口。
        new_active_error = (active_goal - next_eef_pos).astype(np.float32)   # [3]
        next_active_err_window = np.concatenate(
        [curr_active_err_window[1:], new_active_error[None, :]],
        axis=0
        ).astype(np.float32)
        # 本步目标距离只取新误差窗口最后一个位置的欧氏范数。
        goal_dist = float(np.linalg.norm(next_active_err_window[-1]))

        # 调用公共奖励函数，结合环境奖励、子目标距离及演示位置轨迹计算基础塑形奖励。
        shaped_reward = compute_shaped_reward(cfg = cfg,reward=reward,
                                              goal_dist = goal_dist, rollout_obs_window = next_rollout_obs_window,
                                              traj_pos_np=demo_info["traj_pos_np"], To=To, rollout_step = t, device=device) 
        # 开抽屉任务：在基础奖励上加入拉动阶段的正向位移奖励和最后一步的开度奖励。
        if cfg.skill.name == "calvin_open_drawer":
        
            # 将本步开度变化限制在 [-0.01, 0.01]，避免单步位移奖励过大。
            delta_open = np.clip(next_drawer_open_length - drawer_open_length, -0.01, 0.01)

            if curr_stage >= 2.0:
                # 仅奖励正向开度变化，最大额外单步奖励为 1。
                shaped_reward += 100.0 * max(delta_open, 0.0)
            
            # 仅在达到预设最后一步时添加分档奖励；提前 done 的回合不会进入该末步奖励分支。
            if t >= max_step -1:

                thresholds = np.array([
                0.075, 0.080, 0.085, 0.090, 0.095,
                0.100, 0.105, 0.110, 0.115, 0.120, 0.125,0.13
                ])

                rewards = np.array([
                0.5, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0 # 11.0
                ])

                # 寻找严格小于当前开度的最大阈值索引；恰好等于某阈值时落入前一档。
                idx = np.searchsorted(thresholds, next_drawer_open_length, side="left") - 1

                if idx >= 0:
                    shaped_reward += rewards[idx]
        # 关抽屉任务奖励开度减小，并在预设最后一步按剩余开度加分。
        elif cfg.skill.name == "calvin_close_drawer":
            # 将本步开度变化限制在 [-0.01, 0.01]，避免单步位移奖励过大。
            delta_open = np.clip(next_drawer_open_length - drawer_open_length, -0.01, 0.01) 
            if curr_stage >= 2.0:
                # 只奖励负向开度变化，奖励上限同样为 1。
                shaped_reward += 100.0 * max((-delta_open), 0.0)
            # 仅在达到预设最后一步时添加分档奖励；提前 done 的回合不会进入该末步奖励分支。
            if t >= max_step -1:

                thresholds = np.array([

                0.0,0.01, 0.02, 0.03, 0.04, 0.05,
                0.06, 0.07, 0.09, 0.1, 0.13, 0.14,0.15, 0.16
                ])

                rewards = np.array([
                # 保留原数组：末尾的 0,5 是两个整数元素，并非小数 0.5。
                12.0, 11.0, 10.0, 9.0, 8.5,
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0,5
                ])

                # 寻找严格小于当前开度的最大阈值索引；恰好等于某阈值时落入前一档。
                idx = np.searchsorted(thresholds, next_drawer_open_length, side="left") - 1
          

                # 本行覆盖前一次 idx 计算；关抽屉实际使用小于或等于当前开度的最大阈值档位。
                idx = np.searchsorted(thresholds, next_drawer_open_length, side="right") - 1
                # 限制奖励索引，防止小于最低阈值或超过奖励数组范围时发生越界。
                idx = int(np.clip(idx, 0, len(rewards) - 1))

                shaped_reward += rewards[idx]
                        

        print(f"out has keys: {out.keys()}")
        # 保存完整转移：当前观测、执行动作、下一观测、塑形奖励和环境 done。
        transition_seq.append(
            {
                "obs": out,
                "action": v_exec.copy(),
                "next_obs": next_obs_out,
                "reward": shaped_reward,
                "done": done,
            }
        )

        # 将下一状态变为下一轮循环的当前状态，并更新位置与关节历史。
        out = next_obs_out
        eef_pos = next_obs["position"].astype(np.float32)
        joint_pos = next_obs["joints"].astype(np.float32)

        obs_hist.append(eef_pos.copy())
        joint_hist.append(joint_pos.copy())

        # 若视觉历史已启用且下一观测含图像，则追加归一化后的新图像。
        if gripper_hist is not None and next_obs.get("rgb_gripper", None) is not None:
            next_gripper = next_obs["rgb_gripper"].astype(np.float32) / 255.0
            gripper_hist.append(next_gripper.copy())

        # 记录动作执行后的实际位置和关节状态。
        next_position.append(eef_pos.copy())
        next_joint_positions.append(joint_pos.copy())

        # 累计原始环境奖励，不包含上面额外构造的塑形奖励。
        total_r += float(reward)

        # 环境返回结束标志时立即停止本回合。
        if done:
            break

    print(f"transition len is: {len(transition_seq)}, replaybuffer.seq_len is: {replaybuffer.seq_len}")

    # 只有完整达到指定长度的轨迹才写入回放；提前结束的短轨迹在这里不会保存到回放。
    if len(transition_seq) == replaybuffer.seq_len:
        replaybuffer.save_sequence(transition_seq)

    # 汇总本回合轨迹为 NumPy 数组，同时保留转移序列及最终开度。
    result = {
        "final_open_len":final_open_len,
        "pred_actions": np.asarray(pred_actions, dtype=np.float32),
        "exec_actions": np.asarray(exec_actions, dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
        "joint_positions": np.asarray(joint_positions, dtype=np.float32),
        "next_position": np.asarray(next_position, dtype=np.float32),
        "next_joint_positions": np.asarray(next_joint_positions, dtype=np.float32),
        "errors": np.asarray(errors, dtype=np.float32),
        "controls": np.asarray(controls, dtype=np.float32),
        "goal_positions": np.asarray(goal_positions, dtype=np.float32),
        "goal_indices": np.asarray(goal_indices, dtype=np.int32).reshape(-1, 1),
        "stage_seq": np.asarray(rollout_stage_seq, dtype=np.float32),
        "progress_seq": np.asarray(rollout_progress_seq, dtype=np.float32),
        "demo_pull_dir": demo_info["demo_pull_dir_np"],
        "total_reward": total_r,
        "transition_seq": transition_seq,
        "error_norms": np.asarray(err_norm, dtype=np.float32),
        "action_norms": np.asarray(action_norm, dtype=np.float32),

    }

    # 按阶段、训练轮数和回合编号组织单回合分析目录。
    roll_dir = (
        Path(log_dir)
        / "rollout_analysis"
        / phase
        / f"epoch_{epoch:04d}"
        / f"rollout_{rollout_i:02d}"
    )
    result["rollout_dir"] = roll_dir

    # 将本回合记录和模型诊断日志保存到对应目录。
    save_rollout_result(
        model=model,
        result=result,
        roll_dir=roll_dir,
    )

    return result


# 批量执行多个回合，保存各回合结果及最终开度汇总，最后恢复模型原有训练状态。
def rollout(
    model,
    critic_network,
    select_with_critic,
    cond_norm_params,
    cfg,
    device,
    replaybuffer,
    rollout_env,
    training_dataset,
    action_norm_params,
    log_dir,
    epoch,
    phase="rollout",
):
    # 开始本次评估前清空模型的噪声、去噪和动作日志。
    model.noise_stats_log.clear()
    model.diffusion_trace_log.clear()
    model.action_trace_log.clear()
    # 记录模型进入本函数时是否为训练模式，以便结束或异常时恢复。
    was_training = model.training

    try:
        # 切换评估模式；本函数没有显式 no_grad，是否禁用梯度还取决于 predict_action 等内部实现。
        model.eval()
        # 本批所有回合共享从固定演示中提取的参考目标和方向。
        demo_info = prepare_rollout_targets(training_dataset=training_dataset,cfg=cfg)

        all_results = []
        # 按配置的评估次数逐个运行回合，每次由 rollout_one_episode 重置环境。
        for rollout_i in range(cfg.training.rollout_times):
            result = rollout_one_episode(
                model=model,
                critic_network=critic_network,
                select_with_critic=select_with_critic,
                cond_norm_params=cond_norm_params,
                cfg=cfg,
                device=device,
                replaybuffer=replaybuffer,
                rollout_env=rollout_env,
                action_norm_params=action_norm_params,
                demo_info=demo_info,
                log_dir=log_dir,
                epoch=epoch,
                rollout_i=rollout_i,
                phase=phase,
            )
            all_results.append(result)
        # 收集非 None 的最终开度，转换成数组以计算跨回合统计。
        final_open_lengths = np.asarray([
                                    
            result["final_open_len"]
            for result in all_results
            if result["final_open_len"] is not None
        ],dtype = np.float32,
        )
        # 有有效开度时计算均值、总体方差和总体标准差。
        if final_open_lengths.size > 0:
            final_open_mean = float(np.mean(final_open_lengths))

    # 总体方差，除以 N
            final_open_variance = float(np.var(final_open_lengths, ddof=0))

    # 标准差
            final_open_std = float(np.std(final_open_lengths, ddof=0)) 
            epoch_rollout_dir = (
                Path(log_dir)
                / "rollout_analysis"
                / phase
                / f"epoch_{epoch:04d}"
            )

            epoch_rollout_dir.mkdir(parents=True, exist_ok=True)

            # 将有效结果重新编号，并与最终开度组成两列表格。
            summary_array = np.column_stack(
                [
            np.arange(final_open_lengths.size, dtype=np.int32),
            final_open_lengths,
                ]
            )

            np.savetxt(
                epoch_rollout_dir / "final_open_lengths.csv",
                summary_array,
                delimiter=",",
                header="rollout_index,final_open_length",
                comments="",
                fmt=["%d", "%.8f"],
            )
            # 准备回合数量、原始开度列表及均值/方差/标准差的 JSON 汇总。
            statistics = {
            "num_rollouts": int(final_open_lengths.size),
            "final_open_lengths": [
                float(x) for x in final_open_lengths
            ],
            "mean": final_open_mean,
            "variance": final_open_variance,
            "std": final_open_std,
            }

            with open(
                epoch_rollout_dir
                / "final_open_length_statistics.json",
                "w",
            ) as f:
                json.dump(
                statistics,
                f,
                indent=4,
            )

        return all_results

    # 无论正常返回还是发生异常，都执行原训练状态的恢复逻辑。
    finally:
        # 进入函数前处于训练模式时恢复 train；原本为 eval 时继续保持 eval。
        if was_training:
            model.train()
