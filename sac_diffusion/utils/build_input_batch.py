
import torch
from typing import Optional, Union
import numpy as np
from omegaconf import DictConfig


def build_sliding_window_batch(
    cfg: DictConfig,
    batch: dict[str, torch.Tensor],
    env_goal_pos,
    t: Optional[Union[int, list[int], np.ndarray, torch.Tensor]] = None,
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
    To = int(getattr(cfg.training, "obs_hor_len", getattr(cfg.training, "obs_horizon", 4)))
    Ta = int(horizon if horizon is not None else getattr(cfg.training, "act_hor_len", getattr(cfg.training, "action_horizon", 16)))

    # Stage boundaries:
    # stage 0: reaching
    # stage 1: alignment
    # stage 2: pulling
    reach_end = int(getattr(cfg.training, "reach_end_t", 32))
    align_end = int(getattr(cfg.training, "align_end_t", 36))

    # Future offsets used to construct stage-dependent goals.
    future_offset = int(getattr(cfg.training, "future_offset", Ta))
    pull_extra_offset = int(getattr(cfg.training, "pull_extra_offset", 4))
    pull_start_t_cfg = int(getattr(cfg.training, "pull_start_t", align_end))

    # Read current and next observations.
    obs = batch["state"] if "state" in batch else batch["obs"]
    next_obs = batch["next_state"] if "next_state" in batch else batch["next_obs"]

    # Optional robot joint states.
    joint_pos = batch.get("joint_pos", None)
    next_joint_pos = batch.get("next_joint_pos", None)

    # Optional wrist/gripper RGB observation.
    rgb_gripper_obs = None
    if "gripper_obs" in batch and batch["gripper_obs"].get("rgb_gripper", None) is not None:
        rgb_gripper_obs = batch["gripper_obs"]["rgb_gripper"]

    # Action and transition information.
    action = batch["action"]
    reward = batch.get("reward", None)
    done = batch.get("done", None)
    next_action = batch.get("next_action", None)

    # B: batch size, T: trajectory length, D: observation dimension.
    B, T, D = obs.shape
    device = obs.device

        # -------- normalize env_goal_pos to [B, 1, 3] --------
    # Convert the environment goal to a float tensor on the same device.
    if not isinstance(env_goal_pos, torch.Tensor):
        env_goal_pos = torch.as_tensor(env_goal_pos, device=device, dtype=torch.float32)
    else:
        env_goal_pos = env_goal_pos.to(device=device, dtype=torch.float32)

    # Handle a single XYZ goal: [3] -> [B,1,3].
    if env_goal_pos.ndim == 1:
        env_goal_pos = env_goal_pos.view(1, 1, 3).expand(B, 1, 3)

    elif env_goal_pos.ndim == 2:
        # [1,3] or [B,3]
        # Broadcast a single goal across the whole batch if necessary.
        if env_goal_pos.shape[0] == 1 and B > 1:
            env_goal_pos = env_goal_pos.expand(B, -1)

        if env_goal_pos.shape[0] != B:
            raise ValueError(f"env_goal_pos batch {env_goal_pos.shape[0]} != B {B}")

        env_goal_pos = env_goal_pos[:, None, :]  # [B,1,3]

    elif env_goal_pos.ndim == 3:
        # [B,1,3] or [B,T,3]
        # Broadcast the batch dimension if only one goal sequence is given.
        if env_goal_pos.shape[0] == 1 and B > 1:
            env_goal_pos = env_goal_pos.expand(B, -1, -1)

        if env_goal_pos.shape[0] != B:
            raise ValueError(f"env_goal_pos batch {env_goal_pos.shape[0]} != B {B}")

        # Keep only one goal position per trajectory.
        env_goal_pos = env_goal_pos[:, :1, :]  # [B,1,3]

    else:
        raise ValueError(f"Bad env_goal_pos shape: {env_goal_pos.shape}")

    # Keep only XYZ coordinates.
    env_goal_pos = env_goal_pos[..., :3].contiguous()  # [B,1,3]

    # Split observations into position and orientation components.
    if D == 6:
        pos = obs[:, :, :3]
        ori = obs[:, :, 3:6]
        next_pos = next_obs[:, :, :3]
        next_ori = next_obs[:, :, 3:6]
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
    t_min = 0
    t_max = T - 1

    # If t is not provided, sample one control step according to sample_mode.
    if t is None:
        sample_mode = getattr(cfg.training, "sample_mode", "uniform")

        # Uniformly sample t across the complete trajectory.
        if sample_mode == "uniform":
            t_i = int(np.random.randint(t_min, t_max + 1))

        elif sample_mode == "biased_50_30_20":
            # Helper for sampling from a closed integer interval.
            def _randint_closed(lo, hi):
                lo = int(lo)
                hi = int(hi)
                if lo > hi:
                    return None
                return int(np.random.randint(lo, hi + 1))

            # Reach-stage interval.
            reach_lo = t_min
            reach_hi = min(t_max, reach_end - 1)

            # Pull-stage interval.
            pull_lo = max(t_min, align_end)
            pull_hi = t_max

            # transition around reach -> align
            # Include a wider temporal region around the reach/align transition.
            trans01_lo = max(t_min, reach_end - To)
            trans01_hi = min(t_max, reach_end + Ta - 1)

            # Sampling distribution:
            # 50% transition region, 30% pull stage, 20% reach stage.
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
    elif isinstance(t, int):
        t_i = int(t)
    elif isinstance(t, torch.Tensor):
        vals = [int(x) for x in t.detach().cpu().view(-1).tolist()]
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]
    elif isinstance(t, np.ndarray):
        vals = [int(x) for x in t.reshape(-1).tolist()]
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]
    else:
        vals = [int(x) for x in t]
        if len(vals) != 1:
            raise ValueError("This version expects a single t.")
        t_i = vals[0]

    # Ensure the selected timestep stays inside the trajectory.
    t_i = int(np.clip(t_i, t_min, t_max))

    # -------- history obs index + future action index --------
    # Observation history: [t-To+1, ..., t].
    obs_raw_idx = torch.arange(t_i - To + 1, t_i + 1, device=device)   # [To]

    # Future action horizon: [t, ..., t+Ta-1].
    act_raw_idx = torch.arange(t_i, t_i + Ta, device=device)           # [Ta]

    # Masks indicate which raw indices correspond to valid trajectory steps.
    obs_mask_1d = ((obs_raw_idx >= 0) & (obs_raw_idx < T)).float()
    action_mask_1d = ((act_raw_idx >= 0) & (act_raw_idx < T)).float()

    # Clamp invalid indices so tensor indexing remains valid.
    # The masks above are later used to identify padded elements.
    obs_idx = obs_raw_idx.clamp(0, T - 1)
    act_idx = act_raw_idx.clamp(0, T - 1)

    # Build one-step-ahead indices.
    next_obs_idx = (obs_idx + 1).clamp(0, T - 1)
    next_act_idx = (act_idx + 1).clamp(0, T - 1)

    # Expand validity masks across the batch dimension.
    obs_mask = obs_mask_1d.view(1, To).expand(B, -1)          # [B,To]
    action_mask = action_mask_1d.view(1, Ta).expand(B, -1)    # [B,Ta]

    # -------- gather obs windows --------
    # Extract current and next observation windows.
    pos_window = pos[:, obs_idx, :]                                    # [B,To,3]
    ori_window = ori[:, obs_idx, :] if ori is not None else None
    next_pos_window = next_pos[:, obs_idx, :]                          # [B,To,3]
    next_ori_window = next_ori[:, obs_idx, :] if next_ori is not None else None

    # Optional joint-state windows.
    joint_pos_window = joint_pos[:, obs_idx, :] if joint_pos is not None else None
    next_joint_window = next_joint_pos[:, obs_idx, :] if next_joint_pos is not None else None

    # Optional wrist/gripper RGB history window.
    rgb_gripper_obs_window = (
        rgb_gripper_obs[:, obs_idx, :, :, :] if rgb_gripper_obs is not None else None
    )

    # -------- gather future action windows --------
    # Extract future action trajectory from the selected timestep.
    act_window = action[:, act_idx, :]                                 # [B,Ta,Da]
    next_action_window = next_action[:, act_idx, :] if next_action is not None else None

    # Corresponding reward and termination windows.
    reward_window = reward[:, act_idx, :] if reward is not None else None
    done_window = done[:, act_idx, :] if done is not None else None

    # -------- stage/progress for obs side --------
    # Encode task stage for each observation timestep:
    # 0 = reach, 1 = align, 2 = pull.
    stage_ids_obs = torch.zeros(To, device=device, dtype=torch.long)
    stage_ids_obs[(obs_idx >= reach_end) & (obs_idx < align_end)] = 1
    stage_ids_obs[obs_idx >= align_end] = 2
    stage_seq = stage_ids_obs.view(1, To, 1).expand(B, -1, -1).float()

    # Stage sequence for the next observation state.
    next_stage_ids_obs = torch.zeros(To, device=device, dtype=torch.long)
    next_stage_ids_obs[(next_obs_idx >= reach_end) & (next_obs_idx < align_end)] = 1
    next_stage_ids_obs[next_obs_idx >= align_end] = 2
    next_stage_seq = next_stage_ids_obs.view(1, To, 1).expand(B, -1, -1).float()

    # Normalized temporal progress in [0, 1].
    progress_seq = (obs_idx.float() / max(T - 1, 1)).view(1, To, 1).expand(B, -1, -1)
    next_progress_seq = (next_obs_idx.float() / max(T - 1, 1)).view(1, To, 1).expand(B, -1, -1)

    # -------- stage/progress for action side --------
    # Stage labels for each future action step.
    action_stage_ids = torch.zeros(Ta, device=device, dtype=torch.long)
    action_stage_ids[(act_idx >= reach_end) & (act_idx < align_end)] = 1
    action_stage_ids[act_idx >= align_end] = 2
    action_stage_seq = action_stage_ids.view(1, Ta, 1).expand(B, -1, -1).float()

    # -------- stage-aware goal positions on obs side --------
    # Stage 0 goal: next demonstrated position at the corresponding timestep.
    reach_goal_seq = next_pos[:, obs_idx, :]                           # [B,To,3]

    # Stage 1 goal: fixed position near the end of the alignment phase.
    align_anchor_idx = min(max(align_end - 1, 0), T - 1)
    align_goal = next_pos[:, align_anchor_idx:align_anchor_idx + 1, :]
    align_goal_seq = align_goal.expand(-1, To, -1)

    # Stage 2 goal: a future demonstrated position ahead of the current timestep.
    pull_goal_idx = torch.clamp(obs_idx + future_offset + pull_extra_offset, 0, T - 1)
    pull_goal_seq = next_pos[:, pull_goal_idx, :]

    # Select the appropriate goal according to the task stage.
    stage_ids_expand = stage_ids_obs.view(1, To, 1).expand(B, -1, 3)
    stage_goal_pos_seq = torch.where(
        stage_ids_expand == 0,
        reach_goal_seq,
        torch.where(stage_ids_expand == 1, align_goal_seq, pull_goal_seq),
    )

    # next stage-aware goal positions
    # Construct equivalent stage-dependent goals for next observations.
    next_reach_goal_seq = next_pos[:, next_obs_idx, :]
    next_align_goal_seq = align_goal.expand(-1, To, -1)

    next_pull_goal_idx = torch.clamp(
        next_obs_idx + future_offset + pull_extra_offset, 0, T - 1
    )
    next_pull_goal_seq = next_pos[:, next_pull_goal_idx, :]

    next_stage_ids_expand = next_stage_ids_obs.view(1, To, 1).expand(B, -1, 3)
    next_stage_goal_pos_seq = torch.where(
        next_stage_ids_expand == 0,
        next_reach_goal_seq,
        torch.where(next_stage_ids_expand == 1, next_align_goal_seq, next_pull_goal_seq),
    )

    # -------- errors --------
    # Cartesian error between the current position and stage-dependent goal.
    goal_error = stage_goal_pos_seq - pos_window
    next_goal_error = next_stage_goal_pos_seq - next_pos_window

    # Squared distance, Euclidean distance, and normalized goal direction.
    goal_dist2 = (goal_error ** 2).sum(dim=-1, keepdim=True)
    goal_dist = torch.sqrt(goal_dist2 + 1e-6)
    goal_dir = goal_error / (goal_dist + 1e-6)

    next_goal_dist2 = (next_goal_error ** 2).sum(dim=-1, keepdim=True)
    next_goal_dist = torch.sqrt(next_goal_dist2 + 1e-6)
    next_goal_dir = next_goal_error / (next_goal_dist + 1e-6)

    # -------- global subgoals --------
    # Define the start of the pull phase and the final trajectory timestep.
    pull_start_t = min(max(pull_start_t_cfg, 0), T - 1)
    final_t = T - 1

    # stage 0/1: env goal / handle goal
    # Use the environment-provided goal for reach/alignment.
    subgoal_reach = env_goal_pos  # [B,1,3]

    # stage 2: demo pull goal
    # Use the final demonstrated position as the pull subgoal.
    subgoal_pull = next_pos[:, final_t:final_t + 1, :]  # [B,1,3]

    # Expand fixed subgoals across the observation horizon.
    subgoal_reach_seq = subgoal_reach.expand(-1, To, -1)  # [B,To,3]
    subgoal_pull_seq = subgoal_pull.expand(-1, To, -1)    # [B,To,3]

    # next side
    # The global subgoal itself is unchanged for the next-state representation.
    next_subgoal_reach_seq = subgoal_reach_seq
    next_subgoal_pull_seq = subgoal_pull_seq

    # Compute errors to both reach and pull subgoals.
    subgoal_reach_error = subgoal_reach_seq - pos_window
    subgoal_pull_error = subgoal_pull_seq - pos_window

    next_subgoal_reach_error = next_subgoal_reach_seq - next_pos_window
    next_subgoal_pull_error = next_subgoal_pull_seq - next_pos_window

    # Select the currently active subgoal according to the task stage.
    # Reach/alignment use env_goal_pos; pull uses the final demonstration position.
    active_subgoal_pos = torch.where(
        stage_seq.expand(-1, -1, 3) < 2.0,
        subgoal_reach_seq,
        subgoal_pull_seq,
    )
    active_subgoal_error = torch.where(
        stage_seq < 2.0,
        subgoal_reach_error,
        subgoal_pull_error,
    )

    # Active subgoal representation for the next state.
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
    raw_pull_vec = next_pos[:, final_t, :] - next_pos[:, pull_start_t, :]
    raw_pull_norm = torch.norm(raw_pull_vec, dim=-1, keepdim=True)

    # If possible, also estimate the average local displacement
    # along the demonstrated pull segment.
    if pull_start_t < final_t:
        pull_seg = next_pos[:, pull_start_t:final_t + 1, :]
        local_deltas = pull_seg[:, 1:, :] - pull_seg[:, :-1, :]
        mean_local_delta = local_deltas.mean(dim=1)
        mean_local_norm = torch.norm(mean_local_delta, dim=-1, keepdim=True)
    else:
        mean_local_delta = torch.zeros_like(raw_pull_vec)
        mean_local_norm = torch.zeros_like(raw_pull_norm)

    # Fall back to the mean local motion if the global displacement is nearly zero.
    use_local = (raw_pull_norm < 1e-6).float()
    pull_vec = (1.0 - use_local) * raw_pull_vec + use_local * mean_local_delta
    pull_norm = (1.0 - use_local) * raw_pull_norm + use_local * mean_local_norm

    # Final fallback direction when no valid motion direction can be estimated.
    default_pull_dir = torch.tensor(
        [-1.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).view(1, 3).expand(B, -1)

    # Normalize the demonstrated pulling direction.
    demo_pull_dir = pull_vec / (pull_norm + 1e-8)
    use_default = (pull_norm < 1e-6).float()
    demo_pull_dir = (1.0 - use_default) * demo_pull_dir + use_default * default_pull_dir

    # Broadcast the pull direction over the observation horizon.
    base_pull_dir_seq = demo_pull_dir.unsqueeze(1).expand(-1, To, -1).clone()

    # Pull direction is only active during stage 2.
    pull_mask = (stage_seq == 2.0).float()
    next_pull_mask = (next_stage_seq == 2.0).float()

    pull_dir_seq = base_pull_dir_seq * pull_mask
    next_pull_dir_seq = base_pull_dir_seq * next_pull_mask

    # Build the conditioning dictionary passed to the policy / critic.
    cond = {
        "pos": pos_window,
        "ori": ori_window,
        "next_pos": next_pos_window,
        "next_ori": next_ori_window,

        # 之前去掉 joint_pos，这里继续默认不放进 cond。
        # 如果之后要加回来，取消注释即可。
        #"joint_pos": joint_pos_window,
        #"next_joint_pos": next_joint_window,

        "next_action": next_action_window,
        "rgb_gripper": rgb_gripper_obs_window,

        # Stage information for current and next observations.
        "stage": stage_seq,
        "next_stage": next_stage_seq,
        #"progress": progress_seq,
        #"next_progress": next_progress_seq,

        # Stage-dependent local target positions.
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
        "obs_valid_mask": obs_mask.bool(),
        "obs_padding_mask": obs_mask == 0,

        # Action masks for valid, padded, and loss-bearing timesteps.
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
        "cond_t": {
            "cond": cond,
            "obs_idx": obs_idx.unsqueeze(0).expand(B, -1),      # [B,To]
            "act_idx": act_idx.unsqueeze(0).expand(B, -1),      # [B,Ta]
            "obs_mask": obs_mask,                              # [B,To]
            "action_mask": action_mask,                        # [B,Ta]
            "src_key_padding_mask": obs_mask == 0,              # [B,To]
            "t": torch.full((B,), t_i, device=device, dtype=torch.long),
            "stage": stage_seq,
            "action_stage": action_stage_seq,
            "batch_size": B,
        }
    }
