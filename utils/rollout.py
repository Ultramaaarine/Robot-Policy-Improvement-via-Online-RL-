# cond: total 20 dim: pos:3, joints: 7, subgoal_active_error: 3, pull_dir: 3, progress: 1, goal: 3 (read from env, not from offline dataset)
# is a torch.tensor 
# obs,pos seq has only one batch size, not like offline dataset
# copy multiple time when rollout, generate multiple action sequences for selection 
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

def prepare_rollout_targets(training_dataset, cfg): # reach goal is acquired from env move to build_cond, here only offline data
    align_end = int(getattr(cfg.training, "align_end_t", 36))
    pull_start_t = int(getattr(cfg.training, "pull_start_t", align_end))

    demo_idx = 0 # np.random.random_integers(0,7)
    sample = training_dataset[demo_idx] # single dict

    traj = sample["obs"]
    if isinstance(traj, torch.Tensor):
        traj_np = traj.detach().cpu().numpy().astype(np.float32)
    else:
        traj_np = np.asarray(traj, dtype=np.float32)

    if traj_np.ndim != 2:
        raise ValueError(f"Expected traj_np shape [T,D], got {traj_np.shape}")

    traj_pos_np = traj_np[:, :3].astype(np.float32)
    traj_T = traj_pos_np.shape[0]

    pull_start_t = min(max(pull_start_t, 0), traj_T - 1)
    final_t = traj_T - 1

    
    subgoal_pull_np = traj_pos_np[final_t].copy()

    raw_pull_vec = traj_pos_np[final_t] - traj_pos_np[pull_start_t]
    raw_pull_norm = float(np.linalg.norm(raw_pull_vec))

    if raw_pull_norm < 1e-6:
        pull_seg = traj_pos_np[pull_start_t:final_t + 1]
        if pull_seg.shape[0] >= 2:
            local_deltas = pull_seg[1:] - pull_seg[:-1]
            raw_pull_vec = local_deltas.mean(axis=0).astype(np.float32)
            raw_pull_norm = float(np.linalg.norm(raw_pull_vec))

    if raw_pull_norm < 1e-6:
        demo_pull_dir_np = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
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

def repeat_tensor_tree(x, K: int):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        reps = [K] + [1] * (x.dim() - 1)
        return x.repeat(*reps)

    if isinstance(x, dict):
        return {k: repeat_tensor_tree(v, K) for k, v in x.items()}

    return x


def repeat_cond_t_batch(cond_t: dict, K: int):
    cond_t_k = {}
    for key, value in cond_t.items():
        if key == "batch_size":
            cond_t_k[key] = K
        else:
            cond_t_k[key] = repeat_tensor_tree(value, K)

    cond_t_k["batch_size"] = K
    return cond_t_k

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
    To = int(cfg.training.obs_hor_len)
    Ta = int(cfg.training.act_hor_len)

    future_offset = int(getattr(cfg.training, "future_offset", Ta))
    goal_window_len = int(getattr(cfg.training, "goal_window_len", To))

    reach_end = int(getattr(cfg.training, "reach_end_t", 32))
    align_end = int(getattr(cfg.training, "align_end_t", 36))

    traj_pos_np = demo_info["traj_pos_np"]
    traj_T = int(demo_info["traj_T"])

    max_rollout_len = int(getattr(replaybuffer, "seq_len", traj_T))

    subgoal_reach_np = np.asarray(env_goal, dtype=np.float32).reshape(-1)[:3]
    subgoal_pull_np = np.asarray(demo_info["subgoal_pull_np"], dtype=np.float32).reshape(-1)[:3]
    demo_pull_dir_np = np.asarray(demo_info["demo_pull_dir_np"], dtype=np.float32).reshape(-1)[:3]

    # ------------------------------------------------------------
    # 1. rollout history window: [t-To+1, ..., t]
    # deque 里已经是 left-padded 后的长度 To
    # ------------------------------------------------------------
    rollout_obs_window = np.stack(list(obs_hist), axis=0).astype(np.float32)
    rollout_joint_window = np.stack(list(joint_hist), axis=0).astype(np.float32)

    if rollout_obs_window.shape[0] != To:
        raise ValueError(
            f"rollout_obs_window length {rollout_obs_window.shape[0]} != To {To}"
        )

    if rollout_joint_window.shape[0] != To:
        raise ValueError(
            f"rollout_joint_window length {rollout_joint_window.shape[0]} != To {To}"
        )

    rollout_gripper_window = None
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
    obs_raw_idx = np.arange(t - To + 1, t + 1)
    act_raw_idx = np.arange(t, t + Ta)

    obs_valid_np = (obs_raw_idx >= 0) & (obs_raw_idx < max_rollout_len)
    action_valid_np = (act_raw_idx >= 0) & (act_raw_idx < max_rollout_len)

    obs_padding_np = ~obs_valid_np
    action_padding_np = ~action_valid_np

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
    obs_step_idx = np.clip(obs_step_raw_idx, 0, max_rollout_len - 1)

    action_step_raw_idx = np.arange(t, t + Ta)
    action_step_idx = np.clip(action_step_raw_idx, 0, max_rollout_len - 1)

    # logging only, not used as model condition
    progress_seq_np = (
        obs_step_idx.astype(np.float32) / max(max_rollout_len - 1, 1)
    ).reshape(To, 1)

    # obs-side stage: [To, 1]
    stage_seq_np = np.zeros((To, 1), dtype=np.float32)
    stage_seq_np[(obs_step_idx >= reach_end) & (obs_step_idx < align_end)] = 1.0
    stage_seq_np[obs_step_idx >= align_end] = 2.0

    # action-side stage: [Ta, 1]
    action_stage_seq_np = np.zeros((Ta, 1), dtype=np.float32)
    action_stage_seq_np[
        (action_step_idx >= reach_end) & (action_step_idx < align_end)
    ] = 1.0
    action_stage_seq_np[action_step_idx >= align_end] = 2.0

    action_stage_seq = torch.from_numpy(action_stage_seq_np).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )  # [1, Ta, 1]

    # ------------------------------------------------------------
    # 5. goal window
    # use rollout step, not progress_idx
    # ------------------------------------------------------------
    goal_ref_idx = int(np.clip(t, 0, traj_T - 1))

    start_idx = min(goal_ref_idx + future_offset, traj_T - 1)
    end_idx = min(start_idx + goal_window_len, traj_T)

    goal_window = traj_pos_np[start_idx:end_idx].astype(np.float32)

    if goal_window.shape[0] == 0:
        goal_window = np.repeat(
            traj_pos_np[-1:],
            goal_window_len,
            axis=0,
        ).astype(np.float32)

    elif goal_window.shape[0] < goal_window_len:
        pad_len = goal_window_len - goal_window.shape[0]
        pad_block = np.repeat(goal_window[-1:], pad_len, axis=0)
        goal_window = np.concatenate([goal_window, pad_block], axis=0)

    if goal_window_len < To:
        pad_len = To - goal_window_len
        pad_block = np.repeat(goal_window[-1:], pad_len, axis=0)
        goal_window = np.concatenate([goal_window, pad_block], axis=0)

    elif goal_window_len > To:
        goal_window = goal_window[:To]

    goal_pos_window = goal_window[:, :3].astype(np.float32)
    goal_pos = goal_pos_window[-1].copy()

    # ------------------------------------------------------------
    # 6. goal errors
    # ------------------------------------------------------------
    goal_error_np = goal_pos_window - rollout_obs_window
    goal_dist2_np = np.sum(goal_error_np ** 2, axis=-1, keepdims=True).astype(np.float32)
    goal_dist_np = np.sqrt(goal_dist2_np + 1e-6).astype(np.float32)
    goal_dir_np = goal_error_np / (goal_dist_np + 1e-6)

    # ------------------------------------------------------------
    # 7. subgoals
    # ------------------------------------------------------------
    subgoal_reach_seq_np = np.repeat(
        subgoal_reach_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

    subgoal_pull_seq_np = np.repeat(
        subgoal_pull_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

    subgoal_reach_error_np = subgoal_reach_seq_np - rollout_obs_window
    subgoal_pull_error_np = subgoal_pull_seq_np - rollout_obs_window

    active_subgoal_pos_np = np.where(
        stage_seq_np < 2.0,
        subgoal_reach_seq_np,
        subgoal_pull_seq_np,
    ).astype(np.float32)

    active_subgoal_error_np = np.where(
        stage_seq_np < 2.0,
        subgoal_reach_error_np,
        subgoal_pull_error_np,
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 8. pull direction
    # 暂时不改，仍然使用 demo_pull_dir_np
    # ------------------------------------------------------------
    pull_dir_np = np.repeat(
        demo_pull_dir_np[None, :],
        To,
        axis=0,
    ).astype(np.float32)

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
            "action_stage": action_stage_seq,             # [1, Ta, 1]

            # obs-side stage
            "stage": cond["stage"],                       # [1, To, 1]

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

def _first_action_np(action_candidate: torch.Tensor) -> np.ndarray:
    """
    Convert selected action candidate to one executable action [Da].
    Accepts:
        [1, Ta, Da]
        [Ta, Da]
        [Da]
    """
    if action_candidate.ndim == 3:
        action = action_candidate[0, 0, :]
    elif action_candidate.ndim == 2:
        action = action_candidate[0, :]
    elif action_candidate.ndim == 1:
        action = action_candidate
    else:
        raise ValueError(f"Unexpected action_candidate shape: {action_candidate.shape}")

    return action.detach().cpu().numpy().astype(np.float32)


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
    K = 8

    cond_t = build_out["cond_t"]
    cond_t_k = repeat_cond_t_batch(cond_t, K)

    action_seq = model.predict_action(
        cond=cond_t_k,
        action_mode="pos",
        action_params=action_norm_params,
        cond_norm_params=cond_norm_params,
        rollout_step=rollout_step,

    )

    action_seq = action_seq.to(device)  # [K, Ta, Da]

    if select_with_critic:
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
        if n_cond.ndim == 3 and n_cond.shape[1] != action_seq.shape[1]:
            n_cond = n_cond[:, -1:, :].expand(
                -1,
                action_seq.shape[1],
                -1,
            ).contiguous()

        if visual_cond is not None and visual_cond.ndim >= 3 and visual_cond.shape[1] != action_seq.shape[1]:
            visual_cond = visual_cond[:, -1:].expand(
                -1,
                action_seq.shape[1],
                *visual_cond.shape[2:],
            ).contiguous()

        action_candidate, selected_idx, scores, probs = select_action(
            critic_network,
            action_seq=action_seq,
            n_cond=n_cond,
            visual_obs_seq=visual_cond,
        )

        v_policy = _first_action_np(action_candidate)

        return {
            "v_policy": v_policy,
            "v_exec": v_policy.copy(),
            "scores": scores,
            "probs": probs,
            "selected_idx": selected_idx,
        }

    else:
        v_policy = action_seq[0, 0, :].detach().cpu().numpy().astype(np.float32)

        return {
            "v_policy": v_policy,
            "v_exec": v_policy.copy(),
            "scores": None,
            "probs": None,
            "selected_idx": None,
        }


def save_rollout_result(
    model,
    result,
    roll_dir:Path,
):
    roll_dir.mkdir(parents=True, exist_ok=True)

    noise_stats = model.noise_stats_log
    if len(noise_stats) > 0:
        noise_arr = np.array(
            [[s["mean"], s["std"], s["min"], s["max"]] for s in noise_stats],
            dtype=np.float32,
        )
        save_array_stats(noise_arr, roll_dir, prefix="init_noise_stats")

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
        _save_trace_by_rollout_step(trace_arr, roll_dir, prefix="diffusion_trace")

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
    

    np.savez(
        roll_dir / "norms.npz",
        error_norms=result["error_norms"],
        action_norms=result["action_norms"],
    )
    final_open_len = float(result["final_open_len"])
    np.save(
        roll_dir / "final_open_length.npy",
        np.asarray([final_open_len], dtype=np.float32),
    )
    with open(roll_dir/"final_open_len.txt","w") as f:
        f.write(f"{final_open_len:.8f}\n")

def _save_trace_by_rollout_step(trace_arr: np.ndarray, out_dir: Path, prefix: str):
        """
        trace_arr: [N, 7] columns =
        [rollout_step, denoise_step_i, timestep, mean, std, min, max]
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        assert trace_arr.ndim == 2 and trace_arr.shape[1] == 11, f"bad trace_arr shape: {trace_arr.shape}"

        rollout_steps = np.unique(trace_arr[:, 0]).astype(int)
        for rs in rollout_steps:
            part = trace_arr[trace_arr[:, 0] == rs]  # [Mi,7]
        # 你想要“看 denoise”，最好不要把第0~2列也算进 stats（它们不是随机变量）
        # 所以这里只保存第3~6列 stats（mean/std/min/max），但原始数据仍然完整保存在 npz 里
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
    To = int(cfg.training.obs_hor_len)
    dt = float(cfg.skill.dt)
    max_rel_pos = float(rollout_env.robot.max_rel_pos)
    max_step = replaybuffer.seq_len
    #rollout_env.manual_base_bias = np.array([0.005, 0.0, 0.0], dtype=np.float32)
    env_obs = rollout_env.reset() # get scene obs (target pos) then send to build_cond func
    # print(f"obs has keys: {env_obs.keys()}")
    # print("eef position:", env_obs["position"])
    # print("eef orientation:", env_obs.get("orientation", None))
    # print("view_mtx:", env_obs.get("view_mtx", None))
    env_goal = rollout_env.gt_keypoint
    out = extract_obs(obs=env_obs) # [position,joint,gripper]
    

    eef_pos = env_obs["position"].astype(np.float32)
    joint_pos = env_obs["joints"].astype(np.float32)

    obs_hist = deque([eef_pos.copy()] * To, maxlen=To) # use this to compute reward 
    
    joint_hist = deque([joint_pos.copy()] * To, maxlen=To)

    gripper_hist = None
    if env_obs.get("rgb_gripper", None) is not None:
        gripper_obs = env_obs["rgb_gripper"].astype(np.float32) / 255.0
        gripper_hist = deque([gripper_obs.copy()] * To, maxlen=To)

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

    for t in range(max_step):
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

        positions.append(build_out["rollout_obs_window"][-1].copy())
        joint_positions.append(build_out["rollout_joint_window"][-1].copy())
        goal_positions.append(build_out["goal_pos"].copy())
        goal_indices.append(build_out["goal_idx"])
        rollout_stage_seq.append(build_out["stage_seq_np"].copy())
        rollout_progress_seq.append(build_out["progress_seq_np"].copy())
        errors.append(build_out["goal_error_np"].copy())
        err_norm.append(float(np.linalg.norm(build_out["goal_error_np"][-1])))
       
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
       
        pred_actions.append(v_policy.copy())
        exec_actions.append(v_exec.copy())
        action_norm.append(float(np.linalg.norm(v_exec)))

        #u = np.clip((v_exec * dt) / max_rel_pos, -1.0, 1.0).astype(np.float32)
        u = v_exec
        controls.append(u.copy())

        #time.sleep(0.05)
        env_obs = rollout_env.get_obs()
        drawer_open_length = env_obs["state"][16]
        next_obs, reward, done, info = rollout_env.step(u)
        next_drawer_open_length =  next_obs["state"][16]
        final_open_len = float(next_drawer_open_length)
        drawer_open_velocity = (next_drawer_open_length- drawer_open_length)/dt
        print(f"drawer_open_velocity is {drawer_open_velocity}")
        next_obs_out = extract_obs(next_obs)
        next_eef_pos = next_obs_out["position"].astype(np.float32)
        next_obs_hist = deque(obs_hist, maxlen=To)
        next_obs_hist.append(next_eef_pos.copy())
        next_rollout_obs_window = np.stack(list(next_obs_hist), axis=0).astype(np.float32)
        curr_active_err_window = build_out["active_subgoal_error_np"]   # [To, 3]
        curr_stage = float(build_out["stage_seq_np"][-1, 0])

        if curr_stage < 2.0:
            active_goal = env_goal.astype(np.float32)
        else:
            active_goal = demo_info["subgoal_pull_np"].astype(np.float32)

        new_active_error = (active_goal - next_eef_pos).astype(np.float32)   # [3]
        next_active_err_window = np.concatenate(
        [curr_active_err_window[1:], new_active_error[None, :]],
        axis=0
        ).astype(np.float32)
        goal_dist = float(np.linalg.norm(next_active_err_window[-1]))

        shaped_reward = compute_shaped_reward(cfg = cfg,reward=reward,
                                              goal_dist = goal_dist, rollout_obs_window = next_rollout_obs_window,
                                              traj_pos_np=demo_info["traj_pos_np"], To=To, rollout_step = t, device=device) 
        if cfg.skill.name == "calvin_open_drawer":
        
            delta_open = np.clip(next_drawer_open_length - drawer_open_length, -0.01, 0.01)

            if curr_stage >= 2.0:
                shaped_reward += 100.0 * max(delta_open, 0.0)
            
            if t >= max_step -1:

                thresholds = np.array([
                0.075, 0.080, 0.085, 0.090, 0.095,
                0.100, 0.105, 0.110, 0.115, 0.120, 0.125,0.13
                ])

                rewards = np.array([
                0.5, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0 # 11.0
                ])

                idx = np.searchsorted(thresholds, next_drawer_open_length, side="left") - 1

                if idx >= 0:
                    shaped_reward += rewards[idx]
        elif cfg.skill.name == "calvin_close_drawer":
            delta_open = np.clip(next_drawer_open_length - drawer_open_length, -0.01, 0.01) 
            if curr_stage >= 2.0:
                shaped_reward += 100.0 * max((-delta_open), 0.0)
            if t >= max_step -1:

                thresholds = np.array([

                0.0,0.01, 0.02, 0.03, 0.04, 0.05,
                0.06, 0.07, 0.09, 0.1, 0.13, 0.14,0.15, 0.16
                ])

                # rewards = np.array([
                # 0.5, 1.0, 2.0, 3.0, 4.0,
                # 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0,12.0
                # ])
                rewards = np.array([
                12.0, 11.0, 10.0, 9.0, 8.5,
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,0.5
                ])

                idx = np.searchsorted(thresholds, next_drawer_open_length, side="left") - 1
          

                idx = np.searchsorted(thresholds, next_drawer_open_length, side="right") - 1
                idx = int(np.clip(idx, 0, len(rewards) - 1))

                shaped_reward += rewards[idx]
                        

        print(f"out has keys: {out.keys()}")
        transition_seq.append(
            {
                "obs": out,
                "action": v_exec.copy(),
                "next_obs": next_obs_out,
                "reward": shaped_reward,
                "done": done,
            }
        )

        out = next_obs_out
        eef_pos = next_obs["position"].astype(np.float32)
        joint_pos = next_obs["joints"].astype(np.float32)

        obs_hist.append(eef_pos.copy())
        joint_hist.append(joint_pos.copy())

        if gripper_hist is not None and next_obs.get("rgb_gripper", None) is not None:
            next_gripper = next_obs["rgb_gripper"].astype(np.float32) / 255.0
            gripper_hist.append(next_gripper.copy())

        next_position.append(eef_pos.copy())
        next_joint_positions.append(joint_pos.copy())

        total_r += float(reward)

        if done:
            break

    print(f"transition len is: {len(transition_seq)}, replaybuffer.seq_len is: {replaybuffer.seq_len}")

    if len(transition_seq) == replaybuffer.seq_len:
        replaybuffer.save_sequence(transition_seq)

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

    roll_dir = (
        Path(log_dir)
        / "rollout_analysis"
        / phase
        / f"epoch_{epoch:04d}"
        / f"rollout_{rollout_i:02d}"
    )
    result["rollout_dir"] = roll_dir

    save_rollout_result(
        model=model,
        result=result,
        roll_dir=roll_dir,
    )

    return result


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
    model.noise_stats_log.clear()
    model.diffusion_trace_log.clear()
    model.action_trace_log.clear()
    was_training = model.training

    try:
        model.eval()
        demo_info = prepare_rollout_targets(training_dataset=training_dataset,cfg=cfg)

        all_results = []
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
        final_open_lengths = np.asarray([
                                    
            result["final_open_len"]
            for result in all_results
            if result["final_open_len"] is not None
        ],dtype = np.float32,
        )
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

    finally:
        if was_training:
            model.train()