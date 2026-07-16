import numpy as np
import torch
def extract_obs(obs: dict) -> dict:
    """
    Convert raw env observation to a compact dict used by rollout / replaybuffer / training.

    Expected raw env keys (depending on env):
        obs["position"]    : [3] or [6]
        obs["joints"]      : [Dj]
        obs["rgb_gripper"] : [H, W, C]  (optional)

    Returns:
        {
            "position":      np.ndarray,   # [3] or [6]
            "joint_pos":  np.ndarray,   # [Dj]
            "rgb_gripper": np.ndarray,  # optional, uint8
        }
    """
    out = {}

    # ---------- state ----------
    if "position" not in obs:
        raise KeyError("extract_obs: raw obs does not contain key 'position'.")

    state = np.asarray(obs["position"], dtype=np.float32)
    out["position"] = state

    # ---------- joint_pos ----------
    if "joints" in obs:
        joint_pos = np.asarray(obs["joints"], dtype=np.float32)
        out["joints"] = joint_pos
    elif "joint_pos" in obs:
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)
        out["joints"] = joint_pos
    else:
        out["joints"],out["joint_pos"] = None,None

    # ---------- optional visual obs ----------
    if "rgb_gripper" in obs:
        rgb = np.asarray(obs["rgb_gripper"])
        
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        out["rgb_gripper"] = rgb

    return out

def compute_shaped_reward(cfg, reward, goal_dist, rollout_obs_window, traj_pos_np, rollout_step, To, device):
    w_env = float(getattr(cfg.training, "reward_w_env", 0.1))
    w_demo = float(getattr(cfg.training, "reward_w_demo", 0.8))
    w_goal = float(getattr(cfg.training, "reward_w_goal", 0.1))
    demo_dist = build_reward_deque_dist(rollout_obs_window, traj_pos_np, rollout_step, To)
    shaped_reward = torch.tensor(
        w_env * float(reward) - w_demo * float(demo_dist) - w_goal * float(goal_dist),
        dtype=torch.float32,
        device=device,
    )
    return shaped_reward

def build_reward_deque_dist(rollout_obs_window, traj_pos_np, rollout_step, To):
    valid_len = min(rollout_step+1,To)
    rollout_obs_window_valid = rollout_obs_window[-valid_len:]
    demo_hist_idx = np.arange(max(0, rollout_step - valid_len + 1), rollout_step + 1)
    demo_obs_window_valid = traj_pos_np[demo_hist_idx]
    demo_window_dist = np.linalg.norm(
    rollout_obs_window_valid - demo_obs_window_valid,
    axis=-1
    ).mean()
    return demo_window_dist