#This code is for testing normalizer function. It is not involved in the training  
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


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

    # Convert the raw observation returned by the environment into
    # a compact and consistent format used by rollout and replay buffer.
    out = {}

    # ---------- state ----------
    # TCP state must always be available in the environment observation.
    if "position" not in obs:
        raise KeyError("extract_obs: raw obs does not contain key 'position'.")

    # Convert TCP position / pose to float32.
    state = np.asarray(obs["position"], dtype=np.float32)
    out["position"] = state

    # ---------- joint_pos ----------
    # Different environments may use either "joints" or "joint_pos".
    if "joints" in obs:
        joint_pos = np.asarray(obs["joints"], dtype=np.float32)
        out["joints"] = joint_pos
    elif "joint_pos" in obs:
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)
        out["joints"] = joint_pos
    else:
        # Joint information is optional.
        out["joints"],out["joint_pos"] = None,None

    # ---------- optional visual obs ----------
    # Store the gripper camera image if visual observation is available.
    if "rgb_gripper" in obs:
        rgb = np.asarray(obs["rgb_gripper"])
        
        # Keep RGB images in uint8 format for consistent storage.
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        out["rgb_gripper"] = rgb

    return out


def compute_shaped_reward(cfg, reward, goal_dist, rollout_obs_window, traj_pos_np, rollout_step, To, device):
    # Weights for three reward sources:
    #   1. environment reward
    #   2. demonstration tracking distance
    #   3. goal distance
    w_env = float(getattr(cfg.training, "reward_w_env", 0.1))
    w_demo = float(getattr(cfg.training, "reward_w_demo", 0.8))
    w_goal = float(getattr(cfg.training, "reward_w_goal", 0.1))

    # Measure how far the rollout history deviates from the demonstration.
    demo_dist = build_reward_deque_dist(rollout_obs_window, traj_pos_np, rollout_step, To)

    # Weighted shaped reward:
    # positive environment reward,
    # minus demonstration deviation,
    # minus distance to the task goal.
    shaped_reward = torch.tensor(
        w_env * float(reward) - w_demo * float(demo_dist) - w_goal * float(goal_dist),
        dtype=torch.float32,
        device=device,
    )
    return shaped_reward


def build_reward_deque_dist(rollout_obs_window, traj_pos_np, rollout_step, To):
    # Only compare the valid part of the observation history.
    # At the beginning of rollout, fewer than To observations may exist.
    valid_len = min(rollout_step+1,To)

    # Extract the valid rollout observation history.
    rollout_obs_window_valid = rollout_obs_window[-valid_len:]

    # Select the temporally aligned demonstration states.
    demo_hist_idx = np.arange(max(0, rollout_step - valid_len + 1), rollout_step + 1)
    demo_obs_window_valid = traj_pos_np[demo_hist_idx]

    # Mean Euclidean distance between rollout states and demonstration states.
    # A smaller value means the rollout follows the demonstration more closely.
    demo_window_dist = np.linalg.norm(
    rollout_obs_window_valid - demo_obs_window_valid,
    axis=-1
    ).mean()
    return demo_window_dist


def robot_obs_to_local_action(robot_obs, euler_order="xyz"):
    """
    Convert a TCP pose trajectory stored in robot_obs into local 6D delta actions.

    The generated action is:

        action_t =
        [dx_body, dy_body, dz_body,
         dRx, dRy, dRz]

    where:
        - translation is expressed in the current TCP/body frame
        - rotation is represented by a rotation vector (SO(3) log map)
        - dRx, dRy, dRz are NOT Euler-angle differences

    Parameters
    ----------
    robot_obs : np.ndarray
        Shape [T, D], with at least 6 dimensions.

        Expected layout:
            robot_obs[:, 0:3] = TCP position [x, y, z]
            robot_obs[:, 3:6] = TCP Euler orientation [roll, pitch, yaw]

    euler_order : str
        Euler convention used by the dataset.
        Default: "xyz".

    Returns
    -------
    actions : np.ndarray
        Shape [T-1, 6]

        actions[t, 0:3]:
            local translation increment

        actions[t, 3:6]:
            local rotation vector
    """

    # Convert input trajectory to a NumPy array.
    robot_obs = np.asarray(robot_obs)

    # The trajectory must have shape [time, feature].
    if robot_obs.ndim != 2:
        raise ValueError(
            f"robot_obs must have shape [T, D], got {robot_obs.shape}"
        )

    # At least 3 position + 3 orientation dimensions are required.
    if robot_obs.shape[1] < 6:
        raise ValueError(
            f"robot_obs must contain at least 6 dimensions, "
            f"got {robot_obs.shape[1]}"
        )

    # --------------------------------------------------
    # 1. Extract raw TCP trajectory
    # --------------------------------------------------

    # First three dimensions are Cartesian position.
    positions = robot_obs[:, 0:3]

    # Next three dimensions are Euler orientation.
    eulers = robot_obs[:, 3:6]

    T = len(robot_obs)

    # At least two poses are required to compute a delta action.
    if T < 2:
        raise ValueError(
            "robot_obs must contain at least two timesteps."
        )

    # --------------------------------------------------
    # 2. Euler -> SO(3)
    # --------------------------------------------------

    # Convert Euler angles to rotation objects to avoid directly
    # subtracting Euler angles when computing orientation changes.
    rotations = R.from_euler(
        euler_order,
        eulers
    )

    # --------------------------------------------------
    # 3. Compute local translation
    #
    # dp_body = R_t^T (p_{t+1} - p_t)
    # --------------------------------------------------

    # Cartesian displacement expressed in the world frame.
    dp_world = positions[1:] - positions[:-1]

    # Transform the displacement into the current TCP/body frame.
    dp_body = rotations[:-1].inv().apply(
        dp_world
    )

    # --------------------------------------------------
    # 4. Compute relative rotation
    #
    # R_delta = R_t^T R_{t+1}
    # --------------------------------------------------

    # Relative orientation from the current TCP frame
    # to the TCP frame at the next timestep.
    R_delta = (
        rotations[:-1].inv()
        * rotations[1:]
    )

    # --------------------------------------------------
    # 5. SO(3) -> so(3) ~= R^3
    #
    # rotvec = log(R_delta)^vee
    # --------------------------------------------------

    # Convert the relative rotation into a 3D rotation vector.
    rotvec = R_delta.as_rotvec()

    # --------------------------------------------------
    # 6. Construct 6D local action
    # --------------------------------------------------

    # Concatenate local translation and local rotation increments.
    actions = np.concatenate(
        [
            dp_body,
            rotvec
        ],
        axis=-1
    )

    return actions.astype(np.float32)
