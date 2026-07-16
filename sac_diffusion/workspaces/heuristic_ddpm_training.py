import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import tqdm
import wandb
import logging
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Union
from torch.utils.data import DataLoader

from sac_diffusion.workspaces.base_workspace import BaseWorkspace
from sac_diffusion.policy.lowdim_policy import LowdimPolicy
from sac_diffusion.datasets.calvin_critic_offline_dataset import CalvinCriticOfflineDataset
from sac_diffusion.models.exponential_moving_average import EMAModel
import copy
from sac_diffusion.models.normalizer import Normalizer
from sac_diffusion.models.lr_scheduler import get_scheduler
from sac_diffusion.utils.env_maker import make_env
from sac_diffusion.utils.soft_label import HeuristicNet, get_soft_label
from sac_diffusion.models.replay_buffer import ReplayBuffer
from sac_diffusion.utils.common_funcs import extract_obs

logger = logging.getLogger(__name__)


class TrainLowDimWorkspace(BaseWorkspace):
    def __init__(
        self,
        cfg: OmegaConf,
        output_dir=Path(__file__).parent.parent.parent.joinpath("outputs")
    ):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device(cfg.training.device)
        self.output_dir = output_dir
        self.global_step = 0
        self.epoch = 0

        # configure model
        self.model: LowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.normalizer = Normalizer()

        # configure optimizer
        self.optimizer: torch.optim.AdamW
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.heu_optimizer: torch.optim.AdamW
        self.heuristic_net: HeuristicNet
        self.heuristic_net = HeuristicNet(6, 256).to(self.device)
        self.heu_optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.heuristic_net.parameters()
        )

        self.replaybuffer: ReplayBuffer
        self.replaybuffer = hydra.utils.instantiate(self.cfg.replay_buffer)

    def _optimizer_to(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device)
        return self.optimizer

    def _heu_optimizer_to(self):
        for state in self.heu_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device)
        return self.heu_optimizer

    def _save_array_stats(self, arr: np.ndarray, out_dir: Path, prefix: str):
        """
        保存原始数据 + stats(json/txt)
        arr: (N, D) or (N, T, D) -> 会展平到 (N*, D)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        if arr.ndim == 3:
            arr2 = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            arr2 = arr
        else:
            raise ValueError(f"arr.ndim must be 2 or 3, got {arr.ndim}")

        stats = {
            "shape": list(arr2.shape),
            "mean": arr2.mean(axis=0).tolist(),
            "std": arr2.std(axis=0).tolist(),
            "min": arr2.min(axis=0).tolist(),
            "max": arr2.max(axis=0).tolist(),
        }

        np.savez(out_dir / f"{prefix}.npz", data=arr2)

        with open(out_dir / f"{prefix}.json", "w") as f:
            json.dump(stats, f, indent=2)
        with open(out_dir / f"{prefix}.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}:{v}\n")
        return stats

    def _save_dataset_action_stats(self, training_dataset, out_dir: Path):
        """
        优先用 dataset.action
        若没有，再 fallback 遍历 __getitem__()
        """
        if hasattr(training_dataset, "action"):
            a = training_dataset.action
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            a = np.asarray(a)
            return self._save_array_stats(a, out_dir, prefix="dataset_actions")

        all_actions = []
        for i in range(len(training_dataset)):
            sample = training_dataset[i]
            act = sample["action"]
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
            all_actions.append(act)
        a = np.stack(all_actions, axis=0)

        return self._save_array_stats(a, out_dir, prefix="dataset_actions")
    
    def _save_trace_by_rollout_step(self, trace_arr: np.ndarray, out_dir: Path, prefix: str):
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
            self._save_array_stats(part, out_dir, prefix=f"{prefix}_rs{rs:03d}")

        # 另外再存一个更可读的 txt/csv（可选）
            csv_path = out_dir / f"{prefix}_rs{rs:03d}.csv"
            np.savetxt(
                csv_path,
                part,
                delimiter=",",
                header="rollout_step,denoise_i,timestep,x_mean,x_std,x_min,x_max,x0_mean,x0_std,x0_min,x0_max",
                comments=""
            )

    def _rollout(
    self,
    rollout_env,
    training_dataset,
    action_norm_params,
    log_dir,
    use_heuristic=False,
    heuristic_net=None,
):
     self.model.noise_stats_log.clear()
     self.model.diffusion_trace_log.clear()
     self.model.action_trace_log.clear()
     was_training = self.model.training

     try:
        self.model.eval()
        if heuristic_net is not None:
            heuristic_net.eval()

        To = int(self.cfg.training.obs_hor_len)
        Ta = int(self.cfg.training.act_hor_len)
        future_offset = int(getattr(self.cfg.training, "future_offset", Ta))
        goal_window_len = int(getattr(self.cfg.training, "goal_window_len", To))

        reach_end = int(getattr(self.cfg.training, "reach_end_t", 36))
        align_end = int(getattr(self.cfg.training, "align_end_t", 44))
        pull_start_t = int(getattr(self.cfg.training, "pull_start_t", align_end))

        demo_idx = 0
        sample = training_dataset[demo_idx]

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

        subgoal_reach_np = traj_pos_np[pull_start_t].copy()
        subgoal_pull_np = traj_pos_np[final_t].copy()

        # ---------------------------------------------------------
        # estimate pull direction from demo trajectory
        # use displacement from pull_start_t to final_t
        # fallback to local average if too small
        # ---------------------------------------------------------
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

        print(f"[rollout] estimated demo_pull_dir = {demo_pull_dir_np}, "
              f"pull_start_t={pull_start_t}, final_t={final_t}", flush=True)

        dt = float(self.cfg.skill.dt)
        max_rel_pos = float(rollout_env.robot.max_rel_pos)

        max_step = self.replaybuffer.seq_len
        heu_noise_scale = float(getattr(self.cfg.training, "heu_noise_scale", 0.1))
        heu_delta_scale = float(getattr(self.cfg.training, "heu_delta_scale", 0.1))

        for rollout_i in range(self.cfg.training.rollout_times):
            obs = rollout_env.reset()
            print(f"obs has keys: {obs.keys()}")

            out = extract_obs(obs=obs)

            eef_pos = obs["position"].astype(np.float32)
            joint_pos = obs["joints"].astype(np.float32)

            obs_hist = deque([eef_pos.copy()] * To, maxlen=To)
            joint_hist = deque([joint_pos.copy()] * To, maxlen=To)

            action_dim = int(training_dataset[0]["action"].shape[-1])

            heu_state_hist = deque(
                [eef_pos.copy()] * self.replaybuffer.seq_len,
                maxlen=self.replaybuffer.seq_len
            )
            heu_action_hist = deque(
                [np.zeros(action_dim, dtype=np.float32)] * self.replaybuffer.seq_len,
                maxlen=self.replaybuffer.seq_len
            )

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
            done = False

            progress_idx = 0

            for t in range(max_step):
                rollout_obs_window = np.stack(list(obs_hist), axis=0).astype(np.float32) # [1,T,D]
                rollout_joint_window = np.stack(list(joint_hist), axis=0).astype(np.float32)

                positions.append(rollout_obs_window[-1].copy())
                joint_positions.append(rollout_joint_window[-1].copy())

                dists = np.linalg.norm(traj_pos_np - eef_pos[None, :], axis=1)
                closest_idx = int(np.argmin(dists))
                if t == 0:
                    progress_idx = closest_idx
                else:
                    progress_idx = max(progress_idx, closest_idx)

                start_idx = min(progress_idx + future_offset, traj_T - 1)
                end_idx = min(start_idx + goal_window_len, traj_T)

                goal_window = traj_pos_np[start_idx:end_idx].astype(np.float32)

                if goal_window.shape[0] == 0:
                    goal_window = np.repeat(traj_pos_np[-1:], goal_window_len, axis=0).astype(np.float32)
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

                goal_positions.append(goal_pos.copy())
                goal_indices.append(start_idx)

                hist_idx = np.arange(t - To + 1, t + 1)
                hist_idx = np.clip(hist_idx, 0, max_step - 1)

                stage_seq_np = np.zeros((To, 1), dtype=np.float32)
                stage_seq_np[(hist_idx >= reach_end) & (hist_idx < align_end)] = 1.0
                stage_seq_np[hist_idx >= align_end] = 2.0

                progress_seq_np = (hist_idx.astype(np.float32) / max(max_step - 1, 1)).reshape(To, 1)

                rollout_stage_seq.append(stage_seq_np.copy())
                rollout_progress_seq.append(progress_seq_np.copy())

                goal_error_np = goal_pos_window - rollout_obs_window
                goal_dist2_np = np.sum(goal_error_np ** 2, axis=-1, keepdims=True).astype(np.float32)
                goal_dist_np = np.sqrt(goal_dist2_np + 1e-6).astype(np.float32)
                goal_dir_np = goal_error_np / (goal_dist_np + 1e-6)

                subgoal_reach_seq_np = np.repeat(subgoal_reach_np[None, :], To, axis=0).astype(np.float32)
                subgoal_pull_seq_np = np.repeat(subgoal_pull_np[None, :], To, axis=0).astype(np.float32)

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

                # ---------------------------------------------------------
                # pull direction feature from demo-estimated pull direction
                # active only in stage 2
                # ---------------------------------------------------------
                pull_dir_np = np.repeat(demo_pull_dir_np[None, :], To, axis=0).astype(np.float32)
                pull_dir_np *= (stage_seq_np == 2.0).astype(np.float32)

                errors.append(goal_error_np.copy())
                err_norm.append(float(np.linalg.norm(goal_error_np[-1])))

                cond = {
                    "pos": torch.from_numpy(rollout_obs_window).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "joint_pos": torch.from_numpy(rollout_joint_window).unsqueeze(0).to(self.device, dtype=torch.float32),

                    "stage": torch.from_numpy(stage_seq_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "progress": torch.from_numpy(progress_seq_np).unsqueeze(0).to(self.device, dtype=torch.float32),

                    "stage_goal_pos": torch.from_numpy(goal_pos_window).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "goal_error": torch.from_numpy(goal_error_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "goal_dist2": torch.from_numpy(goal_dist2_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "goal_dist": torch.from_numpy(goal_dist_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "goal_dir": torch.from_numpy(goal_dir_np).unsqueeze(0).to(self.device, dtype=torch.float32),

                    "subgoal_reach_pos": torch.from_numpy(subgoal_reach_seq_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "subgoal_pull_pos": torch.from_numpy(subgoal_pull_seq_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "subgoal_reach_error": torch.from_numpy(subgoal_reach_error_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "subgoal_pull_error": torch.from_numpy(subgoal_pull_error_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "active_subgoal_pos": torch.from_numpy(active_subgoal_pos_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                    "active_subgoal_error": torch.from_numpy(active_subgoal_error_np).unsqueeze(0).to(self.device, dtype=torch.float32),

                    "pull_dir": torch.from_numpy(pull_dir_np).unsqueeze(0).to(self.device, dtype=torch.float32),

                    "pos_error": torch.from_numpy(goal_error_np).unsqueeze(0).to(self.device, dtype=torch.float32),
                }

                action_seq = self.model.predict_action(
                    cond,
                    action_mode="pos",
                    action_params=action_norm_params,
                    cond_norm_params=self.param_dict,
                    rollout_step=t,
                )

                v_policy = action_seq[0, 0, :].detach().cpu().numpy().astype(np.float32)
                pred_actions.append(v_policy.copy())

                v_exec = v_policy.copy()

                heu_pred_label = None
                if use_heuristic and (heuristic_net is not None):
                    heu_state_hist.append(eef_pos.copy())
                    heu_action_hist.append(v_policy.copy())

                    state_seq = np.stack(list(heu_state_hist), axis=0).astype(np.float32)
                    action_seq_np = np.stack(list(heu_action_hist), axis=0).astype(np.float32)
                    heu_input_np = np.concatenate([state_seq, action_seq_np], axis=-1)

                    heu_input = torch.from_numpy(heu_input_np).to(
                        self.device, dtype=torch.float32
                    ).unsqueeze(0)

                    with torch.no_grad():
                        heu_pred_label = heuristic_net(heu_input)

                    heu_pred_label = heu_pred_label[0].detach().cpu().numpy()
                    if heu_pred_label.ndim == 1:
                        heu_pred_label = heu_pred_label[:, None]

                    last_label = heu_pred_label[-1]
                    init_noise = np.random.normal(
                        loc=0.0,
                        scale=heu_noise_scale,
                        size=v_policy.shape
                    ).astype(np.float32)

                    heuristic_delta = heu_delta_scale * last_label * init_noise
                    v_exec = v_policy + heuristic_delta

                exec_actions.append(v_exec.copy())
                action_norm.append(float(np.linalg.norm(v_exec)))

                u = np.clip((v_exec * dt) / max_rel_pos, -1.0, 1.0).astype(np.float32)
                controls.append(u.copy())

                time.sleep(0.15)
                next_obs, reward, done, info = rollout_env.step(u)

                next_obs_out = extract_obs(next_obs)

                transition = {
                    "obs": out,
                    "action": v_exec.copy(),
                    "next_obs": next_obs_out,
                    "reward": reward,
                    "done": done,
                }
                transition_seq.append(transition)

                out = next_obs_out

                eef_pos = next_obs["position"].astype(np.float32)
                joint_pos = next_obs["joints"].astype(np.float32)

                obs_hist.append(eef_pos.copy())
                joint_hist.append(joint_pos.copy())

                next_position.append(eef_pos.copy())
                next_joint_positions.append(joint_pos.copy())

                total_r += float(reward)

                if done:
                    break

            print(f"transition len is: {len(transition_seq)}, self.replaybuffer.seq_len is: {self.replaybuffer.seq_len}")
            print(f"[before save] len(transition_seq)={len(transition_seq)}")
            print(f"transition obs has keys: {transition['obs'].keys()}")

            if len(transition_seq) == self.replaybuffer.seq_len:
                self.replaybuffer.save_sequence(transition_seq)

            goal_positions = np.asarray(goal_positions, dtype=np.float32)
            goal_indices = np.asarray(goal_indices, dtype=np.int32).reshape(-1, 1)
            pred_actions = np.asarray(pred_actions, dtype=np.float32)
            exec_actions = np.asarray(exec_actions, dtype=np.float32)
            controls = np.asarray(controls, dtype=np.float32)
            errors = np.asarray(errors, dtype=np.float32)
            positions = np.asarray(positions, dtype=np.float32)
            joint_positions = np.asarray(joint_positions, dtype=np.float32)
            next_position = np.asarray(next_position, dtype=np.float32)
            next_joint_positions = np.asarray(next_joint_positions, dtype=np.float32)
            rollout_stage_seq = np.asarray(rollout_stage_seq, dtype=np.float32)
            rollout_progress_seq = np.asarray(rollout_progress_seq, dtype=np.float32)

            roll_tag = "heu_rollout" if use_heuristic else "rollout_analysis"
            roll_dir = Path(log_dir) / roll_tag / f"epoch_{self.epoch:04d}"

            noise_stats = self.model.noise_stats_log
            if len(noise_stats) > 0:
                noise_arr = np.array(
                    [[s["mean"], s["std"], s["min"], s["max"]] for s in noise_stats],
                    dtype=np.float32
                )
                self._save_array_stats(noise_arr, roll_dir, prefix="init_noise_stats")

            trace = self.model.diffusion_trace_log
            if len(trace) > 0:
                trace_arr = np.array([
                    [
                        s.get("rollout_step", -1), s["step"], s["timestep"],
                        s["x_mean"], s["x_std"], s["x_min"], s["x_max"],
                        s["x0_mean"], s["x0_std"], s["x0_min"], s["x0_max"]
                    ]
                    for s in trace
                ], dtype=np.float32)
                self._save_trace_by_rollout_step(trace_arr, roll_dir, prefix="diffusion_trace")

            print(f"rollout {rollout_i} complete")
            self._save_array_stats(pred_actions, roll_dir, prefix="pred_actions")
            self._save_array_stats(exec_actions, roll_dir, prefix="exec_actions")
            self._save_array_stats(controls, roll_dir, prefix="env_controls")
            self._save_array_stats(errors, roll_dir, prefix="errors")
            self._save_array_stats(positions, roll_dir, prefix="positions")
            self._save_array_stats(next_position, roll_dir, prefix="next_positions")
            self._save_array_stats(goal_positions, roll_dir, prefix="goal_positions")
            self._save_array_stats(goal_indices.astype(np.float32), roll_dir, prefix="goal_indices")
            self._save_array_stats(rollout_stage_seq.reshape(-1, 1), roll_dir, prefix="stage_seq")
            self._save_array_stats(rollout_progress_seq.reshape(-1, 1), roll_dir, prefix="progress_seq")

            np.savez(
                roll_dir / "norms.npz",
                error_norms=np.asarray(err_norm, dtype=np.float32),
                action_norms=np.asarray(action_norm, dtype=np.float32),
            )

            return {
                "pred_actions": pred_actions,
                "exec_actions": exec_actions,
                "positions": positions,
                "joint_positions": joint_positions,
                "next_position": next_position,
                "next_joint_positions": next_joint_positions,
                "errors": errors,
                "controls": controls,
                "goal_positions": goal_positions,
                "goal_indices": goal_indices,
                "stage_seq": rollout_stage_seq,
                "progress_seq": rollout_progress_seq,
                "demo_pull_dir": demo_pull_dir_np,
                "total_reward": total_r,
                "rollout_dir": roll_dir,
                "transition_seq": transition_seq,
            }

     finally:
        if was_training:
            self.model.train()
    
    def _build_sliding_window_batch(
    self,
    batch: dict[str, torch.Tensor],
    t: Optional[Union[int, list[int], np.ndarray, torch.Tensor]] = None,
    horizon: Optional[int] = None,
):
     """
      Build one sliding window per batch.
      Mixed stages are allowed inside the window.

      Window length is determined by horizon.
     """
     H = int(horizon if horizon is not None else getattr(self.cfg.training, "sliding_horizon", 16))

    # 3-stage split
     reach_end = int(getattr(self.cfg.training, "reach_end_t", 36))
     align_end = int(getattr(self.cfg.training, "align_end_t", 44))

     future_offset = int(getattr(self.cfg.training, "future_offset", H))
     pull_extra_offset = int(getattr(self.cfg.training, "pull_extra_offset", 4))
     pull_start_t_cfg = int(getattr(self.cfg.training, "pull_start_t", align_end))

     if "state" in batch:
        obs = batch["state"]
     else:
        obs = batch["obs"]

     if "next_state" in batch:
        next_obs = batch["next_obs"]
     else:
        next_obs = batch["next_obs"]

     if "joint_pos" not in batch:
        raise KeyError("batch must contain key 'joint_pos'")
     joint_pos = batch["joint_pos"]

     action = batch["action"]
     reward = batch.get("reward", None)
     done = batch.get("done", None)

     B, T, D = obs.shape
     device = obs.device

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

    # -------- sliding window center range --------
     left_ctx = (H - 1) // 2
     right_ctx = H - 1 - left_ctx

     t_min = left_ctx          # sampling window min
     t_max = T - 1 - right_ctx # sampling window max
     if t_min > t_max:
        raise RuntimeError(f"Window horizon too large: H={H}, T={T}")

    # -------- choose one center t --------
     if t is None:
        sample_mode = getattr(self.cfg.training, "sample_mode", "biased_50_30_20")

        if sample_mode == "uniform":
            t_i = int(np.random.randint(t_min, t_max + 1))

        elif sample_mode == "biased_50_30_20":
            # 50%: transition 0->1 mixed
            # 30%: pull region
            # 20%: reach region

            def _randint_closed(lo, hi):
                lo = int(lo)
                hi = int(hi)
                if lo > hi:
                    return None
                return int(np.random.randint(lo, hi + 1))

            reach_lo = t_min
            reach_hi = min(t_max, reach_end - 1)

            pull_lo = max(t_min, align_end)
            pull_hi = t_max

            # window contains both stage 0 and 1
            trans01_lo = max(t_min, reach_end - right_ctx) 
            trans01_hi = min(t_max, reach_end + left_ctx - 1)

            u = np.random.rand()
            if u < 0.50: # 50% sampling transition
                t_candidate = _randint_closed(trans01_lo, trans01_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)
            elif u < 0.80: # 50%<x<80% 30% of pull
                t_candidate = _randint_closed(pull_lo, pull_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)
            else: # rest 20% of reach
                t_candidate = _randint_closed(reach_lo, reach_hi)
                if t_candidate is None:
                    t_candidate = _randint_closed(t_min, t_max)

            t_i = int(t_candidate)

        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

     elif isinstance(t, int):
        t_i = int(t)
     elif isinstance(t, torch.Tensor):
        vals = [int(x) for x in t.detach().cpu().view(-1).tolist()]
        if len(vals) != 1:
            raise ValueError("Sliding window version expects a single t.")
        t_i = vals[0]
     elif isinstance(t, np.ndarray):
        vals = [int(x) for x in t.reshape(-1).tolist()]
        if len(vals) != 1:
            raise ValueError("Sliding window version expects a single t.")
        t_i = vals[0]
     else:
        vals = [int(x) for x in t]
        if len(vals) != 1:
            raise ValueError("Sliding window version expects a single t.")
        t_i = vals[0]

     t_i = int(np.clip(t_i, t_min, t_max))

    # -------- build window index --------
     win_idx = torch.arange(t_i - left_ctx, t_i + right_ctx + 1, device=device)
     assert win_idx.numel() == H

    # -------- data windows --------
     pos_window = pos[:, win_idx, :]                            # [B,H,3]
     ori_window = ori[:, win_idx, :] if ori is not None else None
     joint_pos_window = joint_pos[:, win_idx, :]                # [B,H,Dj]
     act_window = action[:, win_idx, :]                         # [B,H,Da]

     reward_window = reward[:, win_idx, :] if reward is not None else None
     done_window = done[:, win_idx, :] if done is not None else None

    # -------- 3-stage sequence --------
    # 0: reach, 1: align, 2: pull
     stage_ids_1d = torch.zeros(H, device=device, dtype=torch.long)
     stage_ids_1d[(win_idx >= reach_end) & (win_idx < align_end)] = 1
     stage_ids_1d[win_idx >= align_end] = 2

     stage_seq = stage_ids_1d.view(1, H, 1).expand(B, -1, -1).float()   # [B,H,1]

    # -------- normalized progress --------
     progress_seq = (win_idx.float() / max(T - 1, 1)).view(1, H, 1).expand(B, -1, -1)

    # -------- stage-aware goal positions --------
    # reach goal: same-step next_pos
     reach_goal_seq = next_pos[:, win_idx, :]   # [B,H,3]

    # align goal: fixed anchor around align boundary
     align_anchor_idx = min(max(align_end - 1, 0), T - 1)
     align_goal = next_pos[:, align_anchor_idx:align_anchor_idx + 1, :]   # [B,1,3]
     align_goal_seq = align_goal.expand(-1, H, -1)

    # pull goal: future-shifted point
     pull_goal_idx = torch.clamp(win_idx + future_offset + pull_extra_offset, 0, T - 1)
     pull_goal_seq = next_pos[:, pull_goal_idx, :]                        # [B,H,3]

     stage_ids_expand = stage_ids_1d.view(1, H, 1).expand(B, -1, 3)

     stage_goal_pos_seq = torch.where(
        stage_ids_expand == 0,
        reach_goal_seq,
        torch.where(
            stage_ids_expand == 1,
            align_goal_seq,
            pull_goal_seq
        )
    )   # [B,H,3]

    # -------- vector + scalar error --------
     goal_error = stage_goal_pos_seq - pos_window                         # [B,H,3]
     goal_dist2 = (goal_error ** 2).sum(dim=-1, keepdim=True)            # [B,H,1]
     goal_dist = torch.sqrt(goal_dist2 + 1e-6)                           # [B,H,1] length of vector
     goal_dir = goal_error / (goal_dist + 1e-6)                          # [B,H,3] direction of vector

    # -------- global subgoals --------
     pull_start_t = min(max(pull_start_t_cfg, 0), T - 1) # at 44 step
     final_t = T - 1

     subgoal_reach = next_pos[:, pull_start_t:pull_start_t + 1, :]       # [B,1,3] begin of pull
     subgoal_pull = next_pos[:, final_t:final_t + 1, :]                  # [B,1,3]

     subgoal_reach_seq = subgoal_reach.expand(-1, H, -1)                 # [B,H,3]
     subgoal_pull_seq = subgoal_pull.expand(-1, H, -1)                   # [B,H,3]

     subgoal_reach_error = subgoal_reach_seq - pos_window                # [B,H,3]
     subgoal_pull_error = subgoal_pull_seq - pos_window                  # [B,H,3]

     active_subgoal_pos = torch.where(
        stage_seq.expand(-1, -1, 3) < 2.0,
        subgoal_reach_seq,
        subgoal_pull_seq
    )   # [B,H,3]

     active_subgoal_error = torch.where(
        stage_seq < 2.0,
        subgoal_reach_error,
        subgoal_pull_error
    )   # [B,H,3]

    # -------- pull direction from demo trajectory --------
    # use displacement from pull_start_t to final_t
     raw_pull_vec = next_pos[:, final_t, :] - next_pos[:, pull_start_t, :]   # [B,3]
     raw_pull_norm = torch.norm(raw_pull_vec, dim=-1, keepdim=True)          # [B,1]

    # fallback 1: local average delta on pull segment
     if pull_start_t < final_t:
        pull_seg = next_pos[:, pull_start_t:final_t + 1, :]                 # [B,L,3]
        local_deltas = pull_seg[:, 1:, :] - pull_seg[:, :-1, :]             # [B,L-1,3]
        mean_local_delta = local_deltas.mean(dim=1)                         # [B,3]
        mean_local_norm = torch.norm(mean_local_delta, dim=-1, keepdim=True)
     else:
        mean_local_delta = torch.zeros_like(raw_pull_vec)
        mean_local_norm = torch.zeros_like(raw_pull_norm)

     use_local = (raw_pull_norm < 1e-6).float()
     pull_vec = (1.0 - use_local) * raw_pull_vec + use_local * mean_local_delta
     pull_norm = (1.0 - use_local) * raw_pull_norm + use_local * mean_local_norm

    # fallback 2: default direction
     default_pull_dir = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32).view(1, 3).expand(B, -1)
     demo_pull_dir = pull_vec / (pull_norm + 1e-8)                          # [B,3]
     use_default = (pull_norm < 1e-6).float()
     demo_pull_dir = (1.0 - use_default) * demo_pull_dir + use_default * default_pull_dir

     pull_dir_seq = demo_pull_dir.unsqueeze(1).expand(-1, H, -1).clone()    # [B,H,3]
     pull_mask = (stage_seq == 2.0).float()                                  # [B,H,1]
     pull_dir_seq = pull_dir_seq * pull_mask

     cond = {
        "pos": pos_window,
        "ori": ori_window,
        "joint_pos": joint_pos_window,

        "stage": stage_seq,
        "progress": progress_seq,

        "stage_goal_pos": stage_goal_pos_seq,
        "goal_error": goal_error,
        "goal_dist2": goal_dist2,
        "goal_dist": goal_dist,
        "goal_dir": goal_dir,

        "subgoal_reach_pos": subgoal_reach_seq,
        "subgoal_pull_pos": subgoal_pull_seq,
        "subgoal_reach_error": subgoal_reach_error,
        "subgoal_pull_error": subgoal_pull_error,
        "active_subgoal_pos": active_subgoal_pos,
        "active_subgoal_error": active_subgoal_error,

        "pull_dir": pull_dir_seq,
    }

     return {
        "pos_window": pos_window,
        "ori_window": ori_window,
        "joint_pos_window": joint_pos_window,
        "act_window": act_window,
        "reward_window": reward_window,
        "done_window": done_window,

        "stage_goal_pos_seq": stage_goal_pos_seq,
        "goal_error": goal_error,
        "goal_dist2": goal_dist2,
        "goal_dist": goal_dist,
        "goal_dir": goal_dir,

        "subgoal_reach_pos_seq": subgoal_reach_seq,
        "subgoal_pull_pos_seq": subgoal_pull_seq,
        "subgoal_reach_error": subgoal_reach_error,
        "subgoal_pull_error": subgoal_pull_error,
        "active_subgoal_pos": active_subgoal_pos,
        "active_subgoal_error": active_subgoal_error,

        "progress_seq": progress_seq,
        "stage_seq": stage_seq,
        "pull_dir_seq": pull_dir_seq,
        "demo_pull_dir": demo_pull_dir,   # [B,3]

        "cond_t": {
            "cond": cond,
            "obs_idx": win_idx.unsqueeze(0).expand(B, -1),   # [B,H]
            "act_idx": win_idx.unsqueeze(0).expand(B, -1),   # [B,H]
            "t": torch.full((B,), t_i, device=device, dtype=torch.long),
            "stage": stage_seq,
            "batch_size": B,
        }
    }

    
    def _fit_cond_normalizer(
        self,
        training_dataloader: DataLoader,
        max_batches: int = 50,
        max_rows: int = 200000,
        num_t_samples: int = 2,
    ):
        """
        Fit normalization params from raw cond dict returned by _build_sliding_window_batch.
        """
        pos_xs = []
        ori_xs = []
        joint_xs = []
        error_pos_xs = []

        pos_rows = 0
        ori_rows = 0
        joint_rows = 0
        error_pos_rows = 0

        for bi, batch in enumerate(training_dataloader):
            if bi >= max_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            out = self._build_sliding_window_batch(
                batch=batch,
                t=None,
                horizon=int(getattr(self.cfg.training,"horizon",16)),
            )

            cond = out["cond_t"]["cond"]

            x_pos = cond["pos"].reshape(-1, cond["pos"].shape[-1]).detach().cpu()
            take_pos = min(max_rows - pos_rows, x_pos.shape[0])
            if take_pos > 0:
                pos_xs.append(x_pos[:take_pos])
                pos_rows += take_pos

            if cond["ori"] is not None:
                x_ori = cond["ori"].reshape(-1, cond["ori"].shape[-1]).detach().cpu()
                take_ori = min(max_rows - ori_rows, x_ori.shape[0])
                if take_ori > 0:
                    ori_xs.append(x_ori[:take_ori])
                    ori_rows += take_ori

            if cond["joint_pos"] is not None:
                x_joint = cond["joint_pos"].reshape(-1, cond["joint_pos"].shape[-1]).detach().cpu()
                take_joint = min(max_rows - joint_rows, x_joint.shape[0])
                if take_joint > 0:
                    joint_xs.append(x_joint[:take_joint])
                    joint_rows += take_joint

            if "goal_error" in cond and cond["goal_error"] is not None:
                x_error_pos = cond["goal_error"].reshape(-1, cond["goal_error"].shape[-1]).detach().cpu()
                take_error_pos = min(max_rows - error_pos_rows, x_error_pos.shape[0])
                if take_error_pos > 0:
                    error_pos_xs.append(x_error_pos[:take_error_pos])
                    error_pos_rows += take_error_pos

            pos_done = pos_rows >= max_rows
            ori_done = (len(ori_xs) == 0) or (ori_rows >= max_rows)
            joint_done = (len(joint_xs) == 0) or (joint_rows >= max_rows)
            error_done = (len(error_pos_xs) == 0) or (error_pos_rows >= max_rows)

            if pos_done and ori_done and joint_done and error_done:
                break

        if len(pos_xs) == 0:
            raise RuntimeError("No samples collected for pos normalizer fit.")

        pos_cat = torch.cat(pos_xs, dim=0)
        self.pos_norm_params = self.normalizer.fit(pos_cat, mode="gaussian")

        if len(ori_xs) > 0:
            ori_cat = torch.cat(ori_xs, dim=0)
            self.ori_norm_params = self.normalizer.fit(ori_cat, mode="gaussian")
        else:
            self.ori_norm_params = None

        if len(joint_xs) > 0:
            joint_cat = torch.cat(joint_xs, dim=0)
            self.joint_norm_params = self.normalizer.fit(joint_cat, mode="gaussian")
        else:
            self.joint_norm_params = None

        if len(error_pos_xs) > 0:
            error_pos_cat = torch.cat(error_pos_xs, dim=0)
            self.error_pos_norm_params = self.normalizer.fit(error_pos_cat, mode="gaussian")
        else:
            self.error_pos_norm_params = None

        return {
            "pos": self.pos_norm_params,
            "ori": self.ori_norm_params,
            "joint": self.joint_norm_params,
            "pos_error": self.error_pos_norm_params,
        }

    
    
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(tag)s] %(message)s"
        )

        training_dataset: CalvinCriticOfflineDataset
        training_dataset = hydra.utils.instantiate(cfg.datamodule.training_dataset)
        action_norm_params = training_dataset.get_normalize_params("action_params")
        state_norm_params = training_dataset.get_normalize_params("state_params")

        validation_dataset = hydra.utils.instantiate(
            cfg.datamodule.val_dataset,
            action_params=action_norm_params,
            state_params=state_norm_params
        )

        assert isinstance(training_dataset, CalvinCriticOfflineDataset)
        assert isinstance(validation_dataset, CalvinCriticOfflineDataset)

        batch_size = 8
        training_dataloader = DataLoader(
            training_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=True
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )

        print(f"length of training_dataset:{len(training_dataset)}")

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(training_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        self.param_dict = self._fit_cond_normalizer(training_dataloader)

        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        self.model = self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

        self._optimizer_to()
        self._heu_optimizer_to()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent.parent.parent.joinpath(
            "diffusion model offline training log", run_id
        )
        self.writer = SummaryWriter(log_dir=str(log_dir))

        analysis_dir = Path(log_dir) / "action_analysis"
        ds_stats = self._save_dataset_action_stats(training_dataset, analysis_dir)
        print("[saved] dataset action stats ->", analysis_dir)
        print("dataset action std:", ds_stats["std"])

        np.random.seed(cfg.seed + 1234)

        model_path = Path(self.cfg.model_save_dir)
        os.makedirs(model_path, exist_ok=True)

        rollout_every = getattr(self.cfg.training, "rollout_every", 50)
        rollout_env = None
        if rollout_every is not None:
            rollout_env = make_env(cfg=self.cfg.env, skill=self.cfg.skill)

        for epoch_idx in range(cfg.training.num_epochs):
            train_losses = []

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with tqdm.tqdm(
                training_dataloader,
                desc=f"Training Epoch{epoch_idx+1}",
                mininterval=cfg.training.tqdm_interval_sec
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    loss_sum = 0.0
                    window = self._build_sliding_window_batch(batch=batch,horizon=int(getattr(self.cfg.training,"horizon",16)))

                    action = window["act_window"].to(self.device)
                    cond = window["cond_t"]

                    raw_loss = self.model.compute_loss(
                        action,
                        cond,
                        action_norm_params=action_norm_params,
                        cond_norm_params=self.param_dict,
                        improve=False
                    )

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    loss_sum += float(raw_loss.detach().item())
                    raw_loss_cpu = loss_sum / 1

                    is_last_batch = (batch_idx + 1) == len(training_dataloader)
                    do_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every) == 0 or is_last_batch

                    if do_step:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        self.global_step += 1
                        self.writer.add_scalar("training loss", raw_loss_cpu, global_step=self.global_step)
                        self.writer.add_scalar("training lr", lr_scheduler.get_last_lr()[0], self.global_step)

                        train_losses.append(raw_loss_cpu)

                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        if not is_last_batch:
                            logger.info(
                                "step=%d epoch=%d loss=%.6f lr=%.6f",
                                step_log["global_step"],
                                step_log["epoch"],
                                step_log["train_loss"],
                                step_log["lr"],
                                extra={"tag": "Diffusion training"}
                            )

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss
            self.epoch += 1

            self.model.eval()
            if (self.epoch % cfg.training.val_every) == 0:
                self.model.eval()
                with torch.no_grad():
                    t_list = [8, 20, 36, 48, 54]
                    val_losses = []

                    with tqdm.tqdm(
                        validation_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = {k: v.to(self.device) for k, v in batch.items()}

                            for t in t_list:
                                val_window = self._build_sliding_window_batch(batch=batch, t=t,horizon=int(getattr(self.cfg.training,"horizon",16)))
                                val_action = val_window["act_window"].to(self.device)
                                val_cond = val_window["cond_t"]

                                loss = self.model.compute_loss(
                                    val_action,
                                    val_cond,
                                    action_norm_params=action_norm_params,
                                    cond_norm_params=self.param_dict,
                                    improve=False,
                                )
                                val_losses.append(loss.item())

                            if (cfg.training.max_val_steps is not None
                                    and batch_idx >= (cfg.training.max_val_steps - 1)):
                                break

                    if len(val_losses) > 0:
                        val_loss = float(np.mean(val_losses))
                        self.writer.add_scalar("validation loss", val_loss, global_step=self.global_step)
                        step_log["val_loss"] = val_loss

                self.model.train()

            if (self.epoch % cfg.training.save_every_epoch) == 0:
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "lr_scheduler": lr_scheduler.state_dict()
                }
                if cfg.training.use_ema:
                    ckpt["ema_model"] = self.ema_model.state_dict()

                torch.save(ckpt, model_path / f"epoch_{self.epoch:04d}.pt")

            rollout_every = getattr(self.cfg.training, "rollout_every", 50)
            rollout_after = getattr(self.cfg.training, "rollout_after", 50)
            if self.epoch > rollout_after:
                rollout_every = 1

            rollout_result = None
            if (rollout_every is not None) and (self.epoch % rollout_every == 0):
                print("training rollout...")
                rollout_result = self._rollout(
                    rollout_env=rollout_env,
                    training_dataset=training_dataset,
                    action_norm_params=action_norm_params,
                    log_dir=log_dir,
                    use_heuristic=False,
                    heuristic_net=None
                )

            print("training rollout end")
            print(f"epoch_idx: {epoch_idx}")

            if ((rollout_result is not None)
                    and epoch_idx > 0
                    and (epoch_idx % self.cfg.heuristic_explore) == 0):
                self.model.eval()
                self.heuristic_net.train()

                prob_list_tensor = get_soft_label(
                    B=1,
                    x=np.arange(0, self.replaybuffer.seq_len, 1),
                    label=1,
                    cfg=self.cfg.target_selector
                ).to(self.device)

                for heu_epoch_num in range(self.cfg.training.heu_epoch_num):
                    with tqdm.tqdm(total=100, desc="training heuristic mask", mininterval=1.0) as heu_epoch:
                        for step in range(100):
                            batch = self.replaybuffer.load_sequence("state-action", batch_size=1).to(self.device)

                            self.heu_optimizer.zero_grad()
                            pred_label = self.heuristic_net(batch)
                            heu_train_loss = F.binary_cross_entropy(pred_label, prob_list_tensor)
                            heu_train_loss.backward()
                            self.heu_optimizer.step()
                            heu_epoch.update(1)

                            step_log = {
                                "loss": heu_train_loss,
                                "step": step,
                                "epoch": heu_epoch_num
                            }
                            logger.info(
                                "step=%d epoch=%d loss=%.6f",
                                step_log["step"],
                                step_log["epoch"],
                                float(step_log["loss"].item()),
                                extra={"tag": "Heu_net_train"}
                            )

                    if heu_epoch_num % self.cfg.training.heu_val_every == 0:
                        self.heuristic_net.eval()
                        with torch.no_grad():
                            with tqdm.tqdm(total=10, desc="validation heuristic mask", mininterval=1.0) as heu_val_epoch:
                                for step in range(10):
                                    val_batch = self.replaybuffer.load_sequence("state-action", batch_size=1).to(self.device)

                                    pred_label = self.heuristic_net(val_batch)
                                    heu_val_loss = F.binary_cross_entropy(pred_label, prob_list_tensor)
                                    heu_val_epoch.update(1)
                                    heu_val_epoch.set_postfix(val_loss=float(heu_val_loss.item()))
                        self.heuristic_net.train()

                heu_rollout_result = self._rollout(
                    rollout_env=rollout_env,
                    training_dataset=training_dataset,
                    action_norm_params=action_norm_params,
                    log_dir=log_dir,
                    use_heuristic=True,
                    heuristic_net=self.heuristic_net
                )

                logger.info(
                    "base_rollout_reward=%.6f heu_rollout_reward=%.6f",
                    float(rollout_result["total_reward"]),
                    float(heu_rollout_result["total_reward"]),
                    extra={"tag": "Heu_exploring"}
                )

                self.model.train()

        if rollout_env is not None:
            rollout_env.close()
            del rollout_env


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent.joinpath("config")),
    config_name="heuristic_ddpm_training",
    version_base=None
)
def main(cfg):
    workspace = TrainLowDimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()