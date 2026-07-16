from typing import Union, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_diffusion.utils.target_selector import TargetSelector
from sac_diffusion.policy.base_policy import BasePolicy
from sac_diffusion.models.Diffusion_Unet import Lowdim_Unet
from sac_diffusion.models.normalizer import Normalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class LowdimPolicy(BasePolicy):
    def __init__(
        self,
        model: Lowdim_Unet,
        action_horizon: int,
        num_sampling_steps: int,
        training_timesteps: int,
        beta_start,
        beta_end,
        beta_schedule: str,
        prediction_type: str,
        mode: str,
        error_scale: float,
        pull_start_t: int = 44,
    ):
        super().__init__()
        self.training_timesteps = training_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.model = model
        self.mode = mode
        self.action_horizon = action_horizon
        self.normalizer = Normalizer()

        self.scheduler = DDPMScheduler(
            self.training_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        self.noise_stats_log = []
        self.diffusion_trace_log = []
        self.action_trace_log = []
        self.error_scale = error_scale
        self.pull_start_t = pull_start_t

        self.target_selector = TargetSelector("calvin_open_drawer", sort_by="t")
        self.target_selector.scan()

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        return self.normalizer

    def _assemble_cond(
        self,
        cond_t: dict,
        cond_norm_params: dict,
        batch_size: Optional[int],
        
    ):
        assert isinstance(cond_t, dict)
        assert isinstance(cond_norm_params, dict)

        xs = []

        if "cond" in cond_t:
            cond = cond_t["cond"]
        else:
            cond = cond_t

        if batch_size is None:
            batch_size = cond_t.get("batch_size", None)
        if batch_size is None:
            batch_size = cond_t.get("B", None)

        action_idx = cond_t.get("act_idx", None)

        # -------------------------------------------------
        # build action mask
        # -------------------------------------------------
        if action_idx is not None:
            BK = action_idx.shape[0]
            Ta = action_idx.shape[-1]

            if batch_size is None:
                batch_size = BK

            if BK % batch_size != 0:
                raise ValueError(
                    f"_assemble_cond mismatch: action_idx.shape={action_idx.shape}, "
                    f"batch_size={batch_size}"
                )

            K = BK // batch_size
            action_idx = action_idx.reshape(K, batch_size, Ta).to(self.device)  # [K,B,Ta]

            key_t = torch.tensor(
                float(self.pull_start_t),
                dtype=torch.float32,
                device=self.device
            )

            key_t_expand = key_t.view(1, 1, 1).expand(K, batch_size, Ta)

            post_key = action_idx.float() >= key_t_expand
            rel_dist = (action_idx.float() - key_t_expand).clamp(min=0.0)
            rel_dist = rel_dist / max(Ta - 1, 1)

            base = 1.75
            beta = 3.0

            action_mask = torch.ones_like(
                action_idx, dtype=torch.float32, device=self.device
            )
            action_mask[post_key] = base * torch.exp(beta * rel_dist[post_key])
            action_mask = torch.clamp(action_mask, max=15.0)

            action_mask = action_mask.reshape(-1, Ta).to(
                self.device, dtype=torch.float32
            )  # [B*K,Ta]

        else:
            # inference / rollout fallback
            ref = None
            for k in [
                "goal_error",
                "pos_error",
                "pos",
                "ori",
                "joint_pos",
                "active_subgoal_error",
                "pull_dir",
                "progress",
            ]:
                if k in cond and cond[k] is not None:
                    ref = cond[k]
                    break

            if ref is None:
                raise ValueError(
                    "No valid cond tensor found to infer fallback action_mask shape."
                )

            if ref.dim() == 3:
                action_mask = torch.ones(
                    ref.shape[0], ref.shape[1],
                    device=self.device,
                    dtype=torch.float32
                )
            elif ref.dim() == 2:
                action_mask = torch.ones(
                    ref.shape[0], 1,
                    device=self.device,
                    dtype=torch.float32
                )
            else:
                raise ValueError(
                    f"Unexpected cond tensor shape for fallback mask: {ref.shape}"
                )

        # -------------------------------------------------
        # normalize / assemble cond features
        # -------------------------------------------------
        ref = None

        # prefer new goal_error; fallback to old pos_error
        if "goal_error" in cond and cond["goal_error"] is not None:
            goal_error = cond["goal_error"]
            n_goal_error = self.normalizer.normalize(
                goal_error, cond_norm_params["pos_error"]
            ).to(self.device, dtype=torch.float32)
            xs.append(self.error_scale * n_goal_error)
            ref = goal_error

        elif "pos_error" in cond and cond["pos_error"] is not None:
            pos_error = cond["pos_error"]
            n_error = self.normalizer.normalize(
                pos_error, cond_norm_params["pos_error"]
            ).to(self.device, dtype=torch.float32)
            xs.append(self.error_scale * n_error)
            ref = pos_error

        if "pos" in cond and cond["pos"] is not None:
            pos = cond["pos"]
            n_pos = self.normalizer.normalize(
                pos, cond_norm_params["pos"]
            ).to(self.device, dtype=torch.float32)
            xs.append(n_pos)
            if ref is None:
                ref = pos

        if "ori" in cond and cond["ori"] is not None:
            ori = cond["ori"]
            n_ori = self.normalizer.normalize(
                ori, cond_norm_params["ori"]
            ).to(self.device, dtype=torch.float32)
            xs.append(n_ori)
            if ref is None:
                ref = ori

        if "joint_pos" in cond and cond["joint_pos"] is not None:
            joint = cond["joint_pos"]
            n_joint = self.normalizer.normalize(
                joint, cond_norm_params["joint"]
            ).to(self.device, dtype=torch.float32)
            xs.append(n_joint)
            if ref is None:
                ref = joint

        # active subgoal error
        if "active_subgoal_error" in cond and cond["active_subgoal_error"] is not None:
            active_err = cond["active_subgoal_error"]
            n_active_err = self.normalizer.normalize(
                active_err, cond_norm_params["pos_error"]
            ).to(self.device, dtype=torch.float32)
            xs.append(self.error_scale * n_active_err)
            if ref is None:
                ref = active_err

        # pull_dir: already a normalized direction-like signal, do not gaussian-normalize
        if "pull_dir" in cond and cond["pull_dir"] is not None:
            pull_dir = cond["pull_dir"].to(self.device, dtype=torch.float32)
            xs.append(pull_dir)
            if ref is None:
                ref = pull_dir

        # progress: scalar in [0,1], keep raw
        if "progress" in cond and cond["progress"] is not None:
            progress = cond["progress"].to(self.device, dtype=torch.float32)
            xs.append(progress)
            if ref is None:
                ref = progress

        # -------------------------------------------------
        # stage feature
        # -------------------------------------------------
        stage = cond.get("stage", None)
        stage_feat = None

        if stage is not None:
            stage = stage.to(self.device, dtype=torch.float32)

            if ref is None:
                raise ValueError("No valid cond tensor found to infer stage feature shape.")

            if ref.dim() != 3:
                raise ValueError(f"Expected ref dim=3 for stage broadcast, got {ref.shape}")

            # stage expected [B,1] or [B,T,1]
            if stage.dim() == 2:
                stage_feat = stage.unsqueeze(1).expand(-1, ref.shape[1], -1)
            elif stage.dim() == 3:
                if stage.shape[1] == 1:
                    stage_feat = stage.expand(-1, ref.shape[1], -1)
                else:
                    if stage.shape[1] != ref.shape[1]:
                        raise ValueError(
                            f"stage shape {stage.shape} incompatible with ref shape {ref.shape}"
                        )
                    stage_feat = stage
            else:
                raise ValueError(f"Unexpected stage shape: {stage.shape}")

            xs.append(stage_feat) # append stage

        if len(xs) == 0:
            raise ValueError("No valid entries found in cond to assemble.")

        xs = [x.to(self.device, dtype=torch.float32) for x in xs]
        n_cond = torch.cat(xs, dim=-1).to(self.device, dtype=torch.float32)

        return n_cond, action_mask, stage

    def compute_loss(
        self,
        action: Union[np.ndarray, torch.Tensor],
        cond: Union[Dict, torch.Tensor],
        action_norm_params,
        cond_norm_params: dict,
        improve: bool,
        
        batch_size: Optional[int] = None,
        w: Optional[torch.Tensor] = None
    ):
        assert len(action[0]) != 0

        model = self.model
        device = action.device if isinstance(action, torch.Tensor) else self.device

        n_action = self.normalizer.normalize(
            action, action_norm_params
        ).to(device=device, dtype=torch.float32)

        n_cond, action_mask, stage = self._assemble_cond(
            cond_t=cond,
            cond_norm_params=cond_norm_params,
            
            batch_size=batch_size
        )

        n_cond = n_cond.to(device=device, dtype=torch.float32)
        action_mask = action_mask.to(device=device, dtype=torch.float32)

        noise = torch.randn_like(n_action, device=device)
        training_timesteps = torch.randint(
            0, self.training_timesteps, (n_action.shape[0],), device=device
        ).long()

        noisy_action = self.scheduler.add_noise(n_action, noise, training_timesteps)

        pred = model.forward(
            noisy_action,
            training_timesteps,
            visual_obs=None,
            low_dim_obs=n_cond,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            per_elem_loss = F.mse_loss(pred, noise, reduction="none")   # [B,T,D]
        elif self.scheduler.config.prediction_type == "sample":
            per_elem_loss = F.mse_loss(pred, n_action, reduction="none")   # [B,T,D]
        else:
            raise ValueError(
                f"Unsupported prediction_type: {self.scheduler.config.prediction_type}"
            )

        weight = action_mask.unsqueeze(-1)   # [B,T,1]

        if improve:
            assert w is not None, "w must be provided when improve=True"
            w = w.detach().to(device=per_elem_loss.device, dtype=per_elem_loss.dtype)

            if w.dim() == 2:
                w = w.unsqueeze(-1)   # [B,T] -> [B,T,1]
            elif w.dim() != 3:
                raise ValueError(f"Unexpected w shape: {w.shape}")

            if w.shape[0] != per_elem_loss.shape[0] or w.shape[1] != per_elem_loss.shape[1]:
                raise ValueError(
                    f"w shape {w.shape} incompatible with per_elem_loss shape {per_elem_loss.shape}"
                )

            weight = weight * w

        loss = (per_elem_loss * weight).sum() / (weight.sum() * pred.shape[-1] + 1e-6)
        return loss

    @torch.no_grad()
    def sampling_action_from_cond(
        self,
        cond: dict,
        action_mode,
        cond_norm_params: dict,
        rollout_step,
        
        batch_size=None
    ):
        if action_mode == "pos":
            Da = 3
        elif action_mode == "pos_ori":
            Da = 6
        elif action_mode == "pos_ori_gripper":
            Da = 7
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        generator = None

        n_cond, action_mask, stage = self._assemble_cond(
            cond_t=cond,
            cond_norm_params=cond_norm_params,
            
            batch_size=batch_size
        )

        if "cond" in cond:
            cond_dict = cond["cond"]
        else:
            cond_dict = cond

        if "pos" not in cond_dict or cond_dict["pos"] is None:
            raise ValueError("cond must contain cond['pos'] for sampling")

        pos_ref = cond_dict["pos"]

        if pos_ref.dim() != 3:
            raise ValueError(f"Expected cond['pos'] dim=3, got {pos_ref.shape}")

        model = self.model
        B = pos_ref.shape[0]
        T = pos_ref.shape[1]
        device = n_cond.device

        action = torch.randn(
            size=[B, T, Da],
            dtype=torch.float32,
            device=device,
            generator=generator
        )

        noise_stats = {
            "mean": action.mean().item(),
            "std": action.std().item(),
            "min": action.min().item(),
            "max": action.max().item()
        }
        self.noise_stats_log.append(noise_stats)

        self.scheduler.set_timesteps(self.num_sampling_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            model_output = model.forward(
                action,
                t,
                visual_obs=None,
                low_dim_obs=n_cond,
            )

            out = self.scheduler.step(
                model_output,
                t,
                action,
                generator=generator,
            )
            action = out.prev_sample
            x0 = out.pred_original_sample

            if i % 5 == 0:
                self.diffusion_trace_log.append({
                    "rollout_step": int(rollout_step) if rollout_step is not None else -1,
                    "step": int(i),
                    "timestep": int(t),
                    "x_mean": action.mean().item(),
                    "x_std": action.std().item(),
                    "x_min": action.min().item(),
                    "x_max": action.max().item(),
                    "x0_mean": float(x0.mean().item()),
                    "x0_std": float(x0.std().item()),
                    "x0_min": float(x0.min().item()),
                    "x0_max": float(x0.max().item())
                })

        return action

    def predict_action(
        self,
        cond: dict,
        action_mode: str,
        action_params,
        cond_norm_params,
        rollout_step,
        batch_size=None,
        
    ) -> torch.Tensor:
        n_predicted_action = self.sampling_action_from_cond(
            cond,
            action_mode,
            cond_norm_params,        
            rollout_step=rollout_step,
            batch_size=batch_size,
            
        )

        predicted_action = self.normalizer.unnormalize(
            n_predicted_action, action_params
        )

        i = int(rollout_step) if rollout_step is not None else -1
        t = -1

        self.action_trace_log.append({
            "i": i,
            "t": t,
            "a_n_mean": float(n_predicted_action.mean().item()),
            "a_n_std": float(n_predicted_action.std().item()),
            "a_n_min": float(n_predicted_action.min().item()),
            "a_n_max": float(n_predicted_action.max().item()),
            "a_mean": float(predicted_action.mean().item()),
            "a_std": float(predicted_action.std().item()),
            "a_min": float(predicted_action.min().item()),
            "a_max": float(predicted_action.max().item()),
        })

        return predicted_action