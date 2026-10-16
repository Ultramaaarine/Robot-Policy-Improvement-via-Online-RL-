# Hybrid diffusion policy for CALVIN control.
# It assembles normalized multimodal conditions, trains the diffusion model,
# supports stage-aware loss and advantage weighting, and samples actions with DDPM.

# cond: total 20 dim: pos:3, joints: 7, subgoal_active_error: 3, pull_dir: 3, progress: 1, goal: 3 
# is a torch.tensor 
from typing import Union, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_diffusion.utils.target_selector import TargetSelector
from sac_diffusion.policy.base_policy import BasePolicy
from sac_diffusion.models.transformer_for_diffusion import DiffusionTransformer
from sac_diffusion.models.normalizer import Normalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class HybridPolicy(BasePolicy):
    def __init__(
        self,
        model: DiffusionTransformer, # encoder accept visual obs
        action_horizon: int,
        num_sampling_steps: int,
        training_timesteps: int,
        beta_start,
        beta_end,
        beta_schedule: str,
        prediction_type: str,
        mode: str,
        error_scale: float,
        lambda_stage: float,
        pull_start_t: int = 44,
        
    ):
        super().__init__()

        # Diffusion training and sampling configuration.
        self.training_timesteps = training_timesteps
        self.num_sampling_steps = num_sampling_steps

        # Transformer-based diffusion model.
        self.model = model

        # Policy mode and action horizon.
        self.mode = mode
        self.action_horizon = action_horizon

        # Normalizer used for action and condition features.
        self.normalizer = Normalizer()

        # DDPM scheduler used for forward noising and reverse denoising.
        self.scheduler = DDPMScheduler(
            self.training_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # Logs used for rollout and diffusion debugging.
        self.noise_stats_log = []
        self.diffusion_trace_log = []
        self.action_trace_log = []

        # Scale applied to error-related condition features.
        self.error_scale = error_scale

        # Timestep marking the start of the pull phase.
        self.pull_start_t = pull_start_t

        # Weight of the auxiliary stage classification loss.
        self.lambda_stage = lambda_stage

        # Offline target selector used for task-specific references.
        self.target_selector = TargetSelector("calvin_open_drawer", sort_by="t")
        self.target_selector.scan()


    def set_normalizer(self, normalizer):

        # Replace the default normalizer with an externally provided one.
        self.normalizer = normalizer
        return self.normalizer


    def _assemble_cond(
    self,
    cond_t: dict,
    cond_norm_params: dict,
    batch_size: Optional[int],
):
     # Validate condition inputs.
     assert isinstance(cond_t, dict)
     assert isinstance(cond_norm_params, dict)

     # Collect low-dimensional condition features before concatenation.
     xs = []

     # cond_t may directly contain features or store them under "cond".
     if "cond" in cond_t:
        cond = cond_t["cond"]
     else:
        cond = cond_t

     # Infer batch size if it is not explicitly provided.
     if batch_size is None:
        batch_size = cond_t.get("batch_size", None)
     if batch_size is None:
        batch_size = cond_t.get("B", None)

     # Future action indices used for stage-dependent action weighting.
     action_idx = cond_t.get("act_idx", None)

    # -------------------------------------------------
    # visual cond
    # -------------------------------------------------

     # Optional visual observation condition.
     visual_cond = None

     if "rgb_gripper" in cond and cond["rgb_gripper"] is not None:
        visual_cond = cond["rgb_gripper"].to(self.device, dtype=torch.float32)

        # Convert uint8-style RGB values into approximately [0,1].
        if visual_cond.numel() > 0 and visual_cond.max() > 1.5:
            visual_cond = visual_cond / 255.0

    # -------------------------------------------------
    # build action mask
    # -------------------------------------------------
      
     # Build time-dependent action weights if action indices are available.
     if action_idx is not None:
        BK = action_idx.shape[0]
        Ta = action_idx.shape[-1]

        if batch_size is None:
            batch_size = BK

        # Flattened candidate batch must be divisible by original batch size.
        if BK % batch_size != 0:
            raise ValueError(
                f"_assemble_cond mismatch: action_idx.shape={action_idx.shape}, "
                f"batch_size={batch_size}"
            )

        # Recover number of candidate trajectories.
        K = BK // batch_size

        # [B*K, Ta] -> [K, B, Ta]
        action_idx = action_idx.reshape(K, batch_size, Ta).to(self.device)  # [K,B,Ta]

        # Key timestep marking the beginning of the pull phase.
        key_t = torch.tensor(
            float(self.pull_start_t),
            dtype=torch.float32,
            device=self.device
        )

        key_t_expand = key_t.view(1, 1, 1).expand(K, batch_size, Ta)

        # Identify action steps after pull_start_t.
        post_key = action_idx.float() >= key_t_expand

        # Relative temporal distance from pull_start_t.
        rel_dist = (action_idx.float() - key_t_expand).clamp(min=0.0)
        rel_dist = rel_dist / max(Ta - 1, 1)

        # Exponential weighting parameters.
        base = 1.0
        beta = 2.0

        # Default action weight is one.
        action_mask = torch.ones_like(
            action_idx, dtype=torch.float32, device=self.device
        )

        # Increase action weights during the pull phase.
        action_mask[post_key] = base * torch.exp(beta * rel_dist[post_key])

        # Prevent excessively large weights.
        action_mask = torch.clamp(action_mask, max=15.0)

        # Restore flattened candidate-batch representation.
        action_mask = action_mask.reshape(-1, Ta).to(
            self.device, dtype=torch.float32
        )  # [B*K,Ta]

     else:
        action_mask = None

    # get boolean mask for pseudo actions
     # Retrieve masks used to ignore padded observation/action tokens.
     if "action_padding_mask" in cond:
      
        boolean_action_mask = cond["action_padding_mask"]

     if "obs_padding_mask" in cond:
        boolean_obs_mask = cond["obs_padding_mask"]

     if "action_loss_mask" in cond:
        action_loss_mask = cond["action_loss_mask"]

     else:
        # Use any available condition tensor to infer fallback mask shape.
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

        # Sequence condition.
        if ref.dim() == 3:
            action_mask = torch.ones(
                ref.shape[0], ref.shape[1],
                device=self.device,
                dtype=torch.float32
            )

        # Single-step condition.
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
    # normalize / assemble low-dim cond features
    # -------------------------------------------------

     # Reference feature used to determine temporal shape for stage labels.
     ref = None

     # Prefer goal_error if available.
     if "goal_error" in cond and cond["goal_error"] is not None:
        goal_error = cond["goal_error"]
        
        n_goal_error = self.normalizer.normalize(
            goal_error, cond_norm_params["pos_error"]
        ).to(self.device, dtype=torch.float32)

        # Error features are additionally scaled.
        xs.append(self.error_scale * n_goal_error)
        ref = goal_error

     # Otherwise use pos_error.
     elif "pos_error" in cond and cond["pos_error"] is not None:
        pos_error = cond["pos_error"]

        n_error = self.normalizer.normalize(
            pos_error, cond_norm_params["pos_error"]
        ).to(self.device, dtype=torch.float32)

        xs.append(self.error_scale * n_error)
        ref = pos_error

     # Normalize Cartesian position condition.
     if "pos" in cond and cond["pos"] is not None:
        pos = cond["pos"]

        n_pos = self.normalizer.normalize(
            pos, cond_norm_params["pos"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_pos)

        if ref is None:
            ref = pos

     # Normalize orientation condition.
     if "ori" in cond and cond["ori"] is not None:
        ori = cond["ori"]

        n_ori = self.normalizer.normalize(
            ori, cond_norm_params["ori"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_ori)

        if ref is None:
            ref = ori

     # Normalize joint-position condition.
     if "joint_pos" in cond and cond["joint_pos"] is not None:
        joint = cond["joint_pos"]

        n_joint = self.normalizer.normalize(
            joint, cond_norm_params["joint"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_joint)

        if ref is None:
            ref = joint

     # Normalize stage-dependent active-subgoal error.
     if "active_subgoal_error" in cond and cond["active_subgoal_error"] is not None:
        active_err = cond["active_subgoal_error"]

        n_active_err = self.normalizer.normalize(
            active_err, cond_norm_params["active_subgoal_error"]
        ).to(self.device, dtype=torch.float32)

        xs.append(self.error_scale * n_active_err)

        if ref is None:
            ref = active_err

    #  if "pull_dir" in cond and cond["pull_dir"] is not None:
    #     pull_dir = cond["pull_dir"].to(self.device, dtype=torch.float32)
    #     xs.append(pull_dir)
    #     if ref is None:
    #         ref = pull_dir

     # Add normalized rollout progress if enabled.
     if "progress" in cond and cond["progress"] is not None:
        progress = cond["progress"].to(self.device, dtype=torch.float32)

        xs.append(progress)

        if ref is None:
            ref = progress

    # -------------------------------------------------
    # stage feature
    # -------------------------------------------------
    # -------------------------------------------------
    # stage feature (keep separate, do NOT concat into n_cond)
    # -------------------------------------------------

     # Stage labels remain separate because they are embedded later by the model.
     stage = cond.get("stage", None)
     stage_out = None

     if stage is not None:
        stage = stage.to(self.device)

        # A reference sequence is required to determine stage temporal length.
        if ref is None:
            raise ValueError("No valid cond tensor found to infer stage feature shape.")

        if ref.dim() != 3:
            raise ValueError(f"Expected ref dim=3 for stage broadcast, got {ref.shape}")

        Tref = ref.shape[1]

        # Accept [B,T,1] or [B,T].
        if stage.dim() == 3 and stage.shape[-1] == 1:
            stage_out = stage

        elif stage.dim() == 2:
            stage_out = stage.unsqueeze(-1)

        else:
            raise ValueError(f"Unexpected stage shape: {stage.shape}")

        # Broadcast a single stage value over the whole temporal window.
        if stage_out.shape[1] == 1:
            stage_out = stage_out.expand(-1, Tref, -1)

        if stage_out.shape[1] != Tref:
            raise ValueError(
                f"stage shape {stage_out.shape} incompatible with ref shape {ref.shape}"
            )

        # keep integer-like stage labels for embedding later
        stage_out = stage_out.to(self.device, dtype=torch.long)

     # At least one low-dimensional condition feature must be available.
     if len(xs) == 0:
        raise ValueError("No valid entries found in cond to assemble.")

     # Concatenate normalized condition features along the feature dimension.
     xs = [x.to(self.device, dtype=torch.float32) for x in xs]
     n_cond = torch.cat(xs, dim=-1).to(self.device, dtype=torch.float32)

     # Return assembled low-dimensional condition, visual condition,
     # padding masks, action weighting masks, and stage labels.
     return n_cond, visual_cond, boolean_obs_mask, action_mask, action_loss_mask, boolean_action_mask, stage_out # action mask is designed for penalize error in pull phase, for pull phase is only a minor part of entire seq,
                                                                             # boolean action mask is designed for eliminate loss in action padding


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
     # Ensure action sequence is non-empty.
     assert len(action[0]) != 0

     model = self.model
     device = action.device if isinstance(action, torch.Tensor) else self.device

     # Normalize the target action sequence.
     n_action = self.normalizer.normalize(
        action, action_norm_params
    ).to(device=device, dtype=torch.float32)

     # Assemble normalized condition features and masks.
     n_cond, visual_obs, boolean_obs_mask, action_mask, action_loss_mask,  boolean_action_mask, stage = self._assemble_cond(
        cond_t=cond,
        cond_norm_params=cond_norm_params,
        batch_size=batch_size
    )

     # Move condition and masks to the correct device.
     n_cond = n_cond.to(device=device, dtype=torch.float32)
     action_mask = action_mask.to(device=device, dtype=torch.float32)
     boolean_action_mask = boolean_action_mask.to(self.device,dtype = torch.int32)
     boolean_obs_mask = boolean_obs_mask.to(self.device,dtype = torch.int32)
     action_loss_mask = action_loss_mask.to(self.device,dtype = torch.float32)
     
     if visual_obs is not None:
        visual_obs = visual_obs.to(device=device, dtype=torch.float32)

     # Sample Gaussian diffusion noise.
     noise = torch.randn_like(n_action, device=device)

     # Randomly sample one diffusion timestep for each batch element.
     training_timesteps = torch.randint(
        0, self.training_timesteps, (n_action.shape[0],), device=device
    ).long()

     # Forward diffusion: add noise to normalized action.
     noisy_action = self.scheduler.add_noise(n_action, noise, training_timesteps)

     #print(f"visual_obs has shape: {visual_obs.shape}") # 8,64,84,84,3 need permute

     # Predict denoising target and optional stage logits.
     pred ,stage_logits = model.forward(
        x = noisy_action,
        diffusion_t=training_timesteps,
        visual_obs=visual_obs,
        low_dim_obs=n_cond,
        stage = stage,
        action_padding_mask=boolean_action_mask,
        cond_padding_mask = boolean_obs_mask,
        return_stage_logits=True
    )

     # Standard diffusion objective depending on scheduler prediction type.
     if self.scheduler.config.prediction_type == "epsilon":
        per_elem_loss = F.mse_loss(pred, noise, reduction="none")

     elif self.scheduler.config.prediction_type == "sample":
        per_elem_loss = F.mse_loss(pred, n_action, reduction="none")

     else:
        raise ValueError(
            f"Unsupported prediction_type: {self.scheduler.config.prediction_type}"
        )

     # Ignore padded action tokens in diffusion loss.
     weight =  action_loss_mask.unsqueeze(-1)#* action_mask.unsqueeze(-1)

     # Auxiliary stage-classification loss.
     stage_loss = torch.tensor(0.0, device=device)

     if stage is not None:
        stage_target = stage

        if stage_target.dim() == 3 and stage_target.shape[-1] == 1:
            stage_target = stage_target[..., 0]   # [B,T]

        stage_target = stage_target.long()

        # Use valid action positions for stage supervision.
        valid_mask = (action_mask > 0).float()

        stage_loss = F.cross_entropy(
            stage_logits.view(-1,self.model.num_stages),
            stage_target.view(-1),
            reduction='none'
        ) # [B,T]

        # Mean stage classification loss over valid timesteps.
        stage_loss = (stage_loss*valid_mask.view(-1)).sum()/(valid_mask.sum()+1e-6)

     # During policy improvement, multiply diffusion loss by
     # externally provided advantage weights.
     if improve:
        assert w is not None, "w must be provided when improve=True"

        # Stop gradients through the weighting term.
        w = w.detach().to(device=per_elem_loss.device, dtype=per_elem_loss.dtype)

        # Convert [B,T] -> [B,T,1].
        if w.dim() == 2:
            w = w.unsqueeze(-1)

        elif w.dim() != 3:
            raise ValueError(f"Unexpected w shape: {w.shape}")

        # Weight shape must align with temporal diffusion loss.
        if w.shape[0] != per_elem_loss.shape[0] or w.shape[1] != per_elem_loss.shape[1]:
            raise ValueError(
                f"w shape {w.shape} incompatible with per_elem_loss shape {per_elem_loss.shape}"
            )

        weight = weight * w

     # Weighted mean diffusion loss over valid elements.
     loss = (per_elem_loss * weight).sum() / (weight.sum() * pred.shape[-1] + 1e-6)

     # Add auxiliary stage loss.
     loss = loss + self.lambda_stage*stage_loss

     return loss

    
    @torch.no_grad()
    def sampling_action_from_cond(
    self,
    cond_t: dict,
    action_mode,
    cond_norm_params: dict,
    rollout_step,
    batch_size=None
):
     # Determine action dimension from control mode.
     if action_mode == "pos":
        Da = 3

     elif action_mode == "pos_ori":
        Da = 6

     elif action_mode == "pos_ori_gripper":
        Da = 7

     else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

     generator = None

     # Assemble normalized rollout conditions.
     n_cond, visual_obs, boolean_obs_mask, action_mask, action_loss_mask,  boolean_action_mask, stage = self._assemble_cond( 
        cond_t=cond_t,
        cond_norm_params=cond_norm_params,
        batch_size=batch_size
    ) # in sampling, it is unnecessary to build an action mask, other values in cond_t is unnecessary

     # Extract raw condition dictionary.
     if "cond" in cond_t:
        cond_dict = cond_t["cond"]
     else:
        cond_dict = cond_t

     # Position condition is required to infer rollout batch and temporal size.
     if "pos" not in cond_dict or cond_dict["pos"] is None:
        raise ValueError("cond must contain cond['pos'] for sampling")

     pos_ref = cond_dict["pos"]

     if pos_ref.dim() != 3:
        raise ValueError(f"Expected cond['pos'] dim=3, got {pos_ref.shape}")
     
     model = self.model

     # Batch size and temporal horizon used for diffusion sampling.
     B = pos_ref.shape[0]
     T = pos_ref.shape[1]
     device = n_cond.device

     if visual_obs is not None:
        visual_obs = visual_obs.to(device=device, dtype=torch.float32)
        

     # Initialize reverse diffusion from Gaussian noise.
     action = torch.randn(
        size=[B, T, Da],
        dtype=torch.float32,
        device=device,
        generator=generator
    )

     # Record init
