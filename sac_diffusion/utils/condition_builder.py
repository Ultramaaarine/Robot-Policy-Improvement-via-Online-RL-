# cond: 20 dim, pos: 3, joint: 7, goal_error: 3 active_subgoal_error: 3, stage: 1, pull_dir: 3
import torch
from typing import Optional
from sac_diffusion.models.normalizer import Normalizer


def assemble_conditions(
    cond_t: dict,
    cond_norm_params: dict,
    batch_size: Optional[int],
    device,
    error_scale: float,
    pull_start_t: int,
    
    build_action_mask: bool,
):
    assert isinstance(cond_t, dict)
    assert isinstance(cond_norm_params, dict)
    normalizer = Normalizer()
    cond = cond_t["cond"] if "cond" in cond_t else cond_t

    if batch_size is None:
        batch_size = cond_t.get("batch_size", None)
    if batch_size is None:
        batch_size = cond_t.get("B", None)

    action_idx = cond_t.get("act_idx", None)

    # -------------------------------------------------
    # visual cond (current only for now)
    # -------------------------------------------------
    visual_cond = None
    if "rgb_gripper" in cond and cond["rgb_gripper"] is not None:
        visual_cond = cond["rgb_gripper"].to(device, dtype=torch.float32)
        if visual_cond.numel() > 0 and visual_cond.max() > 1.5:
            visual_cond = visual_cond / 255.0

    # -------------------------------------------------
    # build action mask
    # -------------------------------------------------
    action_mask = None
    if build_action_mask:
        if action_idx is not None:
            BK = action_idx.shape[0]
            Ta = action_idx.shape[-1]

            if batch_size is None:
                batch_size = BK

            if BK % batch_size != 0:
                raise ValueError(
                    f"assemble_conditions mismatch: action_idx.shape={action_idx.shape}, "
                    f"batch_size={batch_size}"
                )

            K = BK // batch_size
            action_idx = action_idx.reshape(K, batch_size, Ta).to(device)  # [K,B,Ta]

            key_t = torch.tensor(
                float(pull_start_t),
                dtype=torch.float32,
                device=device,
            )
            key_t_expand = key_t.view(1, 1, 1).expand(K, batch_size, Ta)

            post_key = action_idx.float() >= key_t_expand
            rel_dist = (action_idx.float() - key_t_expand).clamp(min=0.0)
            rel_dist = rel_dist / max(Ta - 1, 1)

            base = 1.75
            beta = 3.0

            action_mask = torch.ones_like(
                action_idx, dtype=torch.float32, device=device
            )
            action_mask[post_key] = base * torch.exp(beta * rel_dist[post_key])
            action_mask = torch.clamp(action_mask, max=15.0)
            action_mask = action_mask.reshape(-1, Ta).to(device, dtype=torch.float32)

        else:
            ref_for_mask = None
            for k in [
                "goal_error", # not included
                "pos_error",
                "pos",
                "ori",      # not included for open drawer
                "joint_pos",
                "active_subgoal_error",
                "pull_dir",
                "progress",
                "stage",
            ]:
                if k in cond and cond[k] is not None:
                    ref_for_mask = cond[k]
                    break

            if ref_for_mask is None:
                raise ValueError(
                    "No valid cond tensor found to infer fallback action_mask shape."
                )

            if ref_for_mask.dim() == 3:
                action_mask = torch.ones(
                    ref_for_mask.shape[0],
                    ref_for_mask.shape[1],
                    device=device,
                    dtype=torch.float32,
                )
            elif ref_for_mask.dim() == 2:
                action_mask = torch.ones(
                    ref_for_mask.shape[0],
                    1,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    f"Unexpected cond tensor shape for fallback mask: {ref_for_mask.shape}"
                )

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    joint_key = "joint" if "joint" in cond_norm_params else "joint_pos"

    def _append_error_features(xs, ref, prefix: str):
        goal_key = f"{prefix}goal_error"
        pos_err_key = f"{prefix}pos_error"
        active_key = f"{prefix}active_subgoal_error"

        if goal_key in cond and cond[goal_key] is not None:
            x = cond[goal_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos_error"]
            ).to(device, dtype=torch.float32)
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        elif pos_err_key in cond and cond[pos_err_key] is not None:
            x = cond[pos_err_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos_error"]
            ).to(device, dtype=torch.float32)
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        if active_key in cond and cond[active_key] is not None:
            x = cond[active_key]
            nx = normalizer.normalize(
                x, cond_norm_params["active_subgoal_error"]
            ).to(device, dtype=torch.float32)
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        return xs, ref

    def _append_state_features(xs, ref, prefix: str):
        pos_key = f"{prefix}pos"
        ori_key = f"{prefix}ori"
        joint_pos_key = f"{prefix}joint_pos"
        pull_dir_key = f"{prefix}pull_dir"
        progress_key = f"{prefix}progress"

        if pos_key in cond and cond[pos_key] is not None:
            x = cond[pos_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos"]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        if ori_key in cond and cond[ori_key] is not None:
            x = cond[ori_key]
            nx = normalizer.normalize(
                x, cond_norm_params["ori"]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        if joint_pos_key in cond and cond[joint_pos_key] is not None:
          
            x = cond[joint_pos_key]
            nx = normalizer.normalize(
                x, cond_norm_params[joint_key]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        if pull_dir_key in cond and cond[pull_dir_key] is not None:
            x = cond[pull_dir_key].to(device, dtype=torch.float32)
            xs.append(x)
            if ref is None:
                ref = x

        if progress_key in cond and cond[progress_key] is not None:
            x = cond[progress_key].to(device, dtype=torch.float32)
            xs.append(x)
            if ref is None:
                ref = x

        return xs, ref

    def _build_stage(prefix: str, ref):
        stage_key = f"{prefix}stage"
        stage = cond.get(stage_key, None)
        if stage is None:
            return None

        stage = stage.to(device)

        if ref is None:
            raise ValueError(
                f"No valid cond tensor found to infer {stage_key} feature shape."
            )

        if ref.dim() != 3:
            raise ValueError(
                f"Expected ref dim=3 for {stage_key} broadcast, got {ref.shape}"
            )

        Tref = ref.shape[1]

        if stage.dim() == 3 and stage.shape[-1] == 1:
            stage_out = stage
        elif stage.dim() == 2:
            stage_out = stage.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected {stage_key} shape: {stage.shape}")

        if stage_out.shape[1] == 1:
            stage_out = stage_out.expand(-1, Tref, -1)

        if stage_out.shape[1] != Tref:
            raise ValueError(
                f"{stage_key} shape {stage_out.shape} incompatible with ref shape {ref.shape}"
            )

        return stage_out.to(device, dtype=torch.long)

    # -------------------------------------------------
    # current features
    # -------------------------------------------------
    xs = []
    ref = None

    xs, ref = _append_error_features(xs, ref, prefix="")
    xs, ref = _append_state_features(xs, ref, prefix="")

    if len(xs) == 0:
        raise ValueError("No valid current cond entries found in cond.")

    xs = [x.to(device, dtype=torch.float32) for x in xs]
    n_cond = torch.cat(xs, dim=-1).to(device, dtype=torch.float32)
    stage_out = _build_stage(prefix="", ref=ref)

    # -------------------------------------------------
    # next features
    # -------------------------------------------------
    next_xs = []
    next_ref = None

    next_xs, next_ref = _append_error_features(next_xs, next_ref, prefix="next_")
    next_xs, next_ref = _append_state_features(next_xs, next_ref, prefix="next_")

    next_n_cond = None
    next_stage_out = None
    if len(next_xs) > 0:
        next_xs = [x.to(device, dtype=torch.float32) for x in next_xs]
        next_n_cond = torch.cat(next_xs, dim=-1).to(device, dtype=torch.float32)
        next_stage_out = _build_stage(prefix="next_", ref=next_ref)

    return n_cond, next_n_cond, visual_cond, action_mask, stage_out, next_stage_out