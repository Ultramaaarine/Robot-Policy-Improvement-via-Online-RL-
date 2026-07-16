import torch
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from sac_diffusion.utils.build_input_batch import build_sliding_window_batch
from sac_diffusion.models.normalizer import Normalizer


def fit_cond_normalizer(
    cfg: DictConfig,
    training_dataloader: DataLoader,
    env_goal_pos,
    max_batches: int = 50,
    max_rows: int = 200000,
    num_t_samples: int = 2,
):
    """
    Fit normalization params from raw cond dict returned by build_sliding_window_batch.
    """
    pos_xs = []
    ori_xs = []
    joint_xs = []
    error_pos_xs = []
    active_subgoal_error_xs = []

    pos_rows = 0
    ori_rows = 0
    joint_rows = 0
    error_pos_rows = 0
    active_subgoal_error_rows = 0

    normalizer = Normalizer()

    for bi, batch in enumerate(training_dataloader):
        if bi >= max_batches:
            break

        out = build_sliding_window_batch(
            cfg=cfg,
            batch=batch,
            env_goal_pos=env_goal_pos,
            t=None,
            horizon=int(getattr(cfg.training, "horizon", 16)),
        )

        cond = out["cond_t"]["cond"]

        x_pos = cond["pos"].reshape(-1, cond["pos"].shape[-1]).detach().cpu()
        take_pos = min(max_rows - pos_rows, x_pos.shape[0])
        if take_pos > 0:
            pos_xs.append(x_pos[:take_pos])
            pos_rows += take_pos

        if cond.get("ori", None) is not None:
            x_ori = cond["ori"].reshape(-1, cond["ori"].shape[-1]).detach().cpu()
            take_ori = min(max_rows - ori_rows, x_ori.shape[0])
            if take_ori > 0:
                ori_xs.append(x_ori[:take_ori])
                ori_rows += take_ori

        if cond.get("joint_pos", None) is not None:
            x_joint = cond["joint_pos"].reshape(-1, cond["joint_pos"].shape[-1]).detach().cpu()
            take_joint = min(max_rows - joint_rows, x_joint.shape[0])
            if take_joint > 0:
                joint_xs.append(x_joint[:take_joint])
                joint_rows += take_joint

        if cond.get("goal_error", None) is not None:
            x_error_pos = cond["goal_error"].reshape(-1, cond["goal_error"].shape[-1]).detach().cpu()
            take_error_pos = min(max_rows - error_pos_rows, x_error_pos.shape[0])
            if take_error_pos > 0:
                error_pos_xs.append(x_error_pos[:take_error_pos])
                error_pos_rows += take_error_pos

        if cond.get("active_subgoal_error", None) is not None:
            x_active_err = cond["active_subgoal_error"].reshape(
                -1, cond["active_subgoal_error"].shape[-1]
            ).detach().cpu()
            take_active_err = min(max_rows - active_subgoal_error_rows, x_active_err.shape[0])
            if take_active_err > 0:
                active_subgoal_error_xs.append(x_active_err[:take_active_err])
                active_subgoal_error_rows += take_active_err

        pos_done = pos_rows >= max_rows
        ori_done = (len(ori_xs) == 0) or (ori_rows >= max_rows)
        joint_done = (len(joint_xs) == 0) or (joint_rows >= max_rows)
        error_done = (len(error_pos_xs) == 0) or (error_pos_rows >= max_rows)
        active_subgoal_error_done = (
            len(active_subgoal_error_xs) == 0
        ) or (active_subgoal_error_rows >= max_rows)

        if pos_done and ori_done and joint_done and error_done and active_subgoal_error_done:
            break

    if len(pos_xs) == 0:
        raise RuntimeError("No samples collected for pos normalizer fit.")

    pos_norm_params = normalizer.fit(torch.cat(pos_xs, dim=0), mode="gaussian")

    ori_norm_params = (
        normalizer.fit(torch.cat(ori_xs, dim=0), mode="gaussian")
        if len(ori_xs) > 0 else None
    )

    joint_norm_params = (
        normalizer.fit(torch.cat(joint_xs, dim=0), mode="gaussian")
        if len(joint_xs) > 0 else None
    )

    error_pos_norm_params = (
        normalizer.fit(torch.cat(error_pos_xs, dim=0), mode="gaussian")
        if len(error_pos_xs) > 0 else None
    )

    active_subgoal_error_norm_params = (
        normalizer.fit(torch.cat(active_subgoal_error_xs, dim=0), mode="gaussian")
        if len(active_subgoal_error_xs) > 0 else None
    )

    return {
        "pos": pos_norm_params,
        "ori": ori_norm_params,
        "joint": joint_norm_params,
        "pos_error": error_pos_norm_params,
        "active_subgoal_error": active_subgoal_error_norm_params,
    }