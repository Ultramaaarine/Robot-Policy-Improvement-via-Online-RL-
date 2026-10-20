# cond: 20 dim, pos: 3, joint: 7, goal_error: 3 active_subgoal_error: 3, stage: 1, pull_dir: 3
# 上面的维度说明是原有描述，实际条件维度取决于输入中提供了哪些字段。
# 当前实现不把 stage 拼接进条件向量，而是将阶段标签单独返回。
# 当前状态和下一状态使用同一套归一化参数与特征排列规则。
import torch
from typing import Optional
from sac_diffusion.models.normalizer import Normalizer


# 将条件字典整理成网络可使用的输入：
# 当前低维条件、下一低维条件、当前视觉条件、动作时间权重及两组阶段标签。
# 常见低维条件形状为 [B, T, D]：
# B 为批大小，T 为条件序列长度，D 为拼接后的特征维度。
def assemble_conditions(
    # 支持完整条件包 {"cond": {...}, ...}，也支持直接传入特征字典。
    cond_t: dict,
    # 各类特征对应的归一化参数。
    cond_norm_params: dict,
    # 原始批大小；为 None 时尝试从条件元数据或动作索引推断。
    batch_size: Optional[int],
    # 输出张量所在设备，例如 CPU 或 CUDA。
    device,
    # 误差特征归一化后额外乘以的缩放系数。
    error_scale: float,
    # 拉动阶段的轨迹起始步，用于构造动作时间权重。
    pull_start_t: int,
    
    # 是否构造动作时间权重；关闭时 action_mask 返回 None。
    build_action_mask: bool,
):
    # 检查输入结构，并创建用于处理特征的归一化器。
    assert isinstance(cond_t, dict)
    assert isinstance(cond_norm_params, dict)
    normalizer = Normalizer()

    # 如果存在内层 cond，则从内层读取特征。
    cond = cond_t["cond"] if "cond" in cond_t else cond_t

    # 未显式提供批大小时，依次尝试 batch_size 和旧字段 B。
    if batch_size is None:
        batch_size = cond_t.get("batch_size", None)
    if batch_size is None:
        batch_size = cond_t.get("B", None)

    # 未来动作在原轨迹中的索引，通常为 [B*K, Ta]。
    # K 表示每个样本对应的候选数量，Ta 表示未来动作长度。
    action_idx = cond_t.get("act_idx", None)

    # -------------------------------------------------
    # visual cond (current only for now)
    # -------------------------------------------------
    # 当前只处理当前状态的视觉条件，不构造 next_visual_cond。
    visual_cond = None
    if "rgb_gripper" in cond and cond["rgb_gripper"] is not None:
        # 统一转换为目标设备上的 float32。
        visual_cond = cond["rgb_gripper"].to(device, dtype=torch.float32)

        # 通过最大值判断是否仍为近似 0～255 的像素值。
        # 已经缩放到 0～1 的图像通常不会再次除以 255。
        # 这是按数值范围判断，并不是严格检查原始图像类型。
        if visual_cond.numel() > 0 and visual_cond.max() > 1.5:
            visual_cond = visual_cond / 255.0

    # -------------------------------------------------
    # build action mask
    # -------------------------------------------------
    # 注意：这里的 action_mask 是动作时间权重，不是 padding 掩码。
    # 默认权重为 1，拉动阶段的权重可以大于 1。
    action_mask = None
    if build_action_mask:
        if action_idx is not None:
            # BK 为展开后的批大小，Ta 为动作序列长度。
            BK = action_idx.shape[0]
            Ta = action_idx.shape[-1]

            # 没有批大小信息时，把所有输入视为一个普通批次，即 K=1。
            if batch_size is None:
                batch_size = BK

            # 展开后的批大小必须能被原始批大小整除。
            if BK % batch_size != 0:
                raise ValueError(
                    f"assemble_conditions mismatch: action_idx.shape={action_idx.shape}, "
                    f"batch_size={batch_size}"
                )

            # 恢复候选数量，并按 [K, B, Ta] 解释动作索引。
            # 输入展开顺序需要与这里候选优先的 reshape 约定一致。
            K = BK // batch_size
            action_idx = action_idx.reshape(K, batch_size, Ta).to(device)  # [K,B,Ta]

            # 将拉动起点转换成浮点标量，并广播到所有动作位置。
            key_t = torch.tensor(
                float(pull_start_t),
                dtype=torch.float32,
                device=device,
            )
            key_t_expand = key_t.view(1, 1, 1).expand(K, batch_size, Ta)

            # 找出达到拉动起点的动作位置。
            post_key = action_idx.float() >= key_t_expand

            # 计算每个动作距离拉动起点的非负时间差。
            rel_dist = (action_idx.float() - key_t_expand).clamp(min=0.0)

            # 按动作窗口长度缩放时间差，Ta=1 时使用分母 1。
            # 此值不一定小于等于 1，因为动作索引可能远晚于拉动起点。
            rel_dist = rel_dist / max(Ta - 1, 1)

            # 拉动起点处的基础权重为 1.75，之后按指数增长。
            base = 1.75
            beta = 3.0

            # 非拉动阶段的权重保持为 1。
            action_mask = torch.ones_like(
                action_idx, dtype=torch.float32, device=device
            )

            # 拉动阶段越靠后的动作，权重越大。
            action_mask[post_key] = base * torch.exp(beta * rel_dist[post_key])

            # 将权重限制在最大 15，随后恢复展开批次的二维布局。
            action_mask = torch.clamp(action_mask, max=15.0)
            action_mask = action_mask.reshape(-1, Ta).to(device, dtype=torch.float32)

        else:
            # 没有动作索引时，无法根据拉动起点计算时间权重。
            # 从已有条件中寻找参考张量，用其形状创建全 1 权重。
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

            # 所有参考字段都不存在时，无法确定批大小和时间长度。
            if ref_for_mask is None:
                raise ValueError(
                    "No valid cond tensor found to infer fallback action_mask shape."
                )

            # 三维参考按 [B, T, D] 处理，构造 [B, T] 权重。
            # 此处使用的是参考条件长度，不保证等于未来动作长度 Ta。
            if ref_for_mask.dim() == 3:
                action_mask = torch.ones(
                    ref_for_mask.shape[0],
                    ref_for_mask.shape[1],
                    device=device,
                    dtype=torch.float32,
                )

            # 二维参考按 [B, D] 单步特征处理，构造 [B, 1] 权重。
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
    # 兼容两种关节归一化参数名称：优先 joint，否则使用 joint_pos。
    # 只有实际使用关节特征时，才会访问对应参数。
    joint_key = "joint" if "joint" in cond_norm_params else "joint_pos"

    # 添加误差类条件特征。
    # prefix="" 处理当前状态，prefix="next_" 处理下一状态。
    # ref 保存第一个有效特征，用于后面推断阶段标签的时间长度。
    def _append_error_features(xs, ref, prefix: str):
        goal_key = f"{prefix}goal_error"
        pos_err_key = f"{prefix}pos_error"
        active_key = f"{prefix}active_subgoal_error"

        # 优先使用 goal_error，并使用 pos_error 对应的参数归一化。
        if goal_key in cond and cond[goal_key] is not None:
            x = cond[goal_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos_error"]
            ).to(device, dtype=torch.float32)

            # 归一化后额外缩放误差特征。
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        # 没有 goal_error 时才使用 pos_error，避免重复加入同类误差。
        elif pos_err_key in cond and cond[pos_err_key] is not None:
            x = cond[pos_err_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos_error"]
            ).to(device, dtype=torch.float32)
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        # 有效子目标误差是独立特征，可以与上面的目标误差同时存在。
        if active_key in cond and cond[active_key] is not None:
            x = cond[active_key]
            nx = normalizer.normalize(
                x, cond_norm_params["active_subgoal_error"]
            ).to(device, dtype=torch.float32)
            xs.append(error_scale * nx)
            if ref is None:
                ref = x

        return xs, ref

    # 添加位置、姿态、关节、拉动方向和进度等状态特征。
    # 当前与下一状态共用此函数，从而保持一致的特征排列规则。
    def _append_state_features(xs, ref, prefix: str):
        pos_key = f"{prefix}pos"
        ori_key = f"{prefix}ori"
        joint_pos_key = f"{prefix}joint_pos"
        pull_dir_key = f"{prefix}pull_dir"
        progress_key = f"{prefix}progress"

        # 位置使用 pos 对应的归一化参数。
        if pos_key in cond and cond[pos_key] is not None:
            x = cond[pos_key]
            nx = normalizer.normalize(
                x, cond_norm_params["pos"]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        # 姿态使用 ori 对应的归一化参数。
        if ori_key in cond and cond[ori_key] is not None:
            x = cond[ori_key]
            nx = normalizer.normalize(
                x, cond_norm_params["ori"]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        # 关节状态使用前面选择的 joint 或 joint_pos 参数。
        if joint_pos_key in cond and cond[joint_pos_key] is not None:
          
            x = cond[joint_pos_key]
            nx = normalizer.normalize(
                x, cond_norm_params[joint_key]
            ).to(device, dtype=torch.float32)
            xs.append(nx)
            if ref is None:
                ref = x

        # 拉动方向只转换设备与类型，直接拼接，不额外归一化。
        # 输入方向是否为单位向量，由上游条件构建代码负责。
        if pull_dir_key in cond and cond[pull_dir_key] is not None:
            x = cond[pull_dir_key].to(device, dtype=torch.float32)
            xs.append(x)
            if ref is None:
                ref = x

        # 进度同样直接使用，不在这里执行数值缩放。
        if progress_key in cond and cond[progress_key] is not None:
            x = cond[progress_key].to(device, dtype=torch.float32)
            xs.append(x)
            if ref is None:
                ref = x

        return xs, ref

    # 将阶段标签整理成 [B, T, 1] 的整数张量。
    # 阶段独立返回，不作为普通连续特征拼接进 n_cond。
    def _build_stage(prefix: str, ref):
        stage_key = f"{prefix}stage"
        stage = cond.get(stage_key, None)

        # 没有提供阶段时直接返回 None。
        if stage is None:
            return None

        stage = stage.to(device)

        # 必须有参考特征，才能确定阶段应对应多长的时间窗口。
        if ref is None:
            raise ValueError(
                f"No valid cond tensor found to infer {stage_key} feature shape."
            )

        # 当前阶段处理要求参考特征为 [B, T, D]。
        if ref.dim() != 3:
            raise ValueError(
                f"Expected ref dim=3 for {stage_key} broadcast, got {ref.shape}"
            )

        Tref = ref.shape[1]

        # 接受 [B, T, 1]，或将 [B, T] 转换为 [B, T, 1]。
        if stage.dim() == 3 and stage.shape[-1] == 1:
            stage_out = stage
        elif stage.dim() == 2:
            stage_out = stage.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected {stage_key} shape: {stage.shape}")

        # 每个样本只有一个阶段标签时，将其广播到整个时间窗口。
        if stage_out.shape[1] == 1:
            stage_out = stage_out.expand(-1, Tref, -1)

        # 广播后阶段时间长度必须与参考条件一致。
        if stage_out.shape[1] != Tref:
            raise ValueError(
                f"{stage_key} shape {stage_out.shape} incompatible with ref shape {ref.shape}"
            )

        # 转成类别索引类型；此处不检查阶段取值范围，也不做裁剪。
        return stage_out.to(device, dtype=torch.long)

    # -------------------------------------------------
    # current features
    # -------------------------------------------------
    # 当前条件先加入误差类，再加入状态类。
    # 实际顺序为：
    # goal_error 或 pos_error → active_subgoal_error →
    # pos → ori → joint_pos → pull_dir → progress。
    # 缺失或为 None 的字段会跳过。
    xs = []
    ref = None

    xs, ref = _append_error_features(xs, ref, prefix="")
    xs, ref = _append_state_features(xs, ref, prefix="")

    # 至少需要一个当前低维特征；仅提供视觉或阶段标签不足以通过这里。
    if len(xs) == 0:
        raise ValueError("No valid current cond entries found in cond.")

    # 统一设备与类型，然后沿最后一个特征维拼接。
    # 各特征的批次维和时间维必须一致。
    xs = [x.to(device, dtype=torch.float32) for x in xs]
    n_cond = torch.cat(xs, dim=-1).to(device, dtype=torch.float32)

    # 当前阶段与当前参考条件的时间长度对齐。
    stage_out = _build_stage(prefix="", ref=ref)

    # -------------------------------------------------
    # next features
    # -------------------------------------------------
    # 下一状态通过 next_ 前缀读取对应字段，复用相同归一化参数。
    next_xs = []
    next_ref = None

    next_xs, next_ref = _append_error_features(next_xs, next_ref, prefix="next_")
    next_xs, next_ref = _append_state_features(next_xs, next_ref, prefix="next_")

    # 允许完全不提供下一状态特征，例如仅执行当前状态动作推理时。
    next_n_cond = None
    next_stage_out = None
    if len(next_xs) > 0:
        # 按相同规则拼接下一状态条件。
        # 本函数不会自动补齐缺失字段；若当前/下一字段不对称，
        # 两者的特征维度或语义排列可能不一致，需要调用方保证匹配。
        next_xs = [x.to(device, dtype=torch.float32) for x in next_xs]
        next_n_cond = torch.cat(next_xs, dim=-1).to(device, dtype=torch.float32)
        next_stage_out = _build_stage(prefix="next_", ref=next_ref)

    # 返回：
    # n_cond：当前低维条件，通常为 [B, T, D]。
    # next_n_cond：下一低维条件，缺失时为 None。
    # visual_cond：当前视觉条件，缺失时为 None。
    # action_mask：动作时间权重，未启用时为 None，不负责屏蔽 padding。
    # stage_out / next_stage_out：独立的整数阶段标签，缺失时为 None。
    return n_cond, next_n_cond, visual_cond, action_mask, stage_out, next_stage_out
