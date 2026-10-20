# Hybrid diffusion policy for CALVIN control.
# It assembles normalized multimodal conditions, trains the diffusion model,
# supports stage-aware loss and advantage weighting, and samples actions with DDPM.

# cond: total 20 dim: pos:3, joints: 7, subgoal_active_error: 3, pull_dir: 3, progress: 1, goal: 3 
# is a torch.tensor 
# 本文件组装扩散策略条件并计算训练损失；上传内容的采样函数仅包含初始化部分。
# 顶部的 20 维条件为原有说明，实际输入维度取决于 _assemble_cond 中启用的字段。
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


# 混合条件扩散策略：负责条件归一化、动作加噪、去噪目标损失及阶段分类辅助损失。
class HybridPolicy(BasePolicy):
    # 保存策略与扩散配置，创建归一化器、DDPM 调度器和诊断日志。
    def __init__(
        self,
        # 用于预测去噪目标和阶段类别的 Transformer 网络。
        model: DiffusionTransformer, # encoder accept visual obs
        # 保存的动作预测长度；本文件当前采样代码实际取位置条件的时间长度。
        action_horizon: int,
        # 计划的反向采样步数；本次上传部分尚未展示对应的去噪循环。
        num_sampling_steps: int,
        # 扩散训练的时间步总数。
        training_timesteps: int,
        # 扩散噪声调度的起始 beta。
        beta_start,
        # 扩散噪声调度的结束 beta。
        beta_end,
        # 噪声强度随扩散时间变化的调度方式，交给 DDPMScheduler 处理。
        beta_schedule: str,
        # 训练目标类型；下面的损失支持 epsilon（噪声）和 sample（干净动作）。
        prediction_type: str,
        # 策略模式配置，本次上传部分只保存该字段。
        mode: str,
        # 对归一化后的误差类条件额外施加的缩放因子。
        error_scale: float,
        # 阶段分类损失相对于扩散损失的系数。
        lambda_stage: float,
        # 轨迹中拉动阶段的起始索引，用于计算动作时间权重，不是扩散时间步。
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
        # 创建前向加噪与反向去噪调度器，训练中调用其 add_noise。
        self.scheduler = DDPMScheduler(
            self.training_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # Logs used for rollout and diffusion debugging.
        # 分别预留初始噪声、去噪过程和动作记录列表。
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
        # 目标选择器固定使用 calvin_open_drawer，并在初始化时调用 scan。
        # 扫描范围和具体行为由 TargetSelector 实现决定。
        self.target_selector = TargetSelector("calvin_open_drawer", sort_by="t")
        self.target_selector.scan()


    # 用外部传入的归一化器替换当前对象，并返回该对象。
    def set_normalizer(self, normalizer):

        # Replace the default normalizer with an externally provided one.
        self.normalizer = normalizer
        return self.normalizer


    # 从完整条件包或直接特征字典中提取输入，返回特征、视觉、掩码和阶段标签。
    def _assemble_cond(
    self,
    cond_t: dict,
    cond_norm_params: dict,
    batch_size: Optional[int],
):
     # Validate condition inputs.
     # 要求条件包及归一化参数均为字典。
     assert isinstance(cond_t, dict)
     assert isinstance(cond_norm_params, dict)

     # Collect low-dimensional condition features before concatenation.
     # 按代码顺序收集低维特征，最终沿最后一维拼接；该顺序决定模型输入各维含义。
     xs = []

     # cond_t may directly contain features or store them under "cond".
     # 兼容 cond_t["cond"] 嵌套结构和直接传入特征字典两种形式。
     if "cond" in cond_t:
        cond = cond_t["cond"]
     else:
        cond = cond_t

     # Infer batch size if it is not explicitly provided.
     # 若未显式提供批大小，依次尝试读取条件元数据或后续索引形状。
     if batch_size is None:
        batch_size = cond_t.get("batch_size", None)
     # 若未显式提供批大小，依次尝试读取条件元数据或后续索引形状。
     if batch_size is None:
        batch_size = cond_t.get("B", None)

     # Future action indices used for stage-dependent action weighting.
     # 读取未来动作的轨迹索引，用于计算拉动阶段的指数时间权重。
     action_idx = cond_t.get("act_idx", None)

    # -------------------------------------------------
    # visual cond
    # -------------------------------------------------

     # Optional visual observation condition.
     # 默认没有视觉条件；存在夹爪图像时才转换和缩放。
     visual_cond = None

     if "rgb_gripper" in cond and cond["rgb_gripper"] is not None:
        # 将视觉输入移动到策略设备并转换为 float32。
        visual_cond = cond["rgb_gripper"].to(self.device, dtype=torch.float32)

        # Convert uint8-style RGB values into approximately [0,1].
        # 通过最大值判断是否需要除以 255，这是数值启发式检查，不是根据 dtype 判断。
        if visual_cond.numel() > 0 and visual_cond.max() > 1.5:
            visual_cond = visual_cond / 255.0

    # -------------------------------------------------
    # build action mask
    # -------------------------------------------------
      
     # Build time-dependent action weights if action indices are available.
     # 有动作索引时构造时间权重 action_mask；该权重与 padding 掩码含义不同。
     if action_idx is not None:
        # BK 表示展开后的候选批大小，Ta 为动作时间长度。
        BK = action_idx.shape[0]
        Ta = action_idx.shape[-1]

        # 若未显式提供批大小，依次尝试读取条件元数据或后续索引形状。
        if batch_size is None:
            batch_size = BK

        # Flattened candidate batch must be divisible by original batch size.
        # 检查展开批大小能否被原始批大小整除。
        if BK % batch_size != 0:
            raise ValueError(
                f"_assemble_cond mismatch: action_idx.shape={action_idx.shape}, "
                f"batch_size={batch_size}"
            )

        # Recover number of candidate trajectories.
        # 根据原始批大小恢复候选数量。
        K = BK // batch_size

        # [B*K, Ta] -> [K, B, Ta]
        # 按候选优先的顺序恢复 [K, B, Ta]；需要输入的展开顺序与此约定一致。
        action_idx = action_idx.reshape(K, batch_size, Ta).to(self.device)  # [K,B,Ta]

        # Key timestep marking the beginning of the pull phase.
        # 将拉动起始步转换为设备上的浮点标量，随后广播到所有候选和动作时刻。
        key_t = torch.tensor(
            float(self.pull_start_t),
            dtype=torch.float32,
            device=self.device
        )

        key_t_expand = key_t.view(1, 1, 1).expand(K, batch_size, Ta)

        # Identify action steps after pull_start_t.
        # 标记轨迹索引达到或超过拉动起点的动作时刻。
        post_key = action_idx.float() >= key_t_expand

        # Relative temporal distance from pull_start_t.
        # 只保留相对于拉动起点的非负时间差。
        rel_dist = (action_idx.float() - key_t_expand).clamp(min=0.0)
        # 按动作窗口长度减一缩放；结果不一定在 [0, 1] 内。
        rel_dist = rel_dist / max(Ta - 1, 1)

        # Exponential weighting parameters.
        base = 1.0
        beta = 2.0

        # Default action weight is one.
        # 默认时间权重为 1，拉动起点后的动作使用更大的指数权重。
        action_mask = torch.ones_like(
            action_idx, dtype=torch.float32, device=self.device
        )

        # Increase action weights during the pull phase.
        # 按 exp(2 × 相对时间差) 增加拉动阶段权重。
        action_mask[post_key] = base * torch.exp(beta * rel_dist[post_key])

        # Prevent excessively large weights.
        # 将时间权重限制为不超过 15。
        action_mask = torch.clamp(action_mask, max=15.0)

        # Restore flattened candidate-batch representation.
        # 恢复展开候选批次的 [B*K, Ta] 形状。
        action_mask = action_mask.reshape(-1, Ta).to(
            self.device, dtype=torch.float32
        )  # [B*K,Ta]

     else:
        # 没有动作索引时，该时间权重先保持 None。
        action_mask = None

    # get boolean mask for pseudo actions
     # Retrieve masks used to ignore padded observation/action tokens.
     # 从内层 cond 读取动作填充掩码，True 表示忽略；缺少时原代码未给变量设默认值。
     if "action_padding_mask" in cond:
      
        boolean_action_mask = cond["action_padding_mask"]

     # 读取观测填充掩码；这里同样没有缺失字段的默认分支。
     if "obs_padding_mask" in cond:
        boolean_obs_mask = cond["obs_padding_mask"]

     # 读取动作损失掩码，通常有效为 1、填充为 0。
     # 下面 else 只对应本条件，不对应前两个 padding 掩码检查。
     if "action_loss_mask" in cond:
        action_loss_mask = cond["action_loss_mask"]

     else:
        # Use any available condition tensor to infer fallback mask shape.
        ref = None

        # 缺少 action_loss_mask 时，尝试从任意已知特征推断窗口形状。
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

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            raise ValueError(
                "No valid cond tensor found to infer fallback action_mask shape."
            )

        # Sequence condition.
        # 三维参考按 [B, T, D] 处理，创建 [B, T] 的全 1 时间权重。
        # 注意：该回退实际设置 action_mask，没有设置缺失的 action_loss_mask。
        if ref.dim() == 3:
            action_mask = torch.ones(
                ref.shape[0], ref.shape[1],
                device=self.device,
                dtype=torch.float32
            )

        # Single-step condition.
        # 二维参考按单步特征处理，创建 [B, 1] 的时间权重。
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
     # 误差特征优先选 goal_error，否则选 pos_error，两者不会同时加入。
     if "goal_error" in cond and cond["goal_error"] is not None:
        goal_error = cond["goal_error"]
        
        # goal_error 使用 pos_error 对应的归一化参数，再移到策略设备。
        n_goal_error = self.normalizer.normalize(
            goal_error, cond_norm_params["pos_error"]
        ).to(self.device, dtype=torch.float32)

        # Error features are additionally scaled.
        # 误差类特征在归一化后额外乘以 error_scale。
        xs.append(self.error_scale * n_goal_error)
        ref = goal_error

     # Otherwise use pos_error.
     elif "pos_error" in cond and cond["pos_error"] is not None:
        pos_error = cond["pos_error"]

        n_error = self.normalizer.normalize(
            pos_error, cond_norm_params["pos_error"]
        ).to(self.device, dtype=torch.float32)

        # 误差类特征在归一化后额外乘以 error_scale。
        xs.append(self.error_scale * n_error)
        ref = pos_error

     # Normalize Cartesian position condition.
     # 存在位置条件时，使用 pos 参数归一化并加入特征列表。
     if "pos" in cond and cond["pos"] is not None:
        pos = cond["pos"]

        n_pos = self.normalizer.normalize(
            pos, cond_norm_params["pos"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_pos)

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            ref = pos

     # Normalize orientation condition.
     # 姿态特征使用独立的 ori 归一化参数。
     if "ori" in cond and cond["ori"] is not None:
        ori = cond["ori"]

        n_ori = self.normalizer.normalize(
            ori, cond_norm_params["ori"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_ori)

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            ref = ori

     # Normalize joint-position condition.
     # 关节特征使用 joint 参数归一化。
     if "joint_pos" in cond and cond["joint_pos"] is not None:
        joint = cond["joint_pos"]

        n_joint = self.normalizer.normalize(
            joint, cond_norm_params["joint"]
        ).to(self.device, dtype=torch.float32)

        xs.append(n_joint)

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            ref = joint

     # Normalize stage-dependent active-subgoal error.
     # 有效子目标误差作为另一类条件单独归一化和缩放。
     if "active_subgoal_error" in cond and cond["active_subgoal_error"] is not None:
        active_err = cond["active_subgoal_error"]

        n_active_err = self.normalizer.normalize(
            active_err, cond_norm_params["active_subgoal_error"]
        ).to(self.device, dtype=torch.float32)

        # 误差类特征在归一化后额外乘以 error_scale。
        xs.append(self.error_scale * n_active_err)

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            ref = active_err

    #  if "pull_dir" in cond and cond["pull_dir"] is not None:
    #     pull_dir = cond["pull_dir"].to(self.device, dtype=torch.float32)
    #     xs.append(pull_dir)
    #     if ref is None:
    #         ref = pull_dir

     # Add normalized rollout progress if enabled.
     # 进度仅转换设备与类型后直接拼接，此处没有再次调用归一化器。
     if "progress" in cond and cond["progress"] is not None:
        progress = cond["progress"].to(self.device, dtype=torch.float32)

        xs.append(progress)

        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            ref = progress

    # -------------------------------------------------
    # stage feature
    # -------------------------------------------------
    # -------------------------------------------------
    # stage feature (keep separate, do NOT concat into n_cond)
    # -------------------------------------------------

     # Stage labels remain separate because they are embedded later by the model.
     # 阶段标签独立返回，不拼入 n_cond；当前读取内层观测阶段，没有读取外层 action_stage。
     stage = cond.get("stage", None)
     stage_out = None

     # 有阶段标签时，必须有三维参考特征以确定其时间长度。
     if stage is not None:
        stage = stage.to(self.device)

        # A reference sequence is required to determine stage temporal length.
        # 必要时设置参考特征，用来推断阶段标签的时间长度或判断是否可构造输入。
        if ref is None:
            raise ValueError("No valid cond tensor found to infer stage feature shape.")

        if ref.dim() != 3:
            raise ValueError(f"Expected ref dim=3 for stage broadcast, got {ref.shape}")

        # 阶段长度以观测参考长度为准，不一定等于未来动作长度。
        Tref = ref.shape[1]

        # Accept [B,T,1] or [B,T].
        # 接受 [B, T, 1] 或 [B, T]，后者通过 unsqueeze 添加末尾维。
        if stage.dim() == 3 and stage.shape[-1] == 1:
            stage_out = stage

        elif stage.dim() == 2:
            stage_out = stage.unsqueeze(-1)

        else:
            raise ValueError(f"Unexpected stage shape: {stage.shape}")

        # Broadcast a single stage value over the whole temporal window.
        # 每条轨迹只有一个阶段时，将其广播到整个参考时间窗口。
        if stage_out.shape[1] == 1:
            stage_out = stage_out.expand(-1, Tref, -1)

        # 检查阶段时间维与参考特征一致；这里没有检查其与动作预测长度是否一致。
        if stage_out.shape[1] != Tref:
            raise ValueError(
                f"stage shape {stage_out.shape} incompatible with ref shape {ref.shape}"
            )

        # keep integer-like stage labels for embedding later
        # 阶段转为整数标签，供后续嵌入和交叉熵使用。
        stage_out = stage_out.to(self.device, dtype=torch.long)

     # At least one low-dimensional condition feature must be available.
     # 至少需要一个低维特征，即使有视觉输入也不能只返回视觉条件。
     if len(xs) == 0:
        raise ValueError("No valid entries found in cond to assemble.")

     # Concatenate normalized condition features along the feature dimension.
     xs = [x.to(self.device, dtype=torch.float32) for x in xs]
     # 按照追加顺序拼接已归一化特征，除最后一维外其他维度必须兼容。
     n_cond = torch.cat(xs, dim=-1).to(self.device, dtype=torch.float32)

     # Return assembled low-dimensional condition, visual condition,
     # padding masks, action weighting masks, and stage labels.
     # 返回七项供训练和采样复用；原实现要求相关掩码变量已成功赋值。
     return n_cond, visual_cond, boolean_obs_mask, action_mask, action_loss_mask, boolean_action_mask, stage_out # action mask is designed for penalize error in pull phase, for pull phase is only a minor part of entire seq,
                                                                             # boolean action mask is designed for eliminate loss in action padding


    # 训练入口：动作归一化 → 随机加噪 → 网络预测 → 扩散损失与阶段辅助损失。
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
     # 检查首个样本的动作序列非空；此行假定批次本身也非空。
     assert len(action[0]) != 0

     model = self.model
     # 张量动作沿用自身设备，其他输入使用策略设备；调用方需要保证与模型设备一致。
     device = action.device if isinstance(action, torch.Tensor) else self.device

     # Normalize the target action sequence.
     # 将训练目标动作归一化，并统一为 float32。
     n_action = self.normalizer.normalize(
        action, action_norm_params
    ).to(device=device, dtype=torch.float32)

     # Assemble normalized condition features and masks.
     # 获取模型条件和掩码；采样调用处复用相同组装流程。
     n_cond, visual_obs, boolean_obs_mask, action_mask, action_loss_mask,  boolean_action_mask, stage = self._assemble_cond(
        cond_t=cond,
        cond_norm_params=cond_norm_params,
        batch_size=batch_size
    )

     # Move condition and masks to the correct device.
     n_cond = n_cond.to(device=device, dtype=torch.float32)
     # 这里直接调用 to，要求 action_mask 非 None；仅有损失掩码而无 act_idx 时需要注意。
     action_mask = action_mask.to(device=device, dtype=torch.float32)
     # 这里将填充标志转为 int32；是否再转成 bool 由模型内部处理。
     boolean_action_mask = boolean_action_mask.to(self.device,dtype = torch.int32)
     boolean_obs_mask = boolean_obs_mask.to(self.device,dtype = torch.int32)
     action_loss_mask = action_loss_mask.to(self.device,dtype = torch.float32)
     
     if visual_obs is not None:
        visual_obs = visual_obs.to(device=device, dtype=torch.float32)

     # Sample Gaussian diffusion noise.
     # 为每个动作元素采样独立标准高斯噪声，形状与归一化动作一致。
     noise = torch.randn_like(n_action, device=device)

     # Randomly sample one diffusion timestep for each batch element.
     # 为每个批次样本随机采样一个扩散步，同一样本的整段动作共享这个扩散步。
     training_timesteps = torch.randint(
        0, self.training_timesteps, (n_action.shape[0],), device=device
    ).long()

     # Forward diffusion: add noise to normalized action.
     # 由调度器根据扩散步混合干净动作和高斯噪声，得到带噪训练输入。
     noisy_action = self.scheduler.add_noise(n_action, noise, training_timesteps)

     #print(f"visual_obs has shape: {visual_obs.shape}") # 8,64,84,84,3 need permute

     # Predict denoising target and optional stage logits.
     # 请求网络同时输出去噪预测和阶段分类 logits。
     # 这里传入的是 _assemble_cond 返回的观测阶段，若观测与动作长度不同，需注意阶段对齐要求。
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
     # 噪声预测模式：拟合本次加噪使用的高斯噪声。
     if self.scheduler.config.prediction_type == "epsilon":
        per_elem_loss = F.mse_loss(pred, noise, reduction="none")

     # 样本预测模式：直接拟合归一化后的干净动作。
     elif self.scheduler.config.prediction_type == "sample":
        per_elem_loss = F.mse_loss(pred, n_action, reduction="none")

     else:
        raise ValueError(
            f"Unsupported prediction_type: {self.scheduler.config.prediction_type}"
        )

     # Ignore padded action tokens in diffusion loss.
     # 给损失掩码增加特征维，广播到所有动作分量。
     # 原有乘以 action_mask 的部分被注释，因此拉动阶段指数权重当前未用于扩散损失。
     weight =  action_loss_mask.unsqueeze(-1)#* action_mask.unsqueeze(-1)

     # Auxiliary stage-classification loss.
     # 没有阶段标签时，辅助分类损失保持为零。
     stage_loss = torch.tensor(0.0, device=device)

     # 有阶段标签时，必须有三维参考特征以确定其时间长度。
     if stage is not None:
        stage_target = stage

        # 将阶段目标从 [B, T, 1] 整理成 [B, T] 的类别标签。
        if stage_target.dim() == 3 and stage_target.shape[-1] == 1:
            stage_target = stage_target[..., 0]   # [B,T]

        stage_target = stage_target.long()

        # Use valid action positions for stage supervision.
        # 注意：action_mask 是正的时间权重，通常处处大于 0；它不等同于排除填充的 action_loss_mask。
        valid_mask = (action_mask > 0).float()

        # 展平批次和时间维，计算各时刻阶段分类损失，不在函数内部求均值。
        stage_loss = F.cross_entropy(
            stage_logits.view(-1,self.model.num_stages),
            stage_target.view(-1),
            reduction='none'
        ) # [B,T]

        # Mean stage classification loss over valid timesteps.
        # 按上述 valid_mask 求加权平均；要求分类目标、logits 与该掩码的时刻数量一致。
        stage_loss = (stage_loss*valid_mask.view(-1)).sum()/(valid_mask.sum()+1e-6)

     # During policy improvement, multiply diffusion loss by
     # externally provided advantage weights.
     # 在线策略改进时，将外部优势权重进一步乘入扩散损失权重。
     if improve:
        # 改进模式必须传入 w。
        assert w is not None, "w must be provided when improve=True"

        # Stop gradients through the weighting term.
        # 停止通过权重反向传播，仅让权重影响当前策略的损失大小。
        w = w.detach().to(device=per_elem_loss.device, dtype=per_elem_loss.dtype)

        # Convert [B,T] -> [B,T,1].
        # 把 [B, T] 权重变成 [B, T, 1]，便于沿动作特征维广播。
        if w.dim() == 2:
            w = w.unsqueeze(-1)

        elif w.dim() != 3:
            raise ValueError(f"Unexpected w shape: {w.shape}")

        # Weight shape must align with temporal diffusion loss.
        # 检查批次和时间长度一致；末尾维仍需满足后续广播及归一化约定。
        if w.shape[0] != per_elem_loss.shape[0] or w.shape[1] != per_elem_loss.shape[1]:
            raise ValueError(
                f"w shape {w.shape} incompatible with per_elem_loss shape {per_elem_loss.shape}"
            )

        # 结合有效位置掩码和优势权重，使填充位置继续保持零权重。
        weight = weight * w

     # Weighted mean diffusion loss over valid elements.
     # 按总权重和动作维度计算平均误差；此分母对应每个时刻共享一个标量权重的情况。
     loss = (per_elem_loss * weight).sum() / (weight.sum() * pred.shape[-1] + 1e-6)

     # Add auxiliary stage loss.
     # 将阶段分类损失按 lambda_stage 加入总损失；优势权重不直接作用于阶段损失。
     loss = loss + self.lambda_stage*stage_loss

     return loss

    
    # 采样函数不构建梯度计算图，以减少推理开销；该装饰器不会自动切换 eval 模式。
    @torch.no_grad()
    # 根据条件准备动作采样；上传的函数在初始化高斯动作后结束，未包含反向去噪与返回结果。
    def sampling_action_from_cond(
    self,
    cond_t: dict,
    action_mode,
    cond_norm_params: dict,
    rollout_step,
    batch_size=None
):
     # Determine action dimension from control mode.
     # 位置控制使用三个动作分量。
     if action_mode == "pos":
        Da = 3

     # 位置加姿态控制使用六个动作分量。
     elif action_mode == "pos_ori":
        Da = 6

     # 位置、姿态和夹爪联合控制使用七个动作分量。
     elif action_mode == "pos_ori_gripper":
        Da = 7

     else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

     # 没有传入独立随机数生成器，使用当前设备的默认随机数状态。
     generator = None

     # Assemble normalized rollout conditions.
     # 获取模型条件和掩码；采样调用处复用相同组装流程。
     n_cond, visual_obs, boolean_obs_mask, action_mask, action_loss_mask,  boolean_action_mask, stage = self._assemble_cond( 
        cond_t=cond_t,
        cond_norm_params=cond_norm_params,
        batch_size=batch_size
    ) # in sampling, it is unnecessary to build an action mask, other values in cond_t is unnecessary

     # Extract raw condition dictionary.
     # 兼容 cond_t["cond"] 嵌套结构和直接传入特征字典两种形式。
     if "cond" in cond_t:
        cond_dict = cond_t["cond"]
     else:
        cond_dict = cond_t

     # Position condition is required to infer rollout batch and temporal size.
     # 采样要求位置条件存在，即使模型还使用其他低维或视觉特征。
     if "pos" not in cond_dict or cond_dict["pos"] is None:
        raise ValueError("cond must contain cond['pos'] for sampling")

     pos_ref = cond_dict["pos"]

     # 位置输入必须为 [B, T, 3] 形式的三维张量；此处只检查维数。
     if pos_ref.dim() != 3:
        raise ValueError(f"Expected cond['pos'] dim=3, got {pos_ref.shape}")
     
     model = self.model

     # Batch size and temporal horizon used for diffusion sampling.
     # 从位置条件推断批大小。
     B = pos_ref.shape[0]
     # 当前采样长度取观测长度 T，而不是构造参数 self.action_horizon。
     T = pos_ref.shape[1]
     device = n_cond.device

     if visual_obs is not None:
        visual_obs = visual_obs.to(device=device, dtype=torch.float32)
        

     # Initialize reverse diffusion from Gaussian noise.
     # 在归一化动作空间初始化 [B, T, Da] 标准高斯噪声，作为反向扩散起点。
     action = torch.randn(
        size=[B, T, Da],
        dtype=torch.float32,
        device=device,
        generator=generator
    )

     # 上传内容到此结束；当前函数没有显式 return，按现有内容调用会返回 None。
     # Record init
