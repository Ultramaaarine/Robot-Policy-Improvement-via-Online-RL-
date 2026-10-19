# PyTorch 提供张量运算，nn 提供神经网络层；下面还导入类型标注和项目内的编码模块。
import torch
# PyTorch 提供张量运算，nn 提供神经网络层；下面还导入类型标注和项目内的编码模块。
import torch.nn as nn
from typing import Literal, Optional
from sac_diffusion.utils.positional_embedder import SinusoidalEmbbeder, PositionalEmbeder
from sac_diffusion.models.mixed_obs_MLP import Mixed_Obs_MLP


# 扩散模型中的条件 Transformer：观测序列作为条件，带噪动作序列作为解码输入。
# 输出每个动作时刻的预测向量；其训练目标是噪声、动作还是其他量，由外部损失函数决定。
# 维度约定：B 为批大小，To 为观测序列长度，Ta 为未来动作序列长度。
class DiffusionTransformer(nn.Module):
    # 构建观测编码器、动作解码器、阶段嵌入和输出层。
    def __init__(
        self,
        # 每个时刻的低维观测特征数。
        cond_dim: int,
        # 每个时刻的动作向量维度。
        action_dim: int,
        # 是否启用视觉观测；下面的观测相关参数会传给 Mixed_Obs_MLP。
        enable_visual_obs: bool,
        # 低维观测编码器的输出特征维度。
        low_dim_obs_encoder_out_dim: int,
        # 是否仅使用视觉观测，由 Mixed_Obs_MLP 的具体实现处理。
        use_visual_obs_only: bool,
        # 传给混合观测编码器的维度配置，具体用途由该模块定义。
        diffusion_input_dim: int,
        # 视觉观测的特征维度配置。
        visual_obs_feature_dim: int,
        # 观测组合方式开关，具体组合逻辑位于 Mixed_Obs_MLP。
        combine_obs: bool,
        # Transformer 的统一特征维度，需要能被注意力头数 nhead 整除。
        d_model: int,
        # 多头注意力的头数。
        nhead: int,
        # 动作 Transformer 解码器的层数。
        num_layers: int,
        # Transformer 每层前馈网络的隐藏维度。
        dim_feedforward: int,
        # Transformer 内部的 dropout 比例。
        dropout: float,
        # 传给位置编码模块的最大序列长度配置。
        max_len: int,
        # 是否创建时间嵌入的 MLP；当前 forward() 没有调用本类的 time_emb 和 time_mlp。
        use_time_mlp: bool = True,
        # 动作阶段的类别数，合法阶段索引为 0 到 num_stages - 1。
        num_stages: int = 3,
        # 阶段查表嵌入的维度，之后会投影到 d_model。
        stage_emb_dim: int = 32,
        # 观测条件 Transformer 编码器的层数。
        cond_num_layers: int = 1,
    ):
        # 初始化 nn.Module，使后续赋值的子模块和参数能被自动注册。
        super().__init__()
        # 保存主要维度，并确保这些配置是整数。
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_stages = int(num_stages)
        self.stage_emb_dim = int(stage_emb_dim)

        # 保存混合观测编码器所需的配置。
        self.enable_visual_obs = enable_visual_obs
        self.low_dim_obs_encoder_out_dim = low_dim_obs_encoder_out_dim
        self.visual_obs_feature_dim = visual_obs_feature_dim
        self.diffusion_input_dim = diffusion_input_dim
        self.use_visual_obs_only = use_visual_obs_only
        self.combine_obs = combine_obs

        # input projection
        # 把每个带噪动作从 action_dim 维映射到 d_model 维。
        self.input_proj = nn.Linear(self.action_dim, self.d_model)

        # cond encoder -> [B,T,d_model]
        # 构建混合观测编码器，预期输出 [B, To, d_model] 的观测特征序列。
        self.obs_embeder = Mixed_Obs_MLP(
            self.d_model,
            self.cond_dim,
            self.enable_visual_obs,
            self.low_dim_obs_encoder_out_dim,
            self.visual_obs_feature_dim,
            self.diffusion_input_dim,
            self.use_visual_obs_only,
            self.combine_obs
        )

        # 保留此位置编码模块；当前 forward() 使用下面两个专用模块，未调用 self.pos_emb。
        self.pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )
        # 为观测序列提供位置编码，帮助模型区分观测时刻。
        self.cond_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )
        # 为未来动作序列提供位置编码，帮助模型区分动作时刻。
        self.action_pos_emb = PositionalEmbeder(
            emb_dim=self.d_model,
            max_len=max_len
        )


        # diffusion timestep emb
        # 创建扩散步的正弦嵌入模块；当前前向流程直接将 diffusion_t 传给 obs_embeder。
        self.time_emb = SinusoidalEmbbeder(self.d_model)
        # 启用时，用两层线性网络和 SiLU 加工时间嵌入；中间维度扩展为四倍。
        if use_time_mlp:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.SiLU(),
                nn.Linear(self.d_model * 4, self.d_model),
            )
        else:
            # 关闭时使用恒等映射，即输入原样输出。
            self.time_mlp = nn.Identity()

        # stage embedding (Learned Positional Embedding) not same as positional embedding
        # 为离散阶段 ID 学习向量表示；阶段类别与序列中的时间位置是不同概念。
        self.stage_emb = nn.Embedding(self.num_stages, self.stage_emb_dim) # get 3 stages, for each stage, we embded it as a vector [1,2,3]-> [vector,vector,vector]
        
        # 将阶段嵌入映射到 d_model 维，以便与动作特征逐元素相加。
        self.stage_proj = nn.Sequential(
            nn.Linear(self.stage_emb_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        # 定义观测编码层，通过自注意力整合不同观测时刻的信息。
        # batch_first=True 使用 [B, T, D] 布局；norm_first=True 在注意力和前馈子层之前归一化。
        cond_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        # 堆叠指定数量的观测编码层，输出供动作解码器查询的 memory。
        self.cond_encoder = nn.TransformerEncoder(
            cond_encoder_layer,
            num_layers=int(cond_num_layers)
        )

        # 定义动作解码层：动作之间做自注意力，动作与观测 memory 之间做交叉注意力。
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 堆叠 num_layers 个解码层。
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )


        # 下面是原有的旧编码器方案，整段已被注释，不参与当前执行。
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation="gelu",
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.transformer = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=num_layers
        # ) # assembel a transformer with muitiple nn.TRansformerEncoderLayer
        
        # 对解码器输出的最后一个特征维度进行 LayerNorm。
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # 阶段分类头输出未经过 softmax 的 logits；动作输出头恢复 action_dim 维。
        self.stage_head = nn.Linear(self.d_model,self.num_stages)
        self.out_proj_layer = nn.Linear(self.d_model, self.action_dim)

    # 检查 padding 掩码的形状，并统一设备和类型；True 表示填充位置，False 表示有效位置。
    def _to_bool_padding_mask(
        self,
        mask: Optional[torch.Tensor],
        shape: tuple[int, int],
        device: torch.device,
        name: str,
    ) -> Optional[torch.Tensor]:
        """
        Padding mask convention:
            True  = padding / ignore
            False = valid
        """
        # 没有提供掩码时返回 None，后续注意力不应用该 padding 掩码。
        if mask is None:
            return None

        # 要求掩码与对应序列的 [B, T] 维度完全一致。
        if mask.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {mask.shape}"
            )

        # 将掩码放到目标设备上并转换为布尔类型；数值非零会转换为 True。
        return mask.to(device=device, dtype=torch.bool)

    # 将填充位置的全部特征置零，保持张量形状不变。
    def _zero_padded_tokens(
        self,
        tokens: torch.Tensor,                  # [B, T, D]
        padding_mask: Optional[torch.Tensor],  # [B, T], True = padding
    ) -> torch.Tensor:
        # 未提供掩码时直接返回原始特征。
        if padding_mask is None:
            return tokens

        # 将 [B, T] 掩码扩展为 [B, T, 1]，广播到所有特征通道。
        return tokens.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

    # 将阶段标签整理为 [B, Ta]，再编码成与未来动作逐时刻对齐的特征。
    def _build_action_stage_feat(
        self,
        stage: Optional[torch.Tensor],
        B: int,
        Ta: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        stage should align with future action horizon:
            stage: [B, Ta], [B, Ta, 1], or [B, 1]
        """
        # 未提供阶段时，所有动作时刻使用阶段 0 的可学习嵌入，并非零向量。
        if stage is None:
            stage_seq = torch.zeros(
                (B, Ta),
                device=device,
                dtype=torch.long,
            )
        else:
            # 接受 [B, Ta, 1] 格式时，去掉最后一维；二维格式则直接使用。
            if stage.dim() == 3 and stage.shape[-1] == 1:
                stage_seq = stage[..., 0]
            elif stage.dim() == 2:
                stage_seq = stage
            else:
                raise ValueError(f"Unexpected stage shape: {stage.shape}")

            # 阶段标签的批大小必须与动作批大小一致。
            if stage_seq.shape[0] != B:
                raise ValueError(
                    f"stage batch {stage_seq.shape[0]} != action batch {B}"
                )

            # 若每个样本只有一个阶段标签，则将该标签广播到全部 Ta 个动作时刻。
            if stage_seq.shape[1] == 1:
                stage_seq = stage_seq.expand(B, Ta)

            # 广播后必须与未来动作长度一致，不能直接使用长度不同的历史观测阶段序列。
            if stage_seq.shape != (B, Ta):
                raise ValueError(
                    f"stage must broadcast to [B, Ta], got {stage_seq.shape}, "
                    f"expected {(B, Ta)}. "
                    f"Remember: stage should align with future action indices."
                )

            # 将阶段标签移动到动作所在的设备。
            stage_seq = stage_seq.to(device=device)
            # 转为整数索引，并将越界值截断到合法阶段范围。
            stage_seq = stage_seq.long().clamp(
                min=0,
                max=self.num_stages - 1,
            )

        # 查表获取每个动作时刻的阶段嵌入，再通过投影网络统一特征维度。
        stage_feat = self.stage_emb(stage_seq)       # [B, Ta, stage_emb_dim]
        stage_feat = self.stage_proj(stage_feat)     # [B, Ta, d_model]
        return stage_feat

    # 前向流程：编码观测条件 → 编码带噪动作和阶段 → 交叉注意力解码 → 输出预测。
    def forward(
        self,
        # 带噪的未来动作序列；Ta 可以与观测长度 To 不同。
        x: torch.Tensor,                    # [B, Ta, action_dim], noisy action
        # 低维观测序列；即使使用纯视觉配置，本函数仍要求此参数满足形状检查。
        low_dim_obs: torch.Tensor,          # [B, To, cond_dim]
        # 可选的视觉观测序列，实际处理方式由混合观测编码器决定。
        visual_obs: Optional[torch.Tensor], # [B, To, C, H, W] or None
        # 每个样本对应的扩散时间步，不是观测或动作序列中的位置编号。
        diffusion_t: torch.Tensor,          # [B]
        # 阶段标签与未来动作对齐；省略时默认使用阶段 0。
        stage: Optional[torch.Tensor] = None,  # future action stage: [B, Ta, 1] / [B, Ta] / [B, 1]
        # sequence 返回逐时刻动作预测，global 沿动作时间维求平均。
        mode: Literal["sequence", "global"] = "sequence",


        # new names
        cond_padding_mask: Optional[torch.Tensor] = None,     # [B, To], True = padded cond token
        action_padding_mask: Optional[torch.Tensor] = None,  # [B, Ta], True = padded action token

        # 为 True 时返回 (动作序列预测, 阶段 logits)，并提前返回，不执行 mode 分支。
        return_stage_logits: bool = False,
    ) -> torch.Tensor:

        # 检查动作输入的维度和末尾特征数，避免后续投影或注意力计算形状不匹配。
        if x.ndim != 3:
            raise ValueError(f"x must be [B, Ta, action_dim], got {x.shape}")
        if x.shape[-1] != self.action_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} != action_dim {self.action_dim}"
            )

        # 检查低维观测的维度和特征数。
        if low_dim_obs.ndim != 3:
            raise ValueError(
                f"low_dim_obs must be [B, To, cond_dim], got {low_dim_obs.shape}"
            )
        if low_dim_obs.shape[-1] != self.cond_dim:
            raise ValueError(
                f"low_dim_obs last dim {low_dim_obs.shape[-1]} != cond_dim {self.cond_dim}"
            )

        # 提取批大小、动作长度和观测长度，并以动作张量的设备作为目标设备。
        B, Ta, _ = x.shape
        B_obs, To, _ = low_dim_obs.shape
        device = x.device

        # 动作与观测必须来自同一批样本。
        if B_obs != B:
            raise ValueError(
                f"Batch mismatch: x batch={B}, low_dim_obs batch={B_obs}"
            )


        # 检查观测掩码，转换为 [B, To] 的布尔张量。
        obs_padding_mask = self._to_bool_padding_mask(
            cond_padding_mask,
            shape=(B, To),
            device=device,
            name="obs_padding_mask",
        )

        # 检查动作掩码，转换为 [B, Ta] 的布尔张量。
        action_padding_mask = self._to_bool_padding_mask(
            action_padding_mask,
            shape=(B, Ta),
            device=device,
            name="action_padding_mask",
        )

        # ------------------------------------------------------------
        # 1. cond / obs tokens
        # ------------------------------------------------------------
        # 将视觉观测、低维观测和扩散步传给混合观测编码器。
        encoded_cond = self.obs_embeder(
            visual_obs=visual_obs,
            low_dim_obs=low_dim_obs,
            time_emb=diffusion_t,
        )  # [B, To, d_model]

        # 确认编码器输出严格符合 [B, To, d_model]。
        if encoded_cond.ndim != 3:
            raise ValueError(
                f"encoded_cond must be [B, To, d_model], got {encoded_cond.shape}"
            )
        if encoded_cond.shape != (B, To, self.d_model):
            raise ValueError(
                f"encoded_cond shape {encoded_cond.shape} != expected {(B, To, self.d_model)}"
            )

        # padding token 先置零
        encoded_cond = self._zero_padded_tokens(
            encoded_cond,
            obs_padding_mask,
        )

        # cond positional embedding
        # 为观测特征加入位置信息；之前置零的填充位置仍可能因位置编码变为非零。
        encoded_cond = self.cond_pos_emb(encoded_cond)

        # cond encoder
        # 编码观测序列；掩码让注意力忽略填充位置对应的 key。
        # 这里未传入因果掩码，因此有效观测时刻之间可以相互关注。
        memory = self.cond_encoder(
            encoded_cond,
            src_key_padding_mask=obs_padding_mask,
        )  # [B, To, d_model]

        # ------------------------------------------------------------
        # 2. noisy action tokens
        # ------------------------------------------------------------
        # 将带噪动作映射到 Transformer 的特征空间。
        action_tokens = self.input_proj(x)  # [B, Ta, d_model]

        # add future action stage embedding
        # 为每个未来动作时刻构造阶段条件特征。
        stage_feat = self._build_action_stage_feat(
            stage=stage,
            B=B,
            Ta=Ta,
            device=device,
        )
        # 将动作内容与阶段信息相加，形状仍为 [B, Ta, d_model]。
        action_tokens = action_tokens + stage_feat

        # padding action token 置零
        # 先清零填充动作的特征；后续位置编码与解码层仍可能产生非零输出。
        action_tokens = self._zero_padded_tokens(
            action_tokens,
            action_padding_mask,
        )

        # action positional embedding
        # 加入未来动作的位置编码，保留动作序列的先后次序信息。
        action_tokens = self.action_pos_emb(action_tokens)

        # ------------------------------------------------------------
        # 3. decoder: action tokens cross-attend to cond memory
        # ------------------------------------------------------------
        # 动作序列作为 tgt 查询观测 memory；两个掩码分别屏蔽动作和观测的填充 key。
        # 未传入因果 tgt_mask，因此动作时刻之间可以双向关注。
        # key padding 掩码不会保证填充 query 的输出为零，训练损失通常也需要屏蔽填充位置。
        action_tokens = self.action_decoder(
            tgt=action_tokens,
            memory=memory,
            tgt_key_padding_mask=action_padding_mask,
            memory_key_padding_mask=obs_padding_mask,
        )  # [B, Ta, d_model]

        # 归一化解码后的特征，再分别送入阶段分类头和动作输出头。
        action_tokens = self.final_norm(action_tokens)

        # ------------------------------------------------------------
        # 4. output
        # ------------------------------------------------------------
        # 为每个未来动作时刻输出各阶段的分类分数，不在这里转换成概率。
        stage_logits = self.stage_head(action_tokens)      # [B, Ta, num_stages]
        # 为每个未来动作时刻输出 action_dim 维预测向量。
        action_out = self.out_proj_layer(action_tokens)    # [B, Ta, action_dim]

        # 同时请求阶段预测时，返回两个张量；实际返回类型为元组。
        if return_stage_logits:
            return action_out, stage_logits

        # 序列模式保留全部动作时刻，返回 [B, Ta, action_dim]。
        if mode == "sequence":
            return action_out

        # 全局模式沿动作时间维平均，返回 [B, action_dim]。
        elif mode == "global":
            # 没有填充掩码时，所有动作时刻均参与平均。
            if action_padding_mask is None:
                return action_out.mean(dim=1)

            # 取反得到有效位置，并扩展为 [B, Ta, 1]，以便广播到各动作分量。
            valid = (~action_padding_mask).float().unsqueeze(-1)  # [B, Ta, 1]
            # 统计每个样本的有效动作数量；最小设为 1，避免全填充时除零。
            denom = valid.sum(dim=1).clamp_min(1.0)
            # 屏蔽填充位置后求和，再除以有效数量；预测值有限时，全填充样本得到零向量。
            return (action_out * valid).sum(dim=1) / denom

        else:
            # 拒绝未支持的输出模式。
            raise ValueError(f"Unsupported mode: {mode}")
  
