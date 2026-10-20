import torch
import torch.nn as nn
from typing import Literal


# 低维时序观测编码器：
# 先通过两层一维卷积提取局部时间特征，再用 MLP 映射到指定输出维度。
# 当前 GRU 已被注释，因此实际运行的是 Conv1d + MLP。
class Low_Dim_Obs_Encoder(nn.Module):
    """
    Encode low-dimensional temporal condition.

    Input:
        cond: [B, T, cond_dim]

    Output:
        if mode == "sequence":
            [B, T, out_dim]   # for diffusion
        if mode == "global":
            [B, out_dim]      # for critic
    """
    # B 为批大小，T 为序列长度，cond_dim 为每个时刻的输入特征数。
    def __init__(
        self,
        # 每个时刻的低维条件特征维度。
        cond_dim: int,
        # 每个时刻编码后的输出特征维度。
        out_dim: int,
        # 时间卷积的隐藏通道数，默认 128。
        conv_hidden_dim: int = 128,
        #gru_hidden_dim: int = 128,
        #gru_layers: int = 1,
        # 原本用于控制 GRU 是否双向；当前 GRU 未启用，此参数不影响前向计算。
        bidirectional: bool = False,
    ):
        # 初始化父类，使子网络参数能够被 PyTorch 自动注册。
        super().__init__()

        # 将维度配置转换为整数；cond_dim 为 None 或 0 时会得到 0，
        # 随后由下方检查拒绝无效输入维度。
        self.cond_dim = int(cond_dim or 0)
        self.out_dim = int(out_dim)
        self.conv_hidden_dim = int(conv_hidden_dim)
        #self.gru_hidden_dim = int(gru_hidden_dim)
        self.bidirectional = bool(bidirectional)

        # 低维观测至少需要包含一个特征。
        if self.cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {self.cond_dim}")

        # temporal local feature extractor
        # 使用一维卷积沿时间轴提取局部变化。
        # Conv1d 的输入布局为 [B, 特征通道, 时间长度]。
        # 两层卷积的步长默认均为 1，kernel_size=3、padding=1，
        # 因此输出序列长度与输入保持一致。
        self.conv_net = nn.Sequential(
            nn.Conv1d(
                # 将每个时刻的 cond_dim 个特征作为输入通道。
                in_channels=self.cond_dim,
                # 将输入特征映射到隐藏通道空间。
                out_channels=self.conv_hidden_dim,
                # 每次卷积覆盖三个相邻时刻。
                kernel_size=3,
                # 时间序列左右各补一个零，保持长度不变。
                padding=1
            ),
            # 非线性激活，将负值置零。
            nn.ReLU(),
            nn.Conv1d(
                # 第二层继续处理第一层提取的时间特征。
                in_channels=self.conv_hidden_dim,
                out_channels=self.conv_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # 两层大小为 3 的卷积叠加后，每个输出位置最多依赖五个原始时刻。
        # 这是对称时间卷积，内部位置会同时使用其前后邻近时刻，
        # 并不是仅访问当前位置及之前时刻的因果卷积。

        # temporal memory
        # 以下 GRU 定义已被注释，不会创建循环网络。
        # self.gru = nn.GRU(
        #     input_size=self.conv_hidden_dim,
        #     hidden_size=self.gru_hidden_dim,
        #     num_layers=gru_layers,
        #     batch_first=True,
        #     bidirectional=self.bidirectional
        # )

        #gru_out_dim = self.gru_hidden_dim * (2 if self.bidirectional else 1)

        # projection head
        # 对各时刻的卷积特征分别应用同一套 MLP。
        # Linear 只变换最后一个特征维，不改变批大小或时间长度。
        self.mlp = nn.Sequential(
            # 将卷积隐藏特征映射到 out_dim 的四倍宽度。
            nn.Linear(self.conv_hidden_dim, out_dim * 4),
            # SiLU 激活：SiLU(x) = x * sigmoid(x)。
            nn.SiLU(),
            # 压缩到最终输出维度；最后没有额外激活函数。
            nn.Linear(out_dim * 4, out_dim)
        )

    # sequence 模式返回全部时刻的特征；
    # global 模式仅返回最后一个时刻的特征，不做时间平均或全局池化。
    def forward(
        self,
        cond: torch.Tensor,
        mode: Literal["sequence", "global"] = "sequence"
    ) -> torch.Tensor:
        """
        Args:
            cond: [B, T, cond_dim]
            mode:
                - "sequence": return [B, T, out_dim]
                - "global":   return [B, out_dim]

        Returns:
            encoded cond
        """
        # 要求输入具有批次、时间和特征三个维度。
        if cond.ndim != 3:
            raise ValueError(f"cond must be [B, T, cond_dim], got shape {cond.shape}")

        # 输入特征数量必须与初始化时的 cond_dim 一致。
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond last dim {cond.shape[-1]} != cond_dim {self.cond_dim}"
            )

        # [B, T, D] -> [B, D, T]
        # 将特征维移到通道位置，让 Conv1d 沿最后的时间维进行卷积。
        # contiguous() 将换维后的张量整理成连续内存布局。
        x = cond.permute(0, 2, 1).contiguous()

        # Conv over time
        # [B, D, T] -> [B, C, T]
        # C 为 conv_hidden_dim；卷积提取相邻时刻间的局部关系。
        x = self.conv_net(x)

        # [B, C, T] -> [B, T, C]
        # 恢复时间在前、特征在后的布局，供后面的 Linear 使用。
        x = x.permute(0, 2, 1).contiguous()

        # GRU over time
        # [B, T, C] -> [B, T, H]
        # 当前未执行 GRU，因此 x 的末尾维仍然是 C。
        #x, _ = self.gru(x)

        # projection
        # [B, T, H] -> [B, T, out_dim]
        # 上方 H 是原有 GRU 方案的描述；
        # 当前实际变换为 [B, T, conv_hidden_dim] -> [B, T, out_dim]。
        x = self.mlp(x)

        # 返回整个时间序列，适合后续逐时刻处理。
        if mode == "sequence":
            return x
        elif mode == "global":
            # 只提取最后一个时间位置，输出 [B, out_dim]。
            # 因为没有 GRU 或全局池化，它不一定汇总了整段历史；
            # 当前两层卷积下，最后位置最多依赖最后三个真实输入时刻。
            # 此处也没有 padding 掩码，默认最后一个位置就是所需位置。
            return x[:, -1, :]
        else:
            # Literal 仅提供类型提示，运行时仍需显式拒绝其他模式。
            raise ValueError(f"Unsupported mode: {mode}")
