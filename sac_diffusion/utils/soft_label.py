import torch
import torch.nn as nn
import numpy as np
from sac_diffusion.utils.target_selector import TargetSelector
class HeuristicNet(nn.Module):
    def __init__(self,cond_dim,conv_out_dim):
     super().__init__()
     self.cond_dim = cond_dim
     self.conv_layer = nn.Sequential(
        nn.Conv1d(self.cond_dim,128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv1d(128,conv_out_dim,kernel_size=3,padding=1), # [B,256,T]
        nn.ReLU()
     )
     self.mlp = nn.Sequential(
        nn.Linear(conv_out_dim,cond_dim), #[B,T,D]
        nn.ReLU(),
        nn.Linear(cond_dim,1),
        nn.Sigmoid()
     )

    def forward(self,x:torch.Tensor):
        # x: [B,63,6]
        x = x.permute(0,2,1)
        conv_obs = self.conv_layer(x) # [B,256,T]
        conv_obs = conv_obs.permute(0,2,1) #[B,T,conv_dim]
        pred_label = self.mlp(conv_obs) #[B,T,D]
         
        return pred_label # [B,To,1]

def get_soft_label(x, label, cfg, B):
    """
    x: numpy array [T]
    B: batch size
    return: torch tensor [B,T,1]
    """

    target_selector = TargetSelector(skill = cfg.skill,sort_by=cfg.sort_by)
    target_selector.scan()

    out_mean, out_std = target_selector.typical_progress(label=label,method="mean")
    out_mean = out_mean*63
    out_std = out_std*63
    print(f"out_mean:{out_mean}, out_std:{out_std}")

    out_std = max(float(out_std), 1e-6)

    # Gaussian label
    prob = np.exp(-((x - out_mean) ** 2) / (2 * out_std * out_std))  # [T]

    prob = torch.tensor(prob, dtype=torch.float32)  # [T]

    # -> [1,T,1]
    prob = prob.unsqueeze(0).unsqueeze(-1)

    # -> [B,T,1]
    prob = prob.expand(B, -1, -1)

    return prob

