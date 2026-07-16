# first encode all images then pick target t 
# use siamese encoder, deploy heads for different target
# in online training, freeze one on target, update one on current pos
# update params with data colllected in interaction
import torch
from torchvision.models import resnet18, ResNet18_Weights 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class Encoder(nn.Module):
    def __init__(self, z_dim = 128 ,pretrained = True, trainable = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained  else None
        
        
        net = resnet18(weights = weights) 
        
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,z_dim)

        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False # no gradient back propagation

    def forward(self,x):
        
        feature = self.backbone(x) # [B*T,3,84,84] -> [B,512,3,3] resnet 18
        feature = self.pool(feature).flatten(1) # [B*T,512,3,3] -> [B*T,512,1,1] -> [B*T,512]
        z = self.fc(feature) # [B,128] final vector in bottleneck

        return z

class Decoder(nn.Module):
    def __init__(self, z_dim = 128):
        super().__init__()  
        #从向量映射到一个小 feature map，再逐步上采样
        self.fc = nn.Linear(z_dim,256*6*6)
        self.deconv = nn.Sequential(
          nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU(inplace=True),
          nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(inplace = True),
          nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(inplace=True),
          nn.ConvTranspose2d(32,3,4,2,1),
          nn.Sigmoid()
        ) 

    def forward(self,z):  # z 是 bottleneck 向量
         h = self.fc(z).view(z.size(0),256,6,6) # [B,512] -> [B,256,3,3]
         x_hat = self.deconv(h) #[B,256,3,3] ->  [B,3,96,96]
         x_hat = F.interpolate(x_hat,size = (84,84), mode= "bilinear",align_corners=False)
         return x_hat

class ResNetAE(nn.Module):
    def __init__(self,z_dim = 128,pretrained = True, encoder_trainable = False):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim,pretrained=pretrained,trainable=encoder_trainable)
        self.decoder = Decoder(z_dim=z_dim)
    

    def forward(self,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat


def make_mlp(in_dim, hidden=(256,128), out_dim=3): # in_dim:128 according to encoder
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(True)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class SiameseEncoder(nn.Module):
    def __init__(self, emb_dim, state_dim: int, goal_num: int,
                 freeze_encoder: bool = False, hidden=(256,128)):
        super().__init__()
        self.emb_dim = emb_dim
        self.goal_num = goal_num

        self.encoder = Encoder(z_dim=emb_dim)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        in_dim = emb_dim  + state_dim

        # 每个 goal 都创建一个“新的” head（参数互不共享）
        self.heads = nn.ModuleList([
            make_mlp(in_dim, hidden=hidden, out_dim=3)
            for _ in range(goal_num) # 0,1 1,2
        ])

    def sample_t_cur(self,t_goal:Dict[str, int], T:int, M:int, B:int, avoid_goal_frame:bool= True): # t_goal is from target_selector 
         # M: 采样M个
         # 1) 按照顺序排序
         items = []
         for k,v in t_goal.items():
            if isinstance(k,str) and k.startswith("label_"):
               lab = int(k.split("_")[1])
            else:
               lab = int(k)
            items.append((lab,int(v)))
         items.sort(key=lambda x: x[0])#按照key的大小排序，已规定顺序
         
         # 2) 做成 tensor 选 encoder head
         goal_id = torch.tensor([lab for lab,_ in items], device=self.device, dtype=torch.long)  # [G]
         t_goal_t  = torch.tensor([t for _,t in items],     device=self.device, dtype=torch.long)   # [G]
         # 如果你希望按时间顺序分段（更合理），就用 t 排序：
         t_sorted, order = torch.sort(t_goal_t)          # [G]
         goal_id_sorted = goal_id[order]                 # [G] 与 t_sorted 对齐
         G = t_sorted.numel() #得到 goal的数量        
         # 3)构造每个 goal 的采样窗口 [start,end]
         starts = torch.zeros(G, device= self.device,dtype=torch.long)
         ends = torch.full((G,),T,device=self.device,dtype=torch.long) #不是说每个 goal 的窗口都应该是 T，只是初始化时用 T 做默认上界最安全。

         if G == 1: # 只有一个goal
            starts[0],ends[0] = 0,T
         else: #有多个 goal
             starts[0] = 0
             ends[0] = int(t_sorted[1].item()) #第一个 goal
             for g in range(1,G-1):# range(start,stop) 或者 range(stop) 不包括 stop 比如 range(0,3) = 0,1,2
                starts[g] = int(t_sorted[g-1].item())
                ends[g] = int(t_sorted[g+1].item()) # 中间的 goal
             starts[G-1] = int(t_sorted[G-2].item()) #最后一个 goal
             ends[G-1] = T
        # clamp + 处理退化窗口

         starts = torch.clamp(starts,0,T-1)
         ends = torch.clamp(ends,1,T)
         bad = starts >= ends
         if bad.any():
            starts[bad] = 0
            ends[bad] = T
        # 4) 采样 t_cur: [B,G,M]
         t_cur = torch.empty(B,G,M, device=self.device,dtype=torch.long)
         for g in range(G):
              low = int(starts[g].item())
              high = int(ends[g].item())  # exclusive
              cur = torch.randint(low, high, (B, M), device=self.device)

        # 可选：避免采到 t_goal 本身（只有窗口长度>1 才有意义）
              if avoid_goal_frame and (high - low) > 1:
                tg = int(t_sorted[g].item())
                mask = (cur == tg)
                while mask.any():
                    cur[mask] = torch.randint(low, high, (mask.sum().item(),), device=self.device)
                    mask = (cur == tg)

              t_cur[:, g, :] = cur

    # 5) 命名：t_cur_for_(goal_idx)
         t_cur_for_goal = {g+1: t_cur[:, g, :] for g in range(G)} # {int: [B,1,M]}

    # 如果还想按 label 命名（可选）
    # t_cur_for_label = {f"t_cur_for_label_{int(lab.item()+1)}": t_cur[:, g, :]
    #                    for g, lab in enumerate(goal_id_sorted)}
    
         return t_cur, t_cur_for_goal
    
    def get_true_relative_dist(self,
         pos: torch.Tensor,                 # [B,T,3]
             
         t_goal: Dict[int, int],            # {k: int}
     ) -> Dict[int, torch.Tensor]:
         B, T, D = pos.shape
         assert D == 3

         dist_dict: Dict[int, torch.Tensor] = {}

         for k, tg in t_goal.items():
             tg = int(tg)
             goal_pos = pos[:, tg, :]                 # [B,3]

             t = self.t_cur_for_goal[k]
             if t.dim() == 3:                         # [B,1,M]
                 t = t.squeeze(1)
             t = t.long().clamp(0, T-1)               # [B,M]
             idx = t.unsqueeze(-1).expand(-1, -1, D)  # [B,M,3]

             cur_pos = pos.gather(dim=1, index=idx)   # [B,M,3]
             delta = goal_pos[:, None, :] - cur_pos   # [B,M,3]
             dist_dict[k] = delta                     # convert time shape
             #dist_dict[k] = torch.norm(delta, dim=-1) # ✅ [B,M] 相对距离 不要相对距离

         return dist_dict


    def encode_seq(self,imgs)->torch.Tensor: # B,T,C,H,W->B*T,C,H,W for online only C,H,W
          B,T,C,H,W = imgs.shape
          x = imgs.reshape(B*T,C,H,W).contiguous() #[B*T,3,H,W]
          z = self.encoder(x)
          return z.view(B,T,-1)#[B,T,z_dim]
    


    def forward(self, state:torch.Tensor, imgs: torch.Tensor, t_goal:dict)-> dict: # t_cur:{"label":[B,1,M]}  t_goal:{"label":int}
        # goal_id: Python int, 0 ~ goal_num-1 
        # state: [B,T,3]
        B,T,C,H,W = imgs.shape
        cur_state_dict:dict = {} # {int:[B,M,D]}
        goal_state_dict:dict = {}# {int:[B,1,D] or [B,D]}
        # encode whole sequence -> [B,T,H,W,C]->[B,T,D]
        z_seq = self.encode_seq(imgs)
        z_cur: Dict[int,torch.Tensor] = {}
        z_goal: Dict[int,torch.Tensor] = {}
        self.t_cur_for_goal:dict
        t_cur, self.t_cur_for_goal = self.sample_t_cur(t_goal=t_goal,T=T,M=15,B=B)   
        
        for k,v in self.t_cur_for_goal.items(): # key: 1,2
         assert isinstance(v, torch.Tensor)
         assert isinstance(k,int)
         t = self.t_cur_for_goal[k]#[B,M]

         t= t.squeeze(1).long()#[B,1,M]->[B,M]
         idx = t.unsqueeze(-1).expand(-1, -1, z_seq.size(-1))   # [B, M, D]
         idx_state = t.unsqueeze(-1).expand(-1,-1,state.size(-1)) #[B,M]->[B,M,3]
         z_cur[k] = z_seq.gather(dim=1, index=idx)               # {int:[B, M, D]}
         cur_state_dict[k] = state.gather(dim=1,index=idx_state)
    
        D = z_seq.shape[-1]
        dist = {}
        z_diff = {}
        feature:dict = {}
        state_diff = {}
        for k,v in t_goal.items():
         tg = int(t_goal[k])                 # goal time (int)
         goal_state_dict[k]= state[:,tg,:] # [B,T_goal,3]
         z_goal[k] = z_seq[:, tg, :]            # [B, D] 
         z_diff[k] = z_goal[k][:, None, :] - z_cur[k]     # [B, M, D]
         state_diff[k] = cur_state_dict[k]- goal_state_dict[k][:,None,:]
         feature[k] = torch.cat([z_diff[k],state_diff[k]],dim=-1) #[B,M,D+S]
         dist[k] = self.heads[k-1](feature[k])

        return dist            
        
