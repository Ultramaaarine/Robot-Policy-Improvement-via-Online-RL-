import numpy as np

def main():
 d = np.load("diffusion_trace.npz")["data"]   # shape [N,7]
 roll = d[:,0].astype(int)
 step = d[:,1].astype(int)
 std  = d[:,4]

# 看 rollout_step=12 的 std 随 step 的变化 
 mask = roll == 12
 idx = np.argsort(step[mask])
 print(np.stack([step[mask][idx], std[mask][idx]], axis=1)[:10])   # 前10个点
 print(np.stack([step[mask][idx], std[mask][idx]], axis=1)[-10:])  # 后10个点