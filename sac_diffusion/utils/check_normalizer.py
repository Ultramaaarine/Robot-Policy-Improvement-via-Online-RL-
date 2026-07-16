# Normalizer works correctly

import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from sac_diffusion.models.normalizer import Normalizer

@hydra.main(version_base=None, config_path="../../config", config_name="mark_target_config")
def main(cfg):
    # 1) 用 normalized=False 的 dataset 拿 raw_action
    cfg.datamodule.training_dataset.skill.normalized = False  # 或者你在配置里直接关掉
    dataset_raw = hydra.utils.instantiate(cfg.datamodule.training_dataset)
    raw = dataset_raw[0]["action"].detach().cpu().numpy()  # shape [T-1, D] 或 [D]
    print("Raw action sample :", raw)
    params = dataset_raw.get_normalize_params("action_params")
    print("Normalization params:", params)
    norm = Normalizer()

    # 2) round-trip
    a_norm = norm.normalize(raw, params).detach().cpu().numpy()
    a_rt   = norm.unnormalize(a_norm, params).detach().cpu().numpy()
    print("a_rt",a_rt)
    # 3) 误差统计
    err = a_rt - raw
    print("raw shape:", raw.shape)
    print("max_abs_err:", np.max(np.abs(err)))
    print("mean_abs_err:", np.mean(np.abs(err)))
    print("rmse:", np.sqrt(np.mean(err**2)))

    # 每个维度统计（最后一维是 D）
    err_d = np.reshape(err, (-1, err.shape[-1]))
    print("per-dim max_abs_err:", np.max(np.abs(err_d), axis=0))
    print("per-dim mean_abs_err:", np.mean(np.abs(err_d), axis=0))

if __name__ == "__main__":
    main()
