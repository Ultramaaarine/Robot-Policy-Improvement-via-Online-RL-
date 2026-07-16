import numpy as np
from pathlib import Path
import json    

def save_array_stats(arr: np.ndarray, out_dir: Path, prefix: str):
        """
        保存原始数据 + stats(json/txt)
        arr: (N, D) or (N, T, D) -> 会展平到 (N*, D)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        if arr.ndim == 3:
            arr2 = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            arr2 = arr
        else:
            raise ValueError(f"arr.ndim must be 2 or 3, got {arr.ndim}")

        stats = {
            "shape": list(arr2.shape),
            "mean": arr2.mean(axis=0).tolist(),
            "std": arr2.std(axis=0).tolist(),
            "min": arr2.min(axis=0).tolist(),
            "max": arr2.max(axis=0).tolist(),
        }

        np.savez(out_dir / f"{prefix}.npz", data=arr2)

        with open(out_dir / f"{prefix}.json", "w") as f:
            json.dump(stats, f, indent=2)
        with open(out_dir / f"{prefix}.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}:{v}\n")
        return stats

def save_trace_by_rollout_step(trace_arr: np.ndarray, out_dir: Path, prefix: str):
        """
        trace_arr: [N, 7] columns =
        [rollout_step, denoise_step_i, timestep, mean, std, min, max]
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        assert trace_arr.ndim == 2 and trace_arr.shape[1] == 11, f"bad trace_arr shape: {trace_arr.shape}"

        rollout_steps = np.unique(trace_arr[:, 0]).astype(int)
        for rs in rollout_steps:
            part = trace_arr[trace_arr[:, 0] == rs]  # [Mi,7]
        # 你想要“看 denoise”，最好不要把第0~2列也算进 stats（它们不是随机变量）
        # 所以这里只保存第3~6列 stats（mean/std/min/max），但原始数据仍然完整保存在 npz 里
            save_array_stats(part, out_dir, prefix=f"{prefix}_rs{rs:03d}")

        # 另外再存一个更可读的 txt/csv（可选）
            csv_path = out_dir / f"{prefix}_rs{rs:03d}.csv"
            np.savetxt(
                csv_path,
                part,
                delimiter=",",
                header="rollout_step,denoise_i,timestep,x_mean,x_std,x_min,x_max,x0_mean,x0_std,x0_min,x0_max",
                comments=""
            )