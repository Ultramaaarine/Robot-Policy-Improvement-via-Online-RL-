import os
import time
import subprocess

seeds = [1, 10, 20, 30, 40]

max_parallel = 2   # 想同时跑5个就写5
gpu_id = "0"       # 只有一张GPU就用0

running = []

for seed in seeds:
    cmd = [
        "python",
        "sac_diffusion",
        "workspaces",
        "ddpm_critic_training.py",   # 这里换成你的训练文件名
        f"seed={seed}",
        f"model_save_dir=outputs/checkpoints/seed_{seed}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print("Running:", " ".join(cmd))

    p = subprocess.Popen(cmd, env=env)
    running.append(p)

    while len(running) >= max_parallel:
        running = [proc for proc in running if proc.poll() is None]
        time.sleep(10)

for p in running:
    p.wait()

print("All seeds finished.")