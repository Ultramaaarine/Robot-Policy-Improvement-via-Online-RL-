import os
import json
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import hydra
import pygame


def to_numpy(x):

    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception :
        pass
    return x



@hydra.main(version_base=None,config_name="mark_target_config",config_path="../../config")
def main(cfg):
    dataset = hydra.utils.instantiate(cfg.datamodule.training_dataset)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
    env = hydra.utils.instantiate(cfg.env.calvin_env.env)
    # ==== save_dir ====
    save_dir = Path("annotations_time")
    save_dir.mkdir(parents=True,exist_ok=True)
    # ==== pygame UI ====

    pygame.init()
    screen = pygame.display.set_mode((520,220))
    pygame.display.set_caption("Time/Phase Annotation Tool")
    font = pygame.font.SysFont("monospace",18)
    clock = pygame.time.Clock()

    print("\nControls:")
    print(" ENTER: step once")
    print(" SPACE: play/pause")
    print(" R: reset env + rewind t=0")
    print(" 1-9: set current label")
    print(" K: mark annotation at current t")
    print(" S: save current episode annotations")
    print("N: save + next episode")
    print(" ESC/Q: quit\n")

    # ==== state ====
    paused = True
    step_once = False
    cur_label = 1
    episode_idx = 0

    ep_actions = None
    ep_len = None

    t_ptr = 0

    ann_list = []

    def load_episode(batch):
        nonlocal ep_actions, ep_len, t_ptr, ann_list
        actions = batch["action"]  # [1,T,D]
        actions = to_numpy(actions)[0]  # [T,D]
        ep_actions = actions.astype(np.float32)
        print(f"ep_actions shape: {ep_actions.shape}") # [63,3]
        ep_len = ep_actions.shape[0]
        t_ptr = 0
        ann_list = []
    
    def save_annotations(ep_id):
        out = {
            "episode_idx":int(ep_id),
            "episode_len":int(ep_len),
            "annotations":ann_list,
        }

        out_path = save_dir / f"ep_{ep_id:05d}.json"
        with open(out_path, "w",encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii= False, indent = 2)
        print(f"[SAVE]{out_path} (num_marks={len(ann_list)})")
    def go_to_next_episode(save_first = True):
         nonlocal batch, episode_idx, obs,paused, t_ptr
         if save_first and len(ann_list) >0:
                save_annotations(episode_idx)
         try:
                episode_idx, batch = next(iterator)
         except StopIteration:
                print("[END] reached the end of dataset")
                return False
         
         load_episode(batch)
         obs = env.reset()
         paused = True
         t_ptr = 0
         print(f"[EP] load episode_idx = {episode_idx}")
         return True
    #==== main loop ====
    
    iterator = enumerate(dataloader)
    episode_idx, batch = next(iterator)
    load_episode(batch)
    obs = env.reset()

    running = True
    while running:
      step_once = False

      # ---events---
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
             running = False
         elif event.type == pygame.KEYDOWN:
             if  event.key in (pygame.K_ESCAPE, pygame.K_q):
                 running = False
             elif event.key == pygame.K_RETURN:
                 step_once = True
                 paused = True
             elif event.key == pygame.K_SPACE:
                 paused = not paused    

             elif event.key == pygame.K_r:
                 obs = env.reset()
                 t_ptr = 0
                 paused = True
                 print("[RESET] rewind to t = 0")
             # mark annotation
             elif event.key == pygame.K_k:
               if ep_len is None:
                   continue
               progress = float(t_ptr / max(ep_len-1,1))
               ann = {
                    "episode_idx": int(episode_idx),
                    "t": int(t_ptr),
                    "progress":progress,
                    "label": int(cur_label),
                }
               ann_list.append(ann)
               print(f"[MARK] ep={episode_idx} t = {t_ptr}/{ep_len-1} p = {progress:.3f} label = {cur_label}")
             elif event.key == pygame.K_s:
               save_annotations(episode_idx)
             elif event.key == pygame.K_n:
                 ok = go_to_next_episode(save_first=True)
                 if not ok:
                     running = False
         # set label with number keys
             elif pygame.K_1<= event.key <= pygame.K_9:
                 cur_label = event.key - pygame.K_0
                 print(f"[LABEL] curent label = {cur_label}")
      screen.fill((20,20,20))
      status1 = f"ep={episode_idx} t={t_ptr}/{(ep_len-1) if ep_len else -1} paused = {paused} "
      status2 = f"label = {cur_label} marks = {len(ann_list)}"
      hint1 = "ENTER step | SPACE  run/pause | R reset | 1-9 label"
      hint2 = " K mark | S save | ESC/Q quit"
      screen.blit(font.render(status1,True,(230,230,230)),(10,10))
      screen.blit(font.render(status2,True,(230,230,230)),(10,40))
      screen.blit(font.render(hint1,True,(180,180,180)),(10,90))
      screen.blit(font.render(hint2,True,(180,180,180)),(10,120))
      pygame.display.flip()
    # ==== step ====
      do_step = (not paused) or step_once
      if do_step and ep_actions is not None:
          if t_ptr >= ep_len:
            paused = True
            print("[END] reached the end of the episode")
          else:
            a = ep_actions[t_ptr,:] # 注意 ep_action 是一个轨迹 [t_ptr,:]只说明是拿了一个timestep [3,]
            a = to_numpy(a)
            ori= np.zeros(3)
            gripper_action = [1]
            if a.shape[0] == 3:
                print(f"a has shape {a.shape}")
                a = np.concatenate([a, ori, gripper_action],axis=0)
            elif a.shape == 6:
                pass
            next_obs, reward, done, info = env.step(a) # obs 包括所有的obs 是一个 dict
            t_ptr +=1
           
            # 末尾自动暂停
            if t_ptr >= ep_len:
                paused = True
                print("[END] reached the end of the episode")
      clock.tick(30)

      if not running:
        if len(ann_list) > 0:
            save_annotations(episode_idx)
        break
pygame.quit()

if __name__ == "__main__":
    main()


