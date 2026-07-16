# 1 env
# 2 dataset
# 3 reset -> step until done

import os
import argparse
from pathlib import Path
from sac_diffusion.datasets.calvin_critic_offline_dataset import CalvinCriticOfflineDataset
import numpy as np
from torch.utils.data import DataLoader
import pygame
import matplotlib.pyplot as plt
import hydra
from sac_diffusion.models.normalizer import Normalizer

@hydra.main(version_base=None, config_path="../../config", config_name="mark_target_config")
def main(cfg):
    dataset = hydra.utils.instantiate(cfg.datamodule.training_dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    normalizer = Normalizer() 
    env = hydra.utils.instantiate(cfg.env.calvin_env.env)
    obs = env.reset() # obs: dict

    pygame.init()
    pygame.key.set_repeat(0)
    screen = pygame.display.set_mode((400,300))
    pygame.display.set_caption("Step-by-step marker (Enter to step)")
    font = pygame.font.SysFont("monospace",18)
    clock = pygame.time.Clock()

    for idx,batch in enumerate(dataloader):
        rgb_obs = batch["rgb_obs"]
        lowdim_obs = batch["state"]
        
        action = batch["action"]# normalized
        depth_obs = batch["depth_obs"]
        #scene_obs = batch["scene_obs"]
  
    action_param = dataset.get_normalize_params(param_type="action_params")
    #action_norm =normalizer.normalize(action,action_param)
    action_reconstructed = normalizer.unnormalize(action,action_param)
    action_reconstructed = action_reconstructed.detach().cpu().numpy()
    print("raw_action:", action[0,:,:])
    print("reconstructed_action:", action_reconstructed[0,:,:])
    ori = np.zeros(3)
    gripper_action = [1]
    T = action.shape[1]
    t_ptr =  0
    step_once = False
    running = True
    paused = True
    print("\nControls:")
    print(" ENTER: step once")
    print(" SPACE: play/pause")
    print("R: reset env + rewind to t = 0")
    print(" ESC/Q: quit\n")

    while running:
      step_once = False
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False
         elif event.type == pygame.KEYDOWN:

            if event.key in (pygame.K_ESCAPE, pygame.K_q):
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
      if not running:
         break 
      
      #UI
      screen.fill((20,20,20))
      status = f"t={t_ptr}/{T-1} paused={paused}"
      screen.blit(font.render(status,True, (230,230,230)),(10,10))
      screen.blit(font.render("ENTER = step SPACE = run/pause R = reset",True,(180,180,180)),(10,45))
      pygame.display.flip()
      
      do_step = (not paused) or step_once

      if do_step:
         step_once = False

         if t_ptr >= T:
            paused = True
            print("[END] reached the end of the trajectory")
         else:
            a_in = action_reconstructed[0,t_ptr,:].astype(np.float32) # expected action: velocity

            if a_in.shape[0] == 3:
               env_action = np.concatenate([a_in,ori,gripper_action],axis=0)
            else:
               env_action = a_in
            next_obs,reward,done,info = env.step(env_action)

            obs = next_obs
            print(f"[STEP] t={t_ptr} reward={reward} done={done}")
            t_ptr += 1

            if done:
               print("[DONE] env returned done = True, reset (R)")
               paused = True
      clock.tick(30)
    pygame.quit()   

if __name__ == "__main__":
    main()

    
     
