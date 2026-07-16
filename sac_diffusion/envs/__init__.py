from gym.envs.registration import register

register(id="skill-env", entry_point="sac_diffusion.envs.calvin:CalvinSkillEnv", max_episode_steps=64)
