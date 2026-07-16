from omegaconf import DictConfig
from sac_diffusion.envs.calvin.skill_env import CalvinSkillEnv

try:
    from rl_tasks import ENV_TYPES
except ModuleNotFoundError:
    ENV_TYPES = None

def make_env(cfg: DictConfig,skill = None, start_point = None):
    if cfg.env_name == "calvin":
        env = CalvinSkillEnv(cfg,skill,start_point)
    elif cfg.env_name  == "bullet":
        env = ENV_TYPES[cfg.bullet.type](cfg.bullet, show_gui = cfg.show_gui)
    else:
        raise NotImplementedError(f"Environment {cfg.env_name} not implemented")
    return env

if __name__ == "__main__":
    make_env()