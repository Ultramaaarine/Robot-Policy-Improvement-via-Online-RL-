# Training workspace for offline diffusion-policy learning, critic/value learning,
# online rollout collection, advantage-weighted policy improvement, and logging.
# The workflow combines offline demonstration training with online critic-guided refinement.

import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.nn.functional as F
import random
import tqdm
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
import copy
import csv

from sac_diffusion.workspaces.base_workspace import BaseWorkspace
from sac_diffusion.policy.hybrid_transformer_policy import HybridPolicy
from sac_diffusion.datasets.calvin_critic_offline_dataset import CalvinCriticOfflineDataset
from sac_diffusion.models.exponential_moving_average import EMAModel
from sac_diffusion.models.normalizer import Normalizer
from sac_diffusion.utils.build_input_batch import build_sliding_window_batch
from sac_diffusion.utils.fit_cond_normalizer import fit_cond_normalizer
from sac_diffusion.utils.optimizer_to import optimizer_to
from sac_diffusion.utils.rollout import rollout
from sac_diffusion.models.lr_scheduler import get_scheduler
from sac_diffusion.utils.env_maker import make_env
from sac_diffusion.utils.save_array_stats import save_array_stats, save_trace_by_rollout_step
from sac_diffusion.models.replay_buffer import ReplayBuffer
from sac_diffusion.models.SAC_Model.critic import DoubleQNetwork
from sac_diffusion.models.SAC_Model.V_net import ValueNet
from sac_diffusion.utils.mix_data import mix_data
from sac_diffusion.utils.condition_builder import assemble_conditions

logger = logging.getLogger(__name__)


# Main training workspace that coordinates policy, critic, value network,
# replay buffer, datasets, rollout environment, optimization, and evaluation.
class TrainDiffusionWorkspace(BaseWorkspace):
    def __init__(
        self,
        cfg: OmegaConf,
        output_dir=Path(__file__).parent.parent.parent.joinpath("outputs")
    ):
        super().__init__(cfg, output_dir=output_dir)

        # Set deterministic random seeds for reproducibility.
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.output_dir = output_dir
        self.global_step = 0
        self.epoch = 0
        self._offline_iter = None
        
        # Instantiate diffusion policy, critic, value network, and optional EMA model.
        # ---------- models ----------
        self.diffusion_model: HybridPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.diffusion_model)

        self.critic_network: DoubleQNetwork = hydra.utils.instantiate(cfg.critic_network)
        self.v_net: ValueNet = hydra.utils.instantiate(cfg.value_network)

        self.normalizer = Normalizer()

        # Optimizers for diffusion policy, critic, and value network.
        # ---------- optimizers ----------
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.diffusion_model.parameters()
        )
        self.critic_optimizer = hydra.utils.instantiate(
            cfg.critic_optimizer, params=self.critic_network.parameters()
        )
        self.v_net_optimizer = hydra.utils.instantiate(
            cfg.v_net_optimizer, params=self.v_net.parameters()
        )

        # Replay buffer used to store online rollout sequences.
        # ---------- replay ----------
        self.replaybuffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer)

        # Load offline training and validation datasets and normalization parameters.
        # ---------- datasets ----------
        self.training_dataset: CalvinCriticOfflineDataset = hydra.utils.instantiate(
            cfg.datamodule.training_dataset
        )
        self.action_norm_params = self.training_dataset.get_normalize_params("action_params")
        self.state_norm_params = self.training_dataset.get_normalize_params("state_params")

        self.validation_dataset = hydra.utils.instantiate(
            cfg.datamodule.val_dataset,
            action_params=self.action_norm_params,
            state_params=self.state_norm_params
        )

        assert isinstance(self.training_dataset, CalvinCriticOfflineDataset)
        assert isinstance(self.validation_dataset, CalvinCriticOfflineDataset)

        self.batch_size = int(getattr(cfg.training, "batch_size", 8))

        # Build dataloaders for offline diffusion and critic training.
        self.training_dataloader = DataLoader(
            self.training_dataset,
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        )
        # Create rollout environment and read the environment-provided goal position.
        self.rollout_env = make_env(cfg=self.cfg.env, skill=self.cfg.skill)
