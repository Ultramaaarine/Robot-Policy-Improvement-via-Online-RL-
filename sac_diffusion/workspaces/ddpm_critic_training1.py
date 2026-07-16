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


class TrainDiffusionWorkspace(BaseWorkspace):
    def __init__(
        self,
        cfg: OmegaConf,
        output_dir=Path(__file__).parent.parent.parent.joinpath("outputs")
    ):
        super().__init__(cfg, output_dir=output_dir)

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
        
        # ---------- models ----------
        self.diffusion_model: HybridPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.diffusion_model)

        self.critic_network: DoubleQNetwork = hydra.utils.instantiate(cfg.critic_network)
        self.v_net: ValueNet = hydra.utils.instantiate(cfg.value_network)

        self.normalizer = Normalizer()

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

        # ---------- replay ----------
        self.replaybuffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer)

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
        self.rollout_env = make_env(cfg=self.cfg.env, skill=self.cfg.skill)
        self.rollout_env.reset()
        self.env_goal_pos = self.rollout_env.gt_keypoint
        self.improve_step = 0
        self.critic_step = 0

    def run(self):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(__file__).parent.parent.parent.joinpath(
            "diffusion model offline training log", run_id
        )
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self._training_diffusion()

    def _save_dataset_action_stats(self, training_dataset, out_dir: Path):
        if hasattr(training_dataset, "action"):
            a = training_dataset.action
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            a = np.asarray(a)
            return save_array_stats(a, out_dir, prefix="dataset_actions")

        all_actions = []
        for i in range(len(training_dataset)):
            sample = training_dataset[i]
            act = sample["action"]
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
            all_actions.append(act)
        a = np.stack(all_actions, axis=0)
        return save_array_stats(a, out_dir, prefix="dataset_actions")

    def _compute_shaped_reward(self, reward, demo_dist, goal_dist):
        w_env = float(getattr(self.cfg.training, "reward_w_env", 0.3))
        w_demo = float(getattr(self.cfg.training, "reward_w_demo", 0.3))
        w_goal = float(getattr(self.cfg.training, "reward_w_goal", 0.4))
        shaped_reward = torch.tensor(w_env * reward - w_demo * demo_dist - w_goal * goal_dist,dtype=torch.float32,device=self.device)
        return shaped_reward

    def _critic_train_step_from_window(self, critic_window):
        cond_t = critic_window["cond_t"]
        cond = critic_window["cond_t"]["cond"]

        obs = cond["pos"].to(self.device)
        obs_idx = cond_t["obs_idx"]
        critic_action = critic_window["act_window"].to(self.device,dtype = torch.float32)
        reward = critic_window["reward_window"].to(self.device, dtype = torch.float32)
        done = critic_window["done_window"].to(self.device)
        next_obs = cond["next_pos"].to(self.device)

        demo_dist = ((cond["stage_goal_pos"].to(self.device) - obs) ** 2).sum(dim=-1, keepdim=True) # zero in offline, in online, build this with obs_hist and eef pos after step env eef pos1,eef pos2 from deque form a seq, compare  with obs pos seq from demo
        goal_dist = cond["goal_dist"].to(self.device)


        shaped_reward = self._compute_shaped_reward(reward, demo_dist, goal_dist)
   
        n_cond, next_n_cond, visual_cond, action_mask, stage_out, next_stage_out = assemble_conditions(cond_t=cond_t,
                                                                            cond_norm_params=self.cond_norm_param_dict,
                                                                            batch_size=self.batch_size, device=self.device, error_scale=1000,
                                                                            pull_start_t=36, 
                                                                            build_action_mask=False)

        with torch.no_grad():
            v_next = self.v_net(visual_obs=None, low_dim_obs=next_n_cond)
            gamma = self.cfg.training.critic_gamma
            gamma = torch.tensor(float(gamma),device=self.device,dtype = torch.float32) # do not directly read from cfg, convert it to torch.float32 
            one = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            q_target = shaped_reward + gamma * (one - done) * v_next

        q1, q2 = self.critic_network(state=n_cond, visual_obs=None, action=critic_action) # seq q value [B,T,1]

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()
        step_log = {
            #"obs_idx":obs_idx,
            "critic_loss":critic_loss,
            "step_reward_q1":q1, # [B,T,1]
            "step_reward_q2":q2,
            "mean_q1":torch.mean(q1,dim=1), # seq_wise mean [B,1]
            "mean_q2":torch.mean(q2,dim = 1)
        }
        logger.info(step_log)
        with torch.no_grad():
            q1_detach, q2_detach = self.critic_network(state=n_cond, visual_obs=None, action=critic_action)
            q_detach = torch.min(q1_detach, q2_detach)

        v_value = self.v_net(visual_obs=None, low_dim_obs=n_cond)
        diff = q_detach - v_value

        tau = float(self.cfg.training.expectile_tau)
        weight = torch.where(
            diff > 0,
            torch.full_like(diff, tau),
            torch.full_like(diff, 1.0 - tau),
        )
        v_loss = torch.mean(weight * (diff ** 2))

        self.v_net_optimizer.zero_grad()
        v_loss.backward()
        self.v_net_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "v_loss": float(v_loss.item()),
            "q_mean": float(q_detach.mean().item()),
            "v_mean": float(v_value.mean().item()),
            "reward_mean": float(shaped_reward.mean().item()),
            "q_target_mean": float(q_target.mean().item()),
            "q_target_std": float(q_target.std().item()),
            "q_target_max": float(q_target.max().item()),
            "q_target_min": float(q_target.min().item()),
            "reward_std": float(shaped_reward.std().item()),
            "reward_max": float(shaped_reward.max().item()),
            "reward_min": float(shaped_reward.min().item()),
        }

    def _training_diffusion(self):
        self.critic_network.eval()
        self.v_net.eval()

        cfg = copy.deepcopy(self.cfg)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(tag)s] %(message)s"
        )

        print(f"length of training_dataset: {len(self.training_dataset)}")

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(self.training_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        self.cond_norm_param_dict = fit_cond_normalizer(
            training_dataloader=self.training_dataloader,
            env_goal_pos=self.rollout_env.gt_keypoint,
            cfg=self.cfg,
        )

        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        self.diffusion_model = self.diffusion_model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
        self.optimizer = optimizer_to(self.optimizer, device=self.device)

        analysis_dir = Path(self.log_dir) / "action_analysis"
        ds_stats = self._save_dataset_action_stats(self.training_dataset, analysis_dir)

        np.random.seed(cfg.seed + 1234)

        model_path = Path(self.cfg.model_save_dir)
        os.makedirs(model_path, exist_ok=True)

        for epoch_idx in range(cfg.training.num_epochs):
            train_losses = []
            self.diffusion_model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with tqdm.tqdm(
                self.training_dataloader,
                desc=f"Training Epoch {epoch_idx + 1}",
                mininterval=1.0
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    window = build_sliding_window_batch(
                        cfg=self.cfg,
                        batch=batch,
                        env_goal_pos=self.env_goal_pos,
                        horizon=int(getattr(self.cfg.training, "horizon", 16)),
                    )

                    action = window["act_window"].to(self.device)
                    cond = window["cond_t"]

                    raw_loss = self.diffusion_model.compute_loss(
                        action,
                        cond,
                        action_norm_params=self.action_norm_params,
                        cond_norm_params=self.cond_norm_param_dict,
                        improve=False
                    )

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    raw_loss_cpu = float(raw_loss.detach().item())

                    is_last_batch = (batch_idx + 1) == len(self.training_dataloader)
                    do_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every) == 0 or is_last_batch

                    if do_step:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.diffusion_model)

                        self.global_step += 1
                        self.writer.add_scalar("training/loss", raw_loss_cpu, global_step=epoch_idx+1)
                        self.writer.add_scalar("training/lr", lr_scheduler.get_last_lr()[0], epoch_idx+1)

                        train_losses.append(raw_loss_cpu)

                        logger.info(
                            "step=%d epoch=%d loss=%.6f lr=%.6f",
                            self.global_step,
                            self.epoch,
                            raw_loss_cpu,
                            lr_scheduler.get_last_lr()[0],
                            extra={"tag": "Diffusion training"}
                        )

                    # if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                    #     break

            train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
            self.writer.add_scalar("training/epoch_loss", train_loss, global_step=epoch_idx+1)

            self.epoch += 1

            # ---------- validation ----------
            if (self.epoch % cfg.training.val_every) == 0:
                self.diffusion_model.eval()
                with torch.no_grad():
                    t_list = [8, 16, 24, 32, 40, 48, 56]
                    val_losses = []

                    with tqdm.tqdm(
                        self.validation_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=1.0
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            for t in t_list:
                                val_window = build_sliding_window_batch(
                                    cfg=self.cfg,
                                    batch=batch,
                                    t=t,
                                    env_goal_pos=self.env_goal_pos,
                                    horizon=int(getattr(self.cfg.training, "horizon", 16))
                                )
                                val_action = val_window["act_window"].to(self.device)
                                val_cond = val_window["cond_t"]

                                val_loss = self.diffusion_model.compute_loss(
                                    action=val_action,
                                    cond=val_cond,
                                    action_norm_params=self.action_norm_params,
                                    cond_norm_params=self.cond_norm_param_dict,
                                    improve=False,
                                )
                                val_losses.append(val_loss.item())

                            # if (cfg.training.max_val_steps is not None
                            #         and batch_idx >= (cfg.training.max_val_steps - 1)):
                            #     break

                    if len(val_losses) > 0:
                        val_loss = float(np.mean(val_losses))
                        self.writer.add_scalar("validation/loss", val_loss, global_step=epoch_idx+1)

                self.diffusion_model.train()

            # ---------- save checkpoint ----------
            if (self.epoch % cfg.training.save_every_epoch) == 0:
                ckpt = {
                    "model": self.diffusion_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "lr_scheduler": lr_scheduler.state_dict()
                }
                if cfg.training.use_ema:
                    ckpt["ema_model"] = self.ema_model.state_dict()

                torch.save(ckpt, model_path / f"epoch_{self.epoch:04d}.pt")
            rollout_after = getattr(self.cfg.training,"rollout_after")
            rollout_every = getattr(self.cfg.training,"rollout_every")
            if self.epoch >= rollout_after and self.epoch % rollout_every == 0:
                offline_rollout_result = self._rollout(select_with_critic=False)
        # ============================================================
        # After offline diffusion training: always rollout + improve
        # ============================================================
        self.diffusion_model.eval()
        self.critic_network.eval()
        self.v_net.eval()

        for online_cycle in range(1, self.cfg.training.max_online_cycle + 1):

            logger.info(
                "online_cycle=%d start rollout",
                online_cycle,
                extra={"tag": "Online loop"}
    )

            # 1. rollout collect online data
            
            rollout_results = self._rollout(select_with_critic=False)
            


            # 2. train critic with online + offline mixed data
            self._training_critic(mode="online_training")

            # 3. improve diffusion policy using advantage weight
            self._policy_improvement(num_steps=200)

            # 4. rollout again after improvement
            rollout_results_after_improve = self._rollout(select_with_critic=True)
                            

    def _training_critic(self, mode: str):
        self.critic_network = self.critic_network.to(self.device)
        self.v_net = self.v_net.to(self.device)

        self.critic_network.train()
        self.v_net.train()

        self.critic_optimizer = optimizer_to(self.critic_optimizer, device=self.device)
        self.v_net_optimizer = optimizer_to(self.v_net_optimizer, device=self.device)

        horizon = int(getattr(self.cfg.training, "horizon", 16))
        batch_size = int(getattr(self.cfg.training, "batch_size", 8))

        if mode == "offline_training":
            with tqdm.tqdm(
                self.training_dataloader,
                desc="Training Critic",
                mininterval=1.0
            ) as critic_epoch:
                for critic_batch in critic_epoch:
                    critic_window = build_sliding_window_batch(
                        cfg=self.cfg,
                        batch=critic_batch,
                        env_goal_pos=self.env_goal_pos,
                        horizon=horizon,
                    )

                    log = self._critic_train_step_from_window(critic_window)

                    critic_epoch.set_postfix(
                        critic_loss=log["critic_loss"],
                        v_loss=log["v_loss"],
                        q_mean=log["q_mean"],
                        v_mean=log["v_mean"],
                    )

        elif mode == "online_training":
            online_steps = int(getattr(self.cfg.training, "online_critic_steps", 50))
            mix_ratio = float(getattr(self.cfg.training, "online_mix_ratio", 0.8))

            if self._offline_iter is None:
                self._offline_iter = iter(self.training_dataloader)

            with tqdm.tqdm(
                total=online_steps,
                desc="Online Critic Training",
                mininterval=1.0
            ) as pbar:
                for _ in range(online_steps):
                    online_batch = self.replaybuffer.load_sequence(
                        seq_type="all",
                        batch_size=batch_size,
                    )

                    try:
                        offline_batch = next(self._offline_iter)
                    except StopIteration:
                        self._offline_iter = iter(self.training_dataloader)
                        offline_batch = next(self._offline_iter)

                    mixed_raw_batch = mix_data(
                        online_batch=online_batch,
                        offline_batch=offline_batch,
                        ratio=mix_ratio,
                    )

                    mixed_window = build_sliding_window_batch(
                        cfg=self.cfg,
                        batch=mixed_raw_batch,
                        env_goal_pos=self.env_goal_pos,
                        horizon=horizon,
                    )

                    log = self._critic_train_step_from_window(mixed_window)
                    self.writer.add_scalar("critic_loss", log["critic_loss"], self.critic_step)
                    self.writer.add_scalar("v_loss", log["v_loss"], self.critic_step)
                    self.writer.add_scalar("q_mean", log["q_mean"], self.critic_step)
                    self.writer.add_scalar("v_mean", log["v_mean"], self.critic_step)
                    self.writer.add_scalar("reward_mean", log["reward_mean"], self.critic_step)
                    self.writer.add_scalar("critic/q_target_mean", log["q_target_mean"], self.critic_step)
                    self.writer.add_scalar("critic/q_target_std", log["q_target_std"], self.critic_step)
                    self.writer.add_scalar("critic/q_target_max", log["q_target_max"], self.critic_step)
                    self.writer.add_scalar("critic/q_target_min", log["q_target_min"], self.critic_step)
                    self.writer.add_scalar("critic/reward_std", log["reward_std"], self.critic_step)
                    self.writer.add_scalar("critic/reward_max", log["reward_max"], self.critic_step)
                    self.writer.add_scalar("critic/reward_min", log["reward_min"], self.critic_step)
                    self.critic_step += 1

                    pbar.set_postfix(
                        critic_loss=log["critic_loss"],
                        v_loss=log["v_loss"],
                        q_mean=log["q_mean"],
                        v_mean=log["v_mean"],
                    )
                    pbar.update(1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.diffusion_model.train()

    def _rollout(self,select_with_critic):
        rollout_env = self.rollout_env
        
        self.diffusion_model.eval()
        self.critic_network.eval()
        self.v_net.eval()
        
        try:
            rollout_results = rollout(
                model=self.diffusion_model,
                critic_network=self.critic_network,
                select_with_critic=select_with_critic,
                cond_norm_params=self.cond_norm_param_dict,
                cfg=self.cfg,
                device=self.device,
                replaybuffer=self.replaybuffer,
                rollout_env=rollout_env,
                training_dataset=self.training_dataset,
                action_norm_params=self.action_norm_params,
                log_dir=self.log_dir,
                epoch=self.epoch,
                
            )
            return rollout_results
        finally:
           pass
    def _policy_improvement(self, num_steps: int = 400):
        self.diffusion_model.train()
        self.critic_network.eval()
        self.v_net.eval()

        self.optimizer = optimizer_to(self.optimizer, device=self.device)

        batch_size = int(getattr(self.cfg.training, "batch_size", 8))
        horizon = int(getattr(self.cfg.training, "horizon", 16))

        with tqdm.tqdm(total=num_steps, desc="Improve Policy", mininterval=1.0) as pbar:
            for improve_step in range(num_steps):
                improve_batch = self.replaybuffer.load_sequence(
                    seq_type="all",
                    batch_size=batch_size,
                )

                improve_window = build_sliding_window_batch(
                    cfg=self.cfg,
                    batch=improve_batch,
                    env_goal_pos=self.env_goal_pos,
                    horizon=horizon,
                )

                cond_t = improve_window["cond_t"]
                improve_action = improve_window["act_window"].to(
                    self.device,
                    dtype=torch.float32,
                )

                n_cond, next_n_cond, visual_cond, action_mask, stage_out, next_stage_out = assemble_conditions(
                    cond_t=cond_t,
                    cond_norm_params=self.cond_norm_param_dict,
                    batch_size=batch_size,
                    device=self.device,
                    error_scale=1,
                    pull_start_t=36,
                    build_action_mask=False,
                )


                # ---------- compute advantage weight ----------
                with torch.no_grad():
                    q1, q2 = self.critic_network(
                        state=n_cond,
                        visual_obs=None,
                        action=improve_action,
                    )
                    q = torch.min(q1, q2)

                    v = self.v_net(
                        visual_obs=None,
                        low_dim_obs=next_n_cond,
                    )

                    if q.shape != v.shape:
                        v = v.expand_as(q)

                    adv = q - v

                    beta = float(getattr(self.cfg.training, "improve_beta", 1.0))
                    max_w = float(getattr(self.cfg.training, "max_improve_weight", 20.0))
                    if self.cfg.training.enable_entropy == True:
                        entropy_alpha = float(getattr(self.cfg.training, "improve_entropy_alpha", 0.01))
                        action_std = improve_action.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True)

                            # Normalize entropy proxy to keep its scale stable.
                        entropy_proxy = action_std / (action_std.mean().detach() + 1e-6)

                            # Match advantage shape: [B, T, 1]
                        entropy_proxy = entropy_proxy.expand_as(adv)

                        # Only give entropy bonus to relatively good actions.
                        # This avoids encouraging random low-Q actions.
                        adv_gate = (adv > adv.mean()).float()

                        adv_entropy = adv + entropy_alpha * entropy_proxy * adv_gate
                        weight = torch.exp(adv_entropy / beta)
                        weight = torch.clamp(weight, max=max_w)
                        weight = weight / (weight.mean().detach() + 1e-6)
                    elif self.cfg.training.enable_entropy == False:

                        weight = torch.exp(adv / beta)
                        weight = torch.clamp(weight, max=max_w)
                        weight = weight / (weight.mean() + 1e-6)

                # ---------- weighted diffusion update ----------
                self.optimizer.zero_grad(set_to_none=True)

                raw_loss = self.diffusion_model.compute_loss(
                    action=improve_action,
                    cond=cond_t,   # 注意这里应该传 cond_t，不是 cond_t["cond"]
                    action_norm_params=self.action_norm_params,
                    cond_norm_params=self.cond_norm_param_dict,
                    improve=True,
                    w=weight,
                )

                raw_loss.backward()
                self.optimizer.step()

                improve_log = {
                    "loss": float(raw_loss.item()),
                    "weight_mean": float(weight.mean().item()),
                    "weight_max": float(weight.max().item()),
                    "adv_mean": float(adv.mean().item()),
                    "adv_max": float(adv.max().item()),
                }

                self.writer.add_scalar("improve/loss", improve_log["loss"], self.improve_step)
                self.writer.add_scalar("improve/weight_mean", improve_log["weight_mean"], self.improve_step)
                self.writer.add_scalar("improve/weight_max", improve_log["weight_max"], self.improve_step)
                self.writer.add_scalar("improve/adv_mean", improve_log["adv_mean"], self.improve_step)
                self.writer.add_scalar("improve/adv_max", improve_log["adv_max"], self.improve_step)

                logger.info(
                    "improve_step=%d loss=%.6f weight_mean=%.6f weight_max=%.6f adv_mean=%.6f adv_max=%.6f",
                    improve_step,
                    improve_log["loss"],
                    improve_log["weight_mean"],
                    improve_log["weight_max"],
                    improve_log["adv_mean"],
                    improve_log["adv_max"],
                    extra={"tag": "Policy improvement"},
                )

                pbar.set_postfix(
                    loss=improve_log["loss"],
                    w_mean=improve_log["weight_mean"],
                    adv_mean=improve_log["adv_mean"],
                )
                pbar.update(1)

                self.improve_step += 1


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent.joinpath("config")),
    config_name="ddpm_critic_training",
    version_base="1.1"
)
def main(cfg):
    workspace = TrainDiffusionWorkspace(cfg=cfg)
    workspace.run()


if __name__ == "__main__":
    main()