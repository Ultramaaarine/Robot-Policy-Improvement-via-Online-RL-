from collections import deque
import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from sac_diffusion.agents.base_agent import BaseAgent


Observation = Union[np.ndarray, torch.Tensor, Dict[str, Any], Sequence[float], float]


class LowDimAgent(BaseAgent):
    """Agent wrapper that handles low-dimensional observations and replay logic."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        super().__init__(cfg)

        self.device = torch.device(cfg.get("device", "cpu"))
        self.total_steps = int(cfg.get("total_steps", 1000))
        self.max_episode_steps = int(cfg.get("max_episode_steps", self.total_steps))
        self.accumulate_rewards_every = int(cfg.get("accumulate_rewards_every", 1))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.obs_keys: Tuple[str, ...] = tuple(cfg.get("obs_space", ()))
        buffer_size = int(cfg.get("replay_buffer_size", 100_000))
        self.replay_buffer = deque(maxlen=buffer_size)

        self._last_obs: Optional[Any] = None
        self._last_done: bool = True
        self._episode_step: int = 0

        if isinstance(self.actor, nn.Module):
            self.actor.to(self.device)
        if isinstance(self.critic, nn.Module):
            self.critic.to(self.device)
            critic_lr = float(cfg.get("critic_lr", 3e-4))
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            self.critic_optimizer = None

    def get_features_from_observation(self, obs: Observation) -> torch.Tensor:
        """Project raw observations (dict, tensor, numpy) into a flat tensor."""
        if obs is None:
            raise ValueError("Observation cannot be None.")

        if torch.is_tensor(obs):
            tensor = obs
        elif isinstance(obs, np.ndarray):
            tensor = torch.from_numpy(obs)
        elif isinstance(obs, (list, tuple)):
            tensor = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, dict):
            features = []
            keys = self.obs_keys or tuple(obs.keys())
            for key in keys:
                if key not in obs:
                    continue
                features.append(self._value_to_tensor(obs[key]))
            if not features:
                raise ValueError(f"No valid observation keys found in: {obs.keys()}.")
            tensor = torch.cat(features, dim=-1)
        elif isinstance(obs, (int, float)):
            tensor = torch.tensor([obs], dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        return tensor.flatten().float().to(self.device)

    def get_state_dim(self, obs: Observation) -> int:
        """Return the dimensionality of the flattened observation."""
        return int(self.get_features_from_observation(obs).numel())

    def get_action(self, obs: Observation, deterministic: bool = False, **policy_kwargs) -> np.ndarray:
        """Query the actor/policy for an action, falling back to random sampling."""
        state = self.get_features_from_observation(obs).unsqueeze(0)
        action_tensor: Optional[torch.Tensor] = None

        if isinstance(self.actor, nn.Module):
            self.actor.eval()
            with torch.no_grad():
                try:
                    action_tensor = self.actor(state, deterministic=deterministic, **policy_kwargs)
                except TypeError:
                    action_tensor = self.actor(state)
        elif hasattr(self.policy, "act"):
            action_tensor = self.policy.act(state, deterministic=deterministic, **policy_kwargs)

        if action_tensor is None and hasattr(self.policy, "sampling_action_from_demos"):
            demos = policy_kwargs.get("demos")
            action_mode = policy_kwargs.get("action_mode", "pos")
            if demos is not None:
                action_tensor = self.policy.sampling_action_from_demos(demos, action_mode=action_mode)

        if action_tensor is None:
            sample = self.env.action_space.sample()
            return sample if isinstance(sample, np.ndarray) else np.asarray(sample, dtype=np.float32)

        action_tensor = action_tensor.squeeze(0)
        return action_tensor.detach().cpu().numpy()

    def evaluate_policy(self, num_episodes: int = 1, deterministic: bool = True) -> Dict[str, Any]:
        """Roll out the policy for a few episodes and return basic statistics."""
        trajectories = []
        rewards = []

        for _ in range(num_episodes):
            obs = self.reset_env()
            done = False
            steps = 0
            total_reward = 0.0
            episode = []

            while not done and steps < self.max_episode_steps:
                action = self.get_action(obs, deterministic=deterministic)
                next_obs, reward, done, info = self._step_env(action)

                transition = {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "done": bool(done),
                    "info": info,
                }
                episode.append(transition)
                self._push_to_buffer(transition)

                total_reward += float(reward)
                obs = next_obs
                steps += 1

            trajectories.append(episode)
            rewards.append(total_reward)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        return {"trajectories": trajectories, "episode_rewards": rewards, "mean_reward": mean_reward}

    def reset_env(self) -> Observation:
        """Reset the wrapped environment and return the initial observation."""
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, _ = result
        else:
            obs = result
        self._last_obs = obs
        self._last_done = False
        self._episode_step = 0
        return obs

    def play_step(self, policy=None, deterministic: bool = False) -> Dict[str, Any]:
        """Play a single transition, optionally using an external policy callable."""
        if self._last_obs is None or self._last_done:
            obs = self.reset_env()
        else:
            obs = self._last_obs

        if callable(policy):
            action = policy(obs)
        else:
            action = self.get_action(obs, deterministic=deterministic)

        next_obs, reward, done, info = self._step_env(action)
        transition = {
            "obs": obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": bool(done),
            "info": info,
        }
        self._push_to_buffer(transition)

        self._episode_step += 1
        self._last_done = done or self._episode_step >= self.max_episode_steps
        self._last_obs = None if self._last_done else next_obs

        return transition

    def train_critic_network(self, batch_size: Optional[int] = None) -> Optional[float]:
        """Train the critic with sampled replay transitions."""
        if not isinstance(self.critic, nn.Module) or self.critic_optimizer is None:
            raise RuntimeError("Critic network or optimizer is not properly configured.")

        batch = self._sample_batch(batch_size or self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        critic_input = torch.cat([states, actions], dim=-1)
        current_q = self.critic(critic_input).squeeze(-1)

        with torch.no_grad():
            if isinstance(self.actor, nn.Module):
                next_actions = self.actor(next_states).detach()
            else:
                next_actions = torch.zeros_like(actions)
            next_input = torch.cat([next_states, next_actions], dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * self.critic(next_input).squeeze(-1)

        loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return float(loss.item())

    def _push_to_buffer(self, transition: Dict[str, Any]) -> None:
        self.replay_buffer.append(transition)

    def _sample_batch(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None

        samples = random.sample(self.replay_buffer, batch_size)
        states = torch.stack([self.get_features_from_observation(t["obs"]) for t in samples])
        actions = torch.stack([self._action_to_tensor(t["action"]) for t in samples])
        rewards = torch.tensor([t["reward"] for t in samples], dtype=torch.float32, device=self.device)
        next_states = torch.stack([self.get_features_from_observation(t["next_obs"]) for t in samples])
        dones = torch.tensor([float(t["done"]) for t in samples], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def _action_to_tensor(self, action: Any) -> torch.Tensor:
        if torch.is_tensor(action):
            tensor = action
        elif isinstance(action, np.ndarray):
            tensor = torch.from_numpy(action)
        elif isinstance(action, (list, tuple)):
            tensor = torch.tensor(action, dtype=torch.float32)
        else:
            tensor = torch.tensor([action], dtype=torch.float32)
        return tensor.float().to(self.device)

    def _value_to_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, (int, float)):
            tensor = torch.tensor([value], dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported observation value type: {type(value)}")
        return tensor.flatten().float().to(self.device)

    def _step_env(self, action: Any):
        step_result = self.env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_result
        return next_obs, reward, done, info
