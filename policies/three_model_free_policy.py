from typing import Tuple, Dict
import gym
import numpy as np
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

from .buffers import RolloutBuffer
from .base_policy import Policy, ActionDict
from .algos import ppo_train


def lr_schedule_fn_1(progress: float) -> float:
    if progress > 0.1:
        return 1e-3
    return 1e-3 * progress / 0.1


def lr_schedule_fn_2(progress: float) -> float:
    if progress > 0.1:
        return 5e-4
    return 5e-4 * progress / 0.1


# device = get_device("cpu")

class ThreeModelFreePolicy(Policy):
    """
    Algorithm implemented using three PPO agents (player1, player2 and intervention agent).

    Player 1 only learns from its own actions, and same for player 2, using interventions in
    the rollout buffer.

    The intervention policy can learn at every timestep.
    """
    def __init__(self, env: gym.Env, 
                 batch_size: int, 
                 intervention_cost: float,
                 gamma: float = 1.0, 
                 device='cuda'):
        self.device = get_device(device)
        self.fixed_cost = intervention_cost
        intervene_action_space = gym.spaces.Discrete(2)
        self.p1_rollout_buffer = RolloutBuffer(
            buffer_size=batch_size, observation_space=env.observation_space,
            action_space=env.action_space, gamma=gamma)

        self.p2_rollout_buffer = RolloutBuffer(
            buffer_size=batch_size, observation_space=env.observation_space,
            action_space=env.action_space, gamma=gamma)

        self.p3_rollout_buffer = RolloutBuffer(
            buffer_size=batch_size, observation_space=env.observation_space,
            action_space=intervene_action_space, gamma=gamma)

        self.p1_policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule_fn_1,
            net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
            activation_fn=torch.nn.ReLU,
        ).to(self.device)

        self.p2_policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule_fn_2,
            net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
            activation_fn=torch.nn.ReLU,
        ).to(self.device)

        self.intervene_policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=intervene_action_space,
            lr_schedule=lr_schedule_fn_2,
            net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
            activation_fn=torch.nn.ReLU,
        ).to(self.device)


    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, ActionDict]:
        with torch.no_grad():
            # Convert to pytorch tensor
            obs_tensor = torch.as_tensor([obs]).to(self.device)
            actions_i, values_i, log_prob_i = self.intervene_policy.forward(
                obs_tensor, deterministic=deterministic)
            actions_1, values_1, log_prob_1 = self.p1_policy.forward(
                obs_tensor, deterministic=deterministic)
            actions_2, values_2, log_prob_2 = self.p2_policy.forward(
                obs_tensor, deterministic=deterministic)

            intervene = bool(actions_i.cpu().numpy()[0])
            actions = actions_2 if intervene else actions_1
            actions = actions.cpu().numpy()[0, :]
            action_dict = {'intervene': intervene,
                           'values_1': values_1,
                           'values_2': values_2,
                           'values_i': values_i,
                           'log_prob_1': log_prob_1,
                           'log_prob_2': log_prob_2,
                           'log_prob_i': log_prob_i,
                           'action_i': actions_i.cpu().numpy(),
                           }
            return actions, action_dict

    def store_transition(
            self,
            obs: np.ndarray,
            action:np.ndarray,
            next_obs: np.ndarray,
            reward: float,
            done: bool,
            last_done: bool,
            info: Dict,
            action_dict: ActionDict):
        intervene = action_dict['intervene']
        cost = info['cost'] + self.fixed_cost if intervene else info['cost']
        self.p1_rollout_buffer.add(obs, action, reward, last_done, action_dict['values_1'],
                                   action_dict['log_prob_1'], not intervene)
        self.p2_rollout_buffer.add(obs, action, -cost, last_done, action_dict['values_2'],
                                   action_dict['log_prob_2'], intervene)
        self.p3_rollout_buffer.add(obs, action_dict['action_i'], -cost, last_done,
                                   action_dict['values_i'], action_dict['log_prob_i'], True)

    def learn(self, obs: np.ndarray, done: bool, current_progress_remaining: float):
        with torch.no_grad():
            # Compute action and value for the last timestep
            obs_tensor = torch.as_tensor([obs]).to(self.device)
            _, values_i, _ = self.intervene_policy.forward(obs_tensor)
            _, values_1, _ = self.p1_policy.forward(obs_tensor)
            _, values_2, _ = self.p2_policy.forward(obs_tensor)

            self.p1_rollout_buffer.compute_returns_and_advantage(last_values=values_1, dones=done)
            self.p2_rollout_buffer.compute_returns_and_advantage(last_values=values_2, dones=done)
            self.p3_rollout_buffer.compute_returns_and_advantage(last_values=values_i, dones=done)

        ppo_train(self.p1_policy, self.p1_rollout_buffer, current_progress_remaining,
                  lr_schedule_fn_1)
        ppo_train(self.p2_policy, self.p2_rollout_buffer, current_progress_remaining,
                  lr_schedule_fn_2)
        ppo_train(self.intervene_policy, self.p3_rollout_buffer, current_progress_remaining,
                  lr_schedule_fn_2)
        self.p1_rollout_buffer.reset()
        self.p2_rollout_buffer.reset()
        self.p3_rollout_buffer.reset()
