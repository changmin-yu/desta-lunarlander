from typing import Tuple, Dict

import gym
import torch
import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.utils import get_device

from .algos import sac_train
from .base_policy import Policy, ActionDict

class SinglePlayerOffPolicy(Policy):
    '''
        Standard single agent RL policy using SAC.

        To account for safety violations, 'cost' of safety violation is
        combined with the reward.    
    '''    
    def __init__(self, 
                 env:gym.Env,
                 buffer_size:int,
                 pi_1_lr=float,
                 gamma:float=0.99,
                 gradient_steps:int=1, 
                 device='cuda'):
        self.device = get_device(device)
        self.lr_schedule_fn = lambda progress: pi_1_lr
        self.gradient_steps = gradient_steps

        self.buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space
            )

        self.policy = \
            SACPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=self.lr_schedule_fn,
                activation_fn=torch.nn.ReLU,
            ).to(self.device)

        self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=self.lr_schedule_fn(1))
        self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)

    def get_action(self, 
                   observation:np.ndarray,
                   deterministic:bool=False) -> Tuple[np.ndarray, ActionDict]:
        '''
            Return action for current observation.

            Arguments:
                observation - np.ndarray | current environment observation
            
            Returns:
                action - np.ndarray | policy's action for observation
                action_dict - IGNORE, just there so code works nicely
        '''
        with torch.no_grad():
            observation = torch.as_tensor([observation]).to(self.device)
            action = self.policy.forward(
                        observation, deterministic=deterministic)
            action = action.cpu().numpy()[0, :]
            return action, {'intervene': False}

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
        '''
            Store interaction in buffer
        '''
        cost = 50. * info['cost']
        info = [info]
        for i in info:
            i['Timelimit.truncated'] = False
        self.buffer.add(obs, next_obs, action, reward - cost, done, info)

    def learn(self, obs: np.ndarray, done: bool, current_progress_remaining: float):
        sac_train(self.policy, self.buffer, self.log_ent_coef, self.ent_coef_optimizer,
                  self.lr_schedule_fn, current_progress_remaining, gradient_steps=self.gradient_steps,
                  target_entropy=self.target_entropy)
