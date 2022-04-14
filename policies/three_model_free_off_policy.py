from typing import Tuple, Dict

import gym
import torch
import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.utils import get_device

from .buffers import RolloutBuffer
from .base_policy import Policy, ActionDict
from .algos import ppo_train, sac_train


def lr_schedule_fn_1(progress: float) -> float:
    return 1e-3


def lr_schedule_fn_2(progress: float) -> float:
    return 1e-3


device = get_device("cpu")
GRADIENT_STEPS = 1

class ThreeModelFreeOffPolicy(Policy):
    '''
        DESTA with:
            - SAC for standard policy
            - SAC for safe policy
            - SAC for intervention policy
    '''
    def __init__(self, 
                 env:gym.Env, 
                 buffer_size:int,
                 pi_standard_lr:float,
                 pi_safe_lr:float,
                 pi_intervene_lr:float,
                 intervention_cost:float,
                 action_dist_threshold:float,
                 gradient_steps:int=1, 
                 device:str='cuda'):
        self.device = get_device(device)
        self.lr_schedule_fn_pi_standard = lambda progress: pi_standard_lr
        self.lr_schedule_fn_pi_safe = lambda progress: pi_safe_lr
        self.lr_schedule_fn_pi_intervene = lambda progress: pi_intervene_lr
        self.intervention_cost = intervention_cost
        self.action_dist_threshold = action_dist_threshold
        self.gradient_steps = gradient_steps

        # Buffers
        self.standard_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size, 
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        self.safe_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        intervene_action_space = gym.spaces.Box(0, 1, shape=(1,))
        self.intervene_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=intervene_action_space
            )

        # Policies
        self.standard_policy = \
            SACPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=self.lr_schedule_fn_pi_standard,
                activation_fn=torch.nn.ReLU,
        ).to(self.device)
        self.safe_policy = \
            SACPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=self.lr_schedule_fn_pi_safe,
                activation_fn=torch.nn.ReLU,
        ).to(self.device)
        self.intervene_policy = \
            SACPolicy(
                observation_space=env.observation_space,
                action_space=intervene_action_space,
                lr_schedule=self.lr_schedule_fn_pi_intervene,
                activation_fn=torch.nn.ReLU,
        ).to(self.device)

        self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=self.lr_schedule_fn_pi_standard(1))
        self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)

    def get_action(self, 
                   observation:np.ndarray,
                   deterministic:bool=False) -> Tuple[np.ndarray, ActionDict]:
        '''
            Compute actions of standard, safe, and intervene for current observation.

            Arguments:
                observation - np.ndarray | current environment observation
            
            Returns:
                action - np.ndarray | policy's action for observation
        '''
        with torch.no_grad():
            observation = torch.as_tensor([observation]).to(self.device)
            standard_action = self.standard_policy.forward(observation, deterministic=deterministic)
            safe_action = self.safe_policy.forward(observation, deterministic=deterministic)
            intervene_action = self.intervene_policy.forward(observation, deterministic=deterministic)

            # NOTE: The implementation of SAC used here doesn't allow for
            # discrete action spaces. Hences to 'simulate' a binary option
            # for intervening or not intervening, we simply intervene if SAC's
            # picked action is > 0 else we don't intervene. 
            intervene = True if intervene_action >= 0 else False
            action = safe_action if intervene else standard_action
            action = action.cpu().numpy()[0, :]
    
            action_dict = \
                {'intervene': intervene,
                 'intervene_action': intervene_action.cpu().numpy()[0, :],
                 'safe_action':safe_action.cpu().numpy()[0,:]
                }
            
            return action, action_dict

    def store_transition(
            self,
            observation:np.ndarray,
            action:np.ndarray,
            next_observation:np.ndarray,
            reward:float,
            done:bool,
            last_done:bool,
            info:dict,
            action_dict: ActionDict):
        '''
            Store appropriate interaction data in each policy's respective
            buffer.
        '''
        # EXPERIMENTAL VERSION
        # Standard
        infos = [info]
        for info in infos:
            info['Timelimit.truncated'] = False
        self.standard_buffer.add( \
            observation,
            next_observation,
            action,
            reward,
            done,
            infos)

        # Safe  
        safe_action = action_dict['safe_action']
        action_difference = np.linalg.norm(action - safe_action)
        if action_difference <= self.action_dist_threshold:
            safe_buffer_cost = infos[0]['cost'] + self.intervention_cost
            self.safe_buffer.add(\
                observation,
                next_observation,
                action,
                safe_buffer_cost,
                done,
                infos)
        
        # Intervene
        intervene = action_dict['intervene']
        intervene_buffer_cost = \
            infos[0]['cost'] + self.intervention_cost if intervene else infos[0]['cost']
        self.intervene_buffer.add( \
            observation, 
            next_observation, 
            float(action_dict['intervene_action']), 
            -intervene_buffer_cost, 
            done, 
            infos
        )

        # DEFINITELY WORKING VERSION.
        # intervene = action_dict['intervene']
        # cost = info['cost'] + self.intervention_cost if intervene else info['cost']
        # info = [info]
        # for i in info:
        #     i['Timelimit.truncated'] = False
        # self.p1_replay_buffer.add(obs, next_obs, action, reward, done, info)
        # self.p2_replay_buffer.add(obs, next_obs, action, p2_cost, done, info)
        # self.intervene_replay_buffer.add(obs, next_obs, float(action_dict['action_i']), -cost, done, info)

    def learn(self,
              observation,
              done,
              progress_remaining:float):
        sac_train(self.standard_policy,
                  self.standard_buffer, 
                  self.log_ent_coef, 
                  self.ent_coef_optimizer,
                  self.lr_schedule_fn_pi_standard,
                  progress_remaining, 
                  gradient_steps=self.gradient_steps,
                  target_entropy=self.target_entropy)
        sac_train(self.safe_policy,
                  self.safe_buffer,
                  self.log_ent_coef,
                  self.ent_coef_optimizer,
                  self.lr_schedule_fn_pi_safe,
                  progress_remaining,
                  gradient_steps=self.gradient_steps,
                  target_entropy=self.target_entropy)
        sac_train(self.intervene_policy,
                  self.intervene_buffer,
                  self.log_ent_coef,
                  self.ent_coef_optimizer,
                  self.lr_schedule_fn_pi_intervene,
                  progress_remaining,
                  gradient_steps=self.gradient_steps,
                  target_entropy=self.target_entropy)
   
