from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
import argparse
import os
import torch

from test_env import TestSafetyGym
from lunarLander_safety import LunarLanderContinuous
from policies import SinglePlayerPolicy, SinglePlayerOffPolicy, \
                     ThreeModelFreePolicy, ThreeModelFreeOffPolicy

def get_args():
    args = argparse.ArgumentParser(description='Configs for running DESTA algorithms on LunarLander-v2')
    args.add_argument('--algorithm', default='desta_sac', 
                      choices=['ppo', 'sac', 'desta_ppo', 'desta_sac', 'ppo_lag', 'sac_lag'], 
                      help='algorithm name')
    args.add_argument('--f-log', default='desta_sac', 
                      help='filename for logging the results')
    args.add_argument('--seed', default=0, 
                      help='Random seed')
    args.add_argument('--episodes', default=10000, 
                      help='Number of training episodes')
    args.add_argument('--batch-size', default=100, 
                      help='Batch size')
    args.add_argument('--rollout-len', default=100, 
                      help='Policy rollout length for PPO')
    args.add_argument('--replay-buffer-size', default=10000, 
                      help='SAC replay buffer size')
    args.add_argument('--intervention-cost', default=0.25, 
                      help='Intervention cost for player 2')
    args.add_argument('--pi-standard-lr', default=1e-3, 
                      help='learning rate for standard policy')
    args.add_argument('--pi-safe-lr', default=1e-3, 
                      help='Learning rate for safe policy')
    args.add_argument('--pi-intervene-lr', default=1e-3, 
                      help='learning rate for intervention policy')
    args.add_argument('--action-dist-threshold', default=100, 
                      help='maximum discrepancy between actions in off-policy training (SAC)')
    args.add_argument('--eval-freq', default=10, 
                      help='evaluation frequency')
    args.add_argument('--eval-episodes', default=5, 
                      help='number of evaluation episodes')
    args.add_argument('--cost-threshold', default=0.2, 
                      help='threshold for incurring safety violation cost in LunarLander')
    args.add_argument('--cost-scale', default=1.0, 
                      help='scaling factor for the safety violation cost')
    args.add_argument('--device', default='cuda', 
                      help='cuda')
    args.add_argument('--max-steps', default=1000, type=int,
                      help='maximum number of steps in each training episode')
    return args.parse_args()
# FLAGS=flags.FLAGS

# # Experiment
# flags.DEFINE_string('algorithm', 'desta_sac', 'Type of algorithm: ppo, sac, desta_ppo, desta_sac')
# flags.DEFINE_string('f_log', 'desta_sac', 'File name to save results')
# flags.DEFINE_integer('seed', 0, 'Seed for random number generator')
# flags.DEFINE_integer('episodes', 10000, 'Total episodes to learn')

# # RL Algorithm
# flags.DEFINE_integer('batch_size', 100, 'Policy rollout length for PPO')
# flags.DEFINE_integer('rollout_len', 100, 'Policy rollout length for PPO')
# flags.DEFINE_integer('replay_buffer_size', 10000, 'Size of Experience Replay for SAC')

# # Important to tune...
# flags.DEFINE_float('intervention_cost', 0.25, 'Cost of Player 2 intervening')
# flags.DEFINE_float('pi_standard_lr', 1e-3, 'Learning rate of standard policy')
# flags.DEFINE_float('pi_safe_lr', 1e-3, 'Learning rate of safe policy')
# flags.DEFINE_float('pi_intervene_lr', 1e-3, 'Learnin rate of intervention policy')
# flags.DEFINE_float('action_dist_threshold', 100, 'Maximum distance between actions \
#     if using off-policy for safe policy (only applicable to ThreeModelFreeOffPolicy)')

def run_eval_episode(env, policy, cumulative_cost, max_steps):
    # post training eval loop
    obs = env.reset()
    done = False
    interventions = []
    cum_reward = 0
    i = 0
    while not done and i < max_steps:
        # post-learning evaluation loop
        action, action_dict = policy.get_action(obs, deterministic=True)
        interventions.append(action_dict['intervene'])
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        i += 1
    return np.sum(interventions), cum_reward, i
        # env.render(interventions=interventions, cumulative_cost=cumulative_cost)

def main(args):
    device = torch.device(args.device)
    log_dir = os.path.join('lunar_logs', f'{args.algorithm}', f'{args.seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = open(os.path.join(log_dir, 'training_logs'), 'w')
    g = open(os.path.join(log_dir, 'evaluation_logs'), 'w')
    env = LunarLanderContinuous()

    total_episodes = args.episodes
    n_steps_learn = args.batch_size
    eval_freq = args.eval_freq
    if args.algorithm == 'ppo':
        policy = SinglePlayerPolicy(env=env, batch_size=args.rollout_len, pi_1_lr=args.pi_standard_lr)
    elif args.algorithm == 'sac':
        policy = SinglePlayerOffPolicy(env=env, buffer_size=args.replay_buffer_size, 
                                       pi_1_lr=args.pi_standard_lr, device=args.device)
    elif args.algorithm == 'desta_ppo':
        policy = ThreeModelFreePolicy(env=env, batch_size=args.batch_size, 
                                      intervention_cost=args.intervention_cost)
    elif args.algorithm == 'desta_sac':
        policy = ThreeModelFreeOffPolicy(env=env, buffer_size=args.replay_buffer_size, 
                                         pi_standard_lr=args.pi_standard_lr, 
                                         pi_safe_lr=args.pi_safe_lr,
                                         pi_intervene_lr=args.pi_intervene_lr, 
                                         intervention_cost=args.intervention_cost, 
                                         action_dist_threshold=args.action_dist_threshold, 
                                         device=args.device)
    
    n_steps = 1
    ep_rewards = []
    ep_timesteps = deque(maxlen=100)
    ep_costs = []
    cumulative_cost = 0
    eval_return_mean = []
    eval_return_std = []
    eval_intervention_mean = []
    eval_return_episode = []

    for i in range(total_episodes):
        if i % eval_freq == 0 and i > 0:
            eval_returns = []
            eval_interventions = []
            for j in range(args.eval_episodes):
                cum_intervention, cum_reward, eval_episode_length = run_eval_episode(env, policy, cumulative_cost, args.max_steps)
                eval_returns.append(cum_reward)
                eval_interventions.append(cum_intervention)
                g.write(f'Evaluation episode: {i}_{j} | Episode length: {eval_episode_length} | Total intervention: {cum_intervention} | Episode return: {cum_reward:.2f}\n')
                g.flush()
            eval_return_mean.append(np.mean(eval_returns))
            eval_return_std.append(np.std(eval_returns))
            eval_intervention_mean.append(np.mean(eval_interventions))
            eval_return_episode.append(i)
            
        obs = env.reset()
        done = False
        last_done = True
        ep_reward = 0
        ep_timestep = 0
        ep_cost = 0
        interventions = []
        while not done and ep_timestep < args.max_steps:
            actions, action_dict = policy.get_action(obs)

            interventions.append(action_dict['intervene'])

            new_obs, reward, done, info = env.step(actions)
            if abs(new_obs[0]) >= args.cost_threshold:
                info['cost'] = abs(new_obs[0]) * args.cost_scale
            else:
                info['cost'] = 0.0
            policy.store_transition(obs, actions, new_obs, reward, done, last_done, info, action_dict)

            obs = new_obs
            last_done = done
            ep_reward += reward
            ep_cost += info['cost']
            cumulative_cost += info['cost']

            ep_timestep += 1

            if n_steps >= n_steps_learn:
                current_progress_remaining = 1 - i / total_episodes
                policy.learn(obs, done, current_progress_remaining)
                n_steps = 0
    
            n_steps += 1
        ep_rewards.append(ep_reward)
        ep_timesteps.append(ep_timestep)
        ep_costs.append(ep_cost)
        f.write(f'Training episode: {i} | Episode length: {ep_timestep} | Total cost: {sum(ep_costs):.2f} | Episode return: {ep_rewards[-1]:.2f}\n')
        f.flush()


        # print(i, sum(interventions), np.mean(ep_timesteps),
              # np.mean(ep_costs[-100:]), np.mean(ep_rewards[-100:]))

    np.savetxt(os.path.join(log_dir, f'{policy.__class__.__name__}_training_data.csv'), 
               np.array([list(range(total_episodes)), ep_costs, ep_rewards]).T,
               delimiter=',')
    np.savetxt(os.path.join(log_dir, f'{policy.__class__.__name__}_evaluation_data.csv'), 
               np.array([eval_return_episode, eval_interventions, eval_return_mean, eval_return_std]).T,
               delimiter=',')

    run_eval_episode(env, policy, cumulative_cost)
    
    f.close()
    g.close()
    # plt.savefig("trajectory.png")


if __name__ == '__main__':
    # app.run(main)
    args = get_args()
    main(args)
