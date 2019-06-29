#!/usr/bin/env python3
"""
This file is for the train of the rl_agent_two
Author: FFAI_WD
Rights Reserved
"""
import gym
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm as tqdm

from batch_dqn_agent import RL_AGENT_ONE

""" check the device """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='init the par')
parser.add_argument('-game_num', '--game_num', nargs='?', default=40000)
parser.add_argument('-step_stop_num', '--step_stop', nargs='?', default=3000000)
parser.add_argument('-max_episode_num', '--max_episode_num', nargs='?', default=9999)
parser.add_argument('-memory_size', '--memory_size', nargs='?', default=9000)
parser.add_argument('-device', '--device', nargs='?', default=device)
parser.add_argument('-model_path', '--model_path', \
                       nargs='?', default='model/')
parser.add_argument('-old_model_name', '--o_model_name', nargs='?', default='0613205430.pt')
parser.add_argument('-r_memory_Fname', '--r_memory_Fname', nargs='?', default='r_memory.pkl')
parser.add_argument('-model_load', '--model_load', nargs='?', default=False)
parser.add_argument('-batch_size', '--batch_size', nargs='?', default=256)
parser.add_argument('-beta_replay_iters', '--beta_replay_iters', nargs='?', default=3000000)
parser.add_argument('-gamma', '--gamma', nargs='?', default=0.93)
parser.add_argument('-learning_rate', '--lr', nargs='?', default=0.0006)
parser.add_argument('-learn_start', '--learn_start_t', nargs='?', default=10000)
parser.add_argument('-learn_fre', '--learn_fre', nargs='?', default=200)
parser.add_argument('-eps_init', '--eps_init', nargs='?', default=-100)
parser.add_argument('-eps_T', '--eps_T', nargs='?', default=1600000)
parser.add_argument('-target_update', '--update_period', nargs='?', default=100)
parser.add_argument('-game', '--game_name', nargs='?', default="grid_world")
args = parser.parse_args()
#print(args.update_period)

""" switch the board size """

""" Smaller variants """
seed = 1
step_now = 0 # record the global steps
reward_num = 0
env = gym.make('MountainCar-v0').unwrapped 
state_size = [env.observation_space.shape[0], 1]
torch.manual_seed(seed)
env.seed(seed)

agent_one = RL_AGENT_ONE(args.memory_size,
                        args.batch_size,
                        args.learn_start_t,
                        args.learn_fre,
                        args.lr,
                        args.step_stop,
                        args.eps_T,
                        args.eps_init,
                        args.gamma,
                        args.update_period,
                        state_size,
                        args.device,
                        args.model_path,
                        args.r_memory_Fname,
                        args.o_model_name,
                        args.model_load)

""" Set seed for reproducibility """
#model_store_set = [args.step_stop-1200000, args.step_stop-1000000, args.step_stop-900000, \
#        args.step_stop-800000, args.step_stop-600000, args.step_stop-400000, \
#        args.step_stop-300000, args.step_stop-100000, args.step_stop-10000]
model_store_set = [args.step_stop-80000, args.step_stop-50000, args.step_stop-20000]

""" Play 10 games """
for i in range(args.game_num):

        """ print the record """
        if i % 3 == 0:
            print('____________________________')
            print('the step now is ', step_now)
            print('the time is', time.strftime('%m%d%H%M%S'))
            print('the memory is', agent_one.memory.__len__())
            print('the e_greedy is', agent_one.e_greedy)
            print('____________________________')

        """ end the train """
        if step_now > args.step_stop:   break

        """ Reset environment """
        obs = env.reset()
        agent_one.reset()
        done = False

        """ Take actions as long as game is not done """
        step_game_start = step_now
        while(1):
            done = False
            step_now += 1 # update the steps
            episode_steps_cnt = step_now - step_game_start

            """ save the model """
            if step_now in model_store_set:
                agent_one.save()
                break

            """ interact with the env and render"""
            action = agent_one.act(obs, step_now)
            obs_new, reward, done, _ = env.step(action)
            #reward = abs(obs_new[0] + 0.5)
            agent_one.store_memory(env, step_now, obs, action, obs_new, reward, done)
            obs = obs_new

            """ print the info """
            if done:
                reward_num += 1
                print('find food for', reward_num,'times', reward)
                #print('the steps used ', episode_steps_cnt)

            """ end the while """
            if done or episode_steps_cnt > args.max_episode_num: 
                agent_one.memory.update(True)
                print('episode steps ', episode_steps_cnt)
                agent_one.learn()
                break
