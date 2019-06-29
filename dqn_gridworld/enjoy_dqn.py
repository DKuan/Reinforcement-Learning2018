#!/usr/bin/env python3
"""
This file is for the enjoy of the AGENT_ONE,
Author: FFAI_WD
Rights Reserved
"""
import gym
import torch
import argparse
import numpy as np
from tqdm import tqdm as tqdm

from batch_dqn_random_gw import RL_AGENT_ONE
from grid_world import Grid_World

""" check the device """
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='init the par')
parser.add_argument('-game_num', '--game_num', nargs='?', default=30000)
parser.add_argument('-step_stop_num', '--step_stop', nargs='?', default=90000)
parser.add_argument('-memory_size', '--memory_size', nargs='?', default=10000)
parser.add_argument('-max_episode_num', '--max_episode_num', nargs='?', default=300)
parser.add_argument('-device', '--device', nargs='?', default=device)
parser.add_argument('-model_path', '--model_path', nargs='?', default='model/')
parser.add_argument('-model_load', '--model_load', nargs='?', default=True)
parser.add_argument('-old_model_name', '--o_model_name', nargs='?', default='0626100317.pt')
parser.add_argument('-r_memory_Fname', '--r_memory_Fname', nargs='?', default='r_memory.pkl')
parser.add_argument('-batch_size', '--batch_size', nargs='?', default=300)
parser.add_argument('-gamma', '--gamma', nargs='?', default=0.999)
parser.add_argument('-learning_rate', '--lr', nargs='?', default=0.001)
parser.add_argument('-learn_start', '--learn_start_t', nargs='?', default=4000)
parser.add_argument('-learn_fre', '--learn_fre', nargs='?', default=200)
parser.add_argument('-eps_init', '--eps_init', nargs='?', default=-10000)
parser.add_argument('-eps_T', '--eps_T', nargs='?', default=40000)
parser.add_argument('-target_update', '--update_period', nargs='?', default=800)
parser.add_argument('-game', '--game_name', nargs='?', default="FFAI-3-v1")
args = parser.parse_args()
#print(args.update_period)

""" switch the board size """
board_size_dict = {'FFAI-3-v1': (7, 14)}

""" Smaller variants """
step_now = 0 # record the global steps
reward_num = 0
env = Grid_World(7, 14, 4) 
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
                        board_size_dict[args.game_name],
                        args.device,
                        args.model_path,
                        args.r_memory_Fname,
                        args.o_model_name,
                        model_load=args.model_load)

""" Set seed for reproducibility """
seed = 0

""" Play 10 games """
for i in range(args.game_num):
        print('the game now is ', i)

        """ Reset environment """
        obs = env.reset()
        agent_one.reset()
        episode_steps_cnt = 0
        done = False

        """ Take actions as long as game is not done """
        while(1):
            episode_steps_cnt += 1
            step_now += 1 # update the steps

            """ interact with the env and render"""
            action = agent_one.act_enjoy(obs)
            obs_new, reward, done = env.step(action) # Gym step function

            """ print the reward info """
            if reward > 0:
                reward_num += 1
                print('find food for', reward_num, 'times', reward)
                print('used time_step for ', episode_steps_cnt)

            env.render()

            """ if the game is done """
            obs = obs_new
            input_str = input()
            if input_str == 'q':
                break

            """ control the game """
            if done: break
            if episode_steps_cnt > args.max_episode_num:
                break
