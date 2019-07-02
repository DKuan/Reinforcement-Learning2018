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

from grid_world  import Grid_World
from a2c_agent import RL_AGENT_A3C 

""" check the device """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='init the par')
parser.add_argument('-game_num', '--game_num', nargs='?', default=40000)
parser.add_argument('-step_stop_num', '--step_stop', nargs='?', default=7000000)
parser.add_argument('-max_episode_num', '--max_episode_num', nargs='?', default=500)
parser.add_argument('-learning_rate', '--lr', nargs='?', default=0.0009)
parser.add_argument('-device', '--device', nargs='?', default=device)
parser.add_argument('-model_path', '--model_path', nargs='?', default='model/')
parser.add_argument('-old_model_name', '--o_model_name', nargs='?', \
        default='0701211805.pt')
parser.add_argument('-r_memory_Fname', '--r_memory_Fname', nargs='?', default='None')
parser.add_argument('-model_load', '--model_load', nargs='?', default=False)
parser.add_argument('-gamma', '--gamma', nargs='?', default=0.99)
parser.add_argument('-game', '--game_name', nargs='?', default="grid_world")
args = parser.parse_args()
#print(args.update_period)

""" switch the board size """
board_size_dict = {'grid_world': (7, 14)}

""" Smaller variants """
step_now = 0 # record the global steps
reward_num = 0
env = Grid_World(7, 14, 4)
agent = RL_AGENT_A3C(args.lr,
                        args.gamma,
                        board_size_dict[args.game_name],
                        args.device,
                        args.model_path,
                        args.r_memory_Fname,
                        args.o_model_name,
                        args.model_load)

""" Set seed for reproducibility """
model_store_set = [
        args.step_stop-1000000, args.step_stop-800000, 
        args.step_stop-600000, args.step_stop-400000, \
        args.step_stop-300000, args.step_stop-100000, args.step_stop-10000]

""" Play 10 games """
for i in range(args.game_num):

        """ print the record """
        if i % 10 == 0:
            print('____________________________')
            print('the step now is ', step_now)
            print('the time is', time.strftime('%m%d%H%M%S'))
            print('____________________________')

        """ end the train """
        if step_now > args.step_stop:   break

        """ Reset environment """
        obs = env.reset()
        done = False

        """ Take actions as long as game is not done """
        step_game_start = step_now
        while(1):
            done = False
            step_now += 1 # update the steps
            episode_steps_cnt = step_now - step_game_start

            """ save the model """
            if step_now in model_store_set:
                agent.save()
                break

            """ interact with the env and render"""
            action = agent.act(obs)
            obs_new, reward, done = env.step(action)
            agent.save_trace(reward, done)

            """ print the info """
            if reward > 0:
                reward_num += 1
                print('find food for', reward_num,'times', episode_steps_cnt)
                #print('the steps used ', episode_steps_cnt)

            """ update the obs """
            obs = obs_new

            """ end the while """
            if done or episode_steps_cnt > args.max_episode_num: 
                agent.learn()
                break
