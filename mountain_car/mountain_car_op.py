"""
This file is created to learn the part 2, chapter 10
The environment is copied from the openai
author: Zachary
all rights reserved
"""
import numpy as np
from mocar_one_sarsa_agent import MCar_Agent
import gym

""" init the par """
alpha = 0.004 # learning rate 0.0004
gamma = 0.98 # discount rate
epsilon = 0.7 # the e-greedy policy
n_step = 8 # for n-steps sarsa
episode_num = 500 # the number of the learning episodes
action_num = 3

""" init the gym env """
env = gym.make('MountainCar-v0').env
agent = MCar_Agent(alpha, gamma, epsilon, n_step, action_num)

""" train the agent """
agent.train(env, episode_num)

""" test the game """
for i_episode in range(10):
    obs = env.reset() 
    for i in range(1000):
        #env.render()
        #action = env.action_space.sample()
        action = agent.act(obs)
        print('obs is {} action is {}'.format(obs, action))
        obs, reward, done, info = env.step(action) # take a random action
        if done:
            print("episode finishd in {} timesteps".format(i+1))
            break
    env.close()
