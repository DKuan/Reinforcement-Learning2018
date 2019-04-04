"""
this file is for the manage of the agents
Author: Zachary
Reserved Rights
"""

from src.track import Track, Car
from mc_agent import MC_Agent
import numpy as np
from tqdm import tqdm 

num_action = 9 # the par for how many action the agent can do
epsilon = 0.9 # the discount for the value of the new state
env_track = Track('data/L-track.txt')  # init the env
env_car = Car(env_track)
agent = MC_Agent(epsilon, num_action) # init the agent

for steps in tqdm(range(100000000)):
    #print('*'*40)
    #print(env_car.pos, "the pos is ")
    action = agent.act_behave(env_car.pos) # evaluation stage
    reward = env_car.step(action) # interact with the env
    #print("the reward is", reward)
    if reward == 1:
        print(env_track)
        env_car.reset()

    agent.update() # update the action-value

print('the value of state is')
print('*'*40)
