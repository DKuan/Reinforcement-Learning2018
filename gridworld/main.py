"""
this file is for the manage of the agents
Author: Zachary
Reserved Rights
"""

from grid_world import Grid_World
from pi_agent import PI_Agent
import numpy as np

num_grid = 4 # the par for the env
num_action = 4 # the par for how many action the agent can do
env_grid = Grid_World(num_grid, num_action)  # init the env
agent = PI_Agent(num_grid, num_action) # init the agent

for _ in range(5): # evaluation for 5 times, sweep 5 times for all state
	for id_x in range(3, -1, -1):
		for id_y in range(3, -1, -1):
			place_new = [id_x, id_y]
			if tuple(place_new) in env_grid.terminate_state: # do not ypdate the two places
				continue
			agent.state_old = [place_new[0], place_new[1]]
			env_grid.place = [place_new[0], place_new[1]]
			action = agent.act() # for agent1 2
			returns = [list(env_grid.step(False, action_)) for action_ in action] # done the action
			rewards = [return_[0] for return_ in returns] # get the reward
			state_new = [return_[1] for return_ in returns] # get the reward
	
			agent.update(rewards, state_new ) # update the action-value
	
			print(agent.value_state)
			print("*"*30)
