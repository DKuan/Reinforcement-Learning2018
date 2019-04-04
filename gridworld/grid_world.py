"""
This file is wrote for the gridworld env, conclude a main module named Grid_World
The class is inherit from the class of Envrionment_Base, and rewrite two methods
"""
import sys
sys.path.append('../')

from base.Environment_Base import Environment_Base
import numpy as np

class Grid_World(Environment_Base):
	"""
	Rewrite the Env_base and realize the file of grid_world
	"""
	def __init__(self, num_grid, num_actions):
		"""
		for init the env
		arg:
		num_grid: the num of the grids
		num_action: the num of the action
		"""
		Environment_Base.__init__(self)
		self.num_grid = num_grid # record the num of the grids
		self.num_actions = num_actions # record the num of the actions
		self.done = False # if the episode is over
		self.reward = None # return the reward to the agent every step
		self.terminate_state = [(0, 0), (num_grid-1, num_grid-1)] # the terminate state
	
	def range_check(self, num):
		"""
		this func for number check
		return the right number 
		arg:
		num: the num should be checked
		"""
		if num >= self.num_grid:
			return self.num_grid - 1
		elif num < 0 :
			return 0
		else:
			return num

	def step(self, in_place, action):
		"""
		act the action which the agent do, and return the reward of this action
		arg:
		in_place:true, then change the place number
		action: int, 0-3, up down left right
		"""
		old_place = [self.place[0], self.place[1]] # store the old place

		if action == 0: # up
			self.place[1] -= 1
		elif action == 1: # down
			self.place[1] += 1
		elif action == 2: # left
			self.place[0] -= 1
		elif action == 3: # right
			self.place[0] += 1
		else:
			self.reward = -1 
			
		self.place[0] = self.range_check(self.place[0]) # check the right range of the place	
		self.place[1] = self.range_check(self.place[1])	
		
		if tuple(self.place) in self.terminate_state:
			self.reward = 0
		else:
			self.reward = -1
			
		place_return = self.place  
		self.place = self.place if in_place == True else old_place

		return self.reward, place_return



