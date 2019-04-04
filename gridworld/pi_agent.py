"""
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
For the policy evaluation method of the gridworld, you can see the algorithm in the Sutton's RLbook of chapter 4.1, part 2.
Author: Zachary
Reserved Rights. 
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
import numpy as np

class PI_Agent(Agent_Base):
	"""
	this class is the policy evaluation agent to find the random policy's state, what the value of the state is
	"""
	def __init__(self, num_grid, num_action):
		"""
		init the data 
		arg:
		num_action: int, the num of the action
		num_grid: int, the number of the gridworld
		"""
		Agent_Base.__init__(self)
		self.value_state = np.zeros([num_grid, num_grid]) # to store the value of the state
		self.posibility = 0.25 # init the choice posibility
		self.action = None # action choose
	
	def act(self):
		"""
		use the random rule to make the choice
		arg:
		"""
		self.action = np.arange(4) # choose a random number in the size	
		return self.action

	def update(self, reward, state_new):
		"""
		use this func to update the action-value table
		arg:
		state_new: list, the place that the agent in now
		reward: 0/-1 list, the reward after the action act to the env
		"""
		#print('state old is in agent', self.state_old)
		self.value_state[self.state_old[0], self.state_old[1]] = np.sum([0.25 * (reward_ + self.value_state[state_[0], state_[1]]) for reward_, state_ in zip(reward, state_new)])
