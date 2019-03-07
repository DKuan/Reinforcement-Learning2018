"""
This file is wrote for the k-arm bandit env, conclude a main module named K_Arm_Bandit
The class is inherit from the class of Envrionment_Base, and rewrite three methods
"""
import sys
sys.path.append('../')

from base.Environment_Base import Environment_Base
import numpy as np

class K_Arm_Bandit(Environment_Base):
	"""
	Rewrite the Env_base and realize the file of k arm bandit
	"""
	def __init__(self, k, max_steps):
		"""
		for init the env
		arg:
		k: the num of the arms
		max_steps: the num of the max steps in one episode
		"""
		Environment_Base.__init__(self)
		self.k = k # record the k for reset
		self.max_steps = max_steps # record the max steps
		self.normal_sigma = 0.7 # the sigma of every normal distribution
		self.uniform_ = [np.random.rand()*8 - 4 for i in range(k)] # for final check
		self.actions = lambda mean :np.random.normal(mean, self.normal_sigma)
		self.steps = 0 # record the steps of the steps
		self.done = False # if the episode is over
		self.reward = None # return the reward to the agent every step
	
	def reset(self):
		"""
		reset the env including reset the flags and records
		"""	
		self.steps = 0
		self.done = False
		self.reward = None
		self.actions = [lambda mean: np.random.normal(mean, self.normal_sigma) for i in range(self.k)]
	
	def step(self, action):
		"""
		act the action which the agent do, and return the reward of this action
		arg:
		action: int, in the range of k
		"""
		if not self.steps < self.max_steps:
			self.done = True # set the flag
			return 0 # make sure is sth wrong
		
		try:
			self.reward = self.actions(self.uniform_[action]) # choose one action to get the reward
			self.steps += 1 # update the step 
		except IndexError:
			return 0 # show this action is wrong

		return self.reward



