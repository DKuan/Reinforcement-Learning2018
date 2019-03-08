"""
file name: Gradient Agent
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
Author: Zachary
Reserved rights
"""
import sys
sys.path.append('../')

import numpy as np
import torch
from base.Agent_Base import Agent_Base 

class Gradient_Agent(Agent_Base):
	"""
	this class is the k arm agent for gradient method to realize RL algorithm
	"""
	def __init__(self, alpha, action_num):
		"""
		init the data 
		arg:
		action_num: int, the num of the action
		alpha: float, 0-1, the positive value of the step-size parameter
		"""
		Agent_Base.__init__(self)
		self.action_preference = np.zeros(action_num) # init action pres with the random number
		self.action_cnt = np.zeros(action_num) # to cnt the time that the actions' used
		self.alpha = alpha # step-size par
		#self.action = None 
		self.reward_all = 0 # set the reward to zero
	
	def act(self):
		"""
		use the gradient rule to make the choose
		"""
		softmax_ = torch.nn.Softmax(dim=0) # register the fuc 
		self.action_phi = np.array(softmax_(torch.Tensor(self.action_preference)).tolist()) # use the softmax to the action_preference to get the probability of the action	
		max_id = np.where(self.action_phi == np.max(self.action_phi))[0] # find the most value
		if not max_id.__len__() == 1:
			action = max_id[np.random.randint(max_id.__len__())] # choose a random number	
		else:
			action = max_id[0] # use the max value
		
		self.action_cnt[action] += 1 # update the cnt
		return action

	def update(self, action, reward):
		"""
		use this func to update the preference of the action
		arg:
		action: int, the last action interact with the env
		reward: float the reward after the action act to the env
		"""
		reward_avg = self.reward_all / np.sum(self.action_cnt) # get the avg reward until now
		#print(reward_avg, "the avg reward is")
		for i in range(self.action_preference.__len__()): # use the update rule to update the action preference
			if i == action:
				self.action_preference[i] += self.alpha * (reward - reward_avg) * (1 - self.action_phi[action])
			else:
				self.action_preference[i] -= self.alpha * (reward - reward_avg) * self.action_phi[action]

		self.reward_all += reward # record all the reward the agent get
