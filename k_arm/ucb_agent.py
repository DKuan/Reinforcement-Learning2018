"""
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
For the UCB method of the k-arm-bandit, you can see the algorithm in the Sutton's RLbook of chapter 2.8, part 2.
Author: Zachary
Reserved rights. 
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
import numpy as np

class UCB_Agent(Agent_Base):
	"""
	this class is the k arm agent for action_value method to realize RL algorithm
	Upper-Confidence-Bound Action Selection
	"""
	def __init__(self, c, action_num):
		"""
		init the data 
		arg:
		action_num: int, the num of the action
		c: float, the number which shows the percentage of the agent's exploration
		"""
		Agent_Base.__init__(self)
		self.action_value = np.random.rand(action_num) # init the action with the random number
		self.action_cnt = np.ones(action_num) # to cnt the actions used time
		self.c = c # exploration percentage
		self.action = None 
		self.reward_all = 0 # set the reward to zero
	
	def act(self):
		"""
		use the UCB rule to make the choice
		arg:
		t: int, the time of now
		"""
		times_go = np.sum(self.action_cnt) # use the action choose cnt to cal the times go
		choose_standard = [action_val + self.c*np.sqrt(np.log(times_go) / (0.01 + self.action_cnt[action_id])) for action_id, action_val in enumerate(self.action_value)]
		max_id = np.where(choose_standard == np.max(choose_standard))[0] # find the most value
		
		if not max_id.__len__() == 1:
			action = max_id[np.random.randint(max_id.__len__())] # choose a random number in the size	
		else:
			action = max_id[0] # use the max value

		self.action_cnt[action] += 1 # update the cnt
		
		return action

	def update(self, action, reward):
		"""
		use this func to update the action-value table
		arg:
		action: int, the last action interact with the env
		reward: float the reward after the action act to the env
		"""
		old_value = self.action_value[action] # stand the old value for cal
		self.action_value[action] += (reward - old_value) / (0.001 + self.action_cnt[action]) # UCB update rule
		self.reward_all += reward # record all the reward the agent get

