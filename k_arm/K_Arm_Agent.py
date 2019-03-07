"""
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
import numpy as np

class K_Arm_GreedyAgent(Agent_Base):
	"""
	this class is the k arm agent for action_value method to realize RL algorithm
	"""
	def __init__(self, epsilon, action_num):
		"""
		init the data 
		arg:
		action_num: int, the num of the action
		epsilon: float, 0-1, the number of the non-greedy
		"""
		Agent_Base.__init__(self)
		self.action_value = np.random.rand(action_num) # init the action with the random number
		self.action_cnt = np.zeros(action_num) # to cnt the actions used time
		self.epsilon = epsilon
		self.action = None 
		self.reward_all = 0 # set the reward to zero
	
	def act(self):
		"""
		use the epsilon-greedy rule to make the choose
		"""
		if np.random.rand() > self.epsilon: #use the greedy rule
			max_id = np.where(self.action_value == np.max(self.action_value)) # find the most value
			#print("the max id is ", max_id)
			#print("the action value is ", self.action_value)

			if not max_id.__len__ == 1:
				action = max_id[np.random.randint(max_id.__len__())][0] # choose a random number in the size	
			else:
				action = self.action_value[max_id][0] # use the max value
  
		else:
			action = np.random.randint(self.action_value.size) # choose randomly
		
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
		self.action_value[action] += (reward - old_value) / (0.001 + self.action_cnt[action])
		self.reward_all += reward # record all the reward the agent get

