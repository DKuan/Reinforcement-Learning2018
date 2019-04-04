"""
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
For the policy iteration method of the car rental, you can see the algorithm in the Sutton's RLbook of chapter 4.3, part 2.
Author: Zachary
Reserved Rights. 
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
import numpy as np

class PI_Agent(Agent_Base):
	"""
	this class is the policy iteration agent to find the car retal problem's optimal solution
	"""
	def __init__(self, num_car_max, num_action, gamma):
		"""
		init the data 
		arg:
		num_action: int, the num of the action
		num_cars_max: int, the max num of the cars that can sotp in one place
		gamma: float,0-1, the discount of the last state's value
		"""
		Agent_Base.__init__(self)
		self.value_state = np.zeros([num_car_max, num_car_max]) # to store the value of the state
		self.posibility = 1/(num_action*2+1) # init the choice posibility
		self.posibility_state = self.posibility*np.ones([num_car_max, num_car_max, num_action*2+1]) # use this array to improve
		self.gamma = gamma # the discount arg
		self.action = None # action choose
		self.state_old = None # last state	

	def act(self, state_now, evaluation_flag):
		"""
		use the max rule to make the choice
		arg:
		state_now:list, the num of the car in the place A,B
		evaluation_flag:Bool, if the policy is to  
		"""
		self.state_old = state_now # store the state now
		if evaluation_flag == True: # then the policy is random
			self.action = range(-5, 6, 1)
		else:	# the policy is find the max posibility
			self.action = np.where(self.posibility_state == np.max(self.posibility_state))[0][0] # choose a random number in the size	
		return self.action

	def update(self, reward, state_new, improve_flag):
		"""
		use this func to update the action-value table
		arg:
		improve_flag: bool, to cal if the module is need to improve
		state_new: list, the place that the agent in now
		reward: 0/-1 list, the reward after the action act to the env
		"""
		#print('state old is in agent', self.state_old)
		value_array_cal = [self.posibility * (reward_ + self.gamma*self.value_state[state_[0], state_[1]]) for reward_, state_ in zip(reward, state_new)] # use this array to increment the posibility of maxarg item
		if improve_flag == True:
			self.posibility_state[self.state_old[0], self.state_old[1], np.where(np.max(value_array_cal)==value_array_cal)[0][0]] += 0.01 #add the posibility of the max one
		self.value_state[self.state_old[0], self.state_old[1]] = np.sum(value_array_cal) # cal the new value of the state
