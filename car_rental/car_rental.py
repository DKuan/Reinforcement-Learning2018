"""
This file is wrote for the car rental env, conclude a main module named Car_Rental
The class is inherit from the class of Envrionment_Base, and rewrite two methods
Author: Zachary
Reserved Rights
"""
import sys
sys.path.append('../')

from base.Environment_Base import Environment_Base
import numpy as np

class Car_Rental(Environment_Base):
	"""
	Rewrite the Env_base and realize the file of Car_Rental
	"""
	def __init__(self, num_cars_init, num_cars_max):
		"""
		for init the env
		arg:
		num_cars_init: the num of the cars of the two locations
		num_cars_max: the max num of the cars in the two locations
		"""
		Environment_Base.__init__(self)
		self.num_cars = [num_cars_init, num_cars_init] # record the num of the cars
		self.num_cars_max = num_cars_max # the max is the init num
		self.done = False # if the episode is over
		self.reward = None # return the reward to the agent every step
		self.cars_rent = [3, 4] #to get the num of the rent cars		
		self.cars_return = [3, 4] #get the num of the return cars		
		self.credit_one_car = 10
		self.cost_trans = 2

	def range_check(self, num, diff_flag):
		"""
		this func for number check
		return the diff between the num with the standard 
		arg:
		num: the num should be checked
		diff_flag: bool, if true, then return the value of the diff
		"""
		if num >= self.num_cars_max:
			diff = 0
			val_return = self.num_cars_max - 1
		elif num < 0 :
			diff = 0 - num
			val_return = 0
		else:
			diff = 0
			val_return = num

		if diff_flag == True:
			return val_return, diff
		else:
			return val_return

	def step(self, action):
		"""
		act the action which the agent do, and return the reward of this action
		arg:
		action: int, -5~5, show the num of cars transported from A to B
		"""
		# update the nights change of the cars' number in the two places
		num_trans = action if min([self.num_cars[0], self.num_cars[1]])>abs(action) else min(self.num_cars[0], self.num_cars[1]) 
		#print('the num of trans ', num_trans)
		self.num_cars = [self.range_check(self.num_cars[0]-num_trans, False), self.range_check(self.num_cars[1]+num_trans, False)]	
		cars_rent_ =np.array([np.random.poisson(self.cars_rent[0]), np.random.poisson(self.cars_rent[1])]) # get the cars rent
		cars_return_ = np.array([np.random.poisson(self.cars_return[0]), np.random.poisson(self.cars_return[1])]) # get the cars rent
		#print(self.num_cars, 'the num of the car i have')
		#print(np.add(-cars_rent_, cars_return_), 'the real change of the car')
		#print(cars_rent_, 'the num of the car to rent')
		rent_cal = np.add(self.num_cars, -cars_rent_) # for cal the car that can be rent
		self.num_cars = np.add(np.add(-cars_rent_, cars_return_), self.num_cars) # get the nights' num of the cars
		_, diff_a = self.range_check(rent_cal[0], True) # check the place A's car
		_, diff_b = self.range_check(rent_cal[1], True) # check the place B's car
		num_a, _ = self.range_check(self.num_cars[0], True) # check the place B's car
		num_b, _ = self.range_check(self.num_cars[1], True) # check the place B's car
		self.num_cars = np.array([num_a, num_b]) # num of the car, after the rent and return in the day
		cars_rent_ =np.array([cars_rent_[0]-diff_a, cars_rent_[1]-diff_b]) # remove the error num of the rent
		#print('the real rent car is ', cars_rent_)
		#print('the diff is ' , diff_a, diff_b)
		self.reward = np.sum(cars_rent_) * self.credit_one_car - abs(action) * self.cost_trans # the reward of this day

		return self.reward, self.num_cars



