"""
this file is for the manage of the agents
Author: Zachary
Reserved Rights
"""

from car_rental import Car_Rental
from pi_agent import PI_Agent
import numpy as np
from tqdm import tqdm 

num_cars_init = 13 # the par for the env to init the num of two place's cars
num_cars_max = 21 # the max num of the car
num_action = 5 # the par for how many action the agent can do
gamma = 0.9 # the discount for the value of the new state
env_car_rental = Car_Rental(num_cars_init, num_cars_max)  # init the env
agent = PI_Agent(num_cars_max, num_action, gamma) # init the agent

for steps in tqdm(range(400)):
	for id_a in range(num_cars_max):
		for id_b in range(num_cars_max):
			#print('*'*40)
			state_now = [id_a, id_b] # the num of the car in twoplace
			#print('the state now is ', state_now)
			env_car_rental.num_cars = [state_now[0], state_now[1]] # evert time update the state in the env
			action = agent.act(state_now, True) # evaluation stage
			returns = [list(env_car_rental.step(action_)) for action_ in action]
			rewards = [return_[0] for return_ in returns]
			state_new = [return_[1] for return_ in returns]
			improve_flag = True if steps > 400 else False
			agent.update(rewards, state_new, improve_flag) # update the action-value
			#if steps >= 0 and id_a < 3 and id_b <= 3 :
				#print('the posibility is', agent.posibility_state[state_now[0], state_now[1]])
				#print('the state now is ', state_now)

print('the value of state is', agent.value_state)
print('*'*40)
