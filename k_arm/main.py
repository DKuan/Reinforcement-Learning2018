"""
this file is for the manage of the agents
"""

from K_Arm_Bandit import K_Arm_Bandit as Bandit
from Epsilon_Greedy_Agent import Greedy_Agent 
from UCB_Agent import UCB_Agent
from Gradient_Agent import Gradient_Agent

env_bandit = Bandit(8, 1200) # for three agents max steps

agent1 = Greedy_Agent(0, 8) # ready for three agents
agent2 = Greedy_Agent(0.1, 8)
#agent3 = Greedy_Agent(0.3, 8)
agent3 = UCB_Agent(2, 8)
agent4 = Gradient_Agent(0.2, 8)

done = False # end flag
time_cnt = 0 # for cnt

while(not done):
	time_cnt += 1 # update the cnt
	#print("Time", time_cnt, "_"*40)
	for id, agent in enumerate([agent1, agent2, agent3, agent4]):
		action = agent.act() # for agent1 2
		reward = env_bandit.step(action) # done the action
		agent.update(action, reward) # update the action-value
		done = env_bandit.done # update the done flag

for id, agent in enumerate([agent1, agent2, agent3, agent4]):
	print("the final cnt agent{}".format(id+1), agent.action_cnt )

print("the env mean is ", env_bandit.uniform_)
print("reward 1 is ", agent1.reward_all)
print("*" * 20)
print("reward 2 is ", agent2.reward_all)
print("*" * 20)
print("reward 3 is ", agent3.reward_all)
print("*" * 20)
print("reward 4 is ", agent4.reward_all)
print("*" * 20)

