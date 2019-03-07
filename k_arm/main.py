"""
this file is for the manage of the agents
"""

from K_Arm_Bandit import K_Arm_Bandit as Bandit
from K_Arm_Agent import K_Arm_GreedyAgent as Greedy_Agent 

env_bandit = Bandit(8, 1200) # for three agents max steps

agent1 = Greedy_Agent(0, 8) # ready for three agents
agent2 = Greedy_Agent(0.1, 8)
agent3 = Greedy_Agent(0.3, 8)

done = False # end flag

while(not done):
	for id, agent in enumerate([agent1, agent2, agent3]):
		action = agent.act() # choose a action
		reward = env_bandit.step(action) # done the action
		agent.update(action, reward) # update the action-value
		done = env_bandit.done # update the done flag

for id, agent in enumerate([agent1, agent2, agent3]):
	print("the final cnt agent{}".format(id+1), agent.action_cnt )

print("the env mean is ", env_bandit.uniform_)
print("reward 1 is ", agent1.reward_all)
print("*" * 20)
print("reward 2 is ", agent2.reward_all)
print("*" * 20)
print("reward 3 is ", agent3.reward_all)
print("*" * 20)

