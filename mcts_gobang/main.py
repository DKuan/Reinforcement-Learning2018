"""
this file is for the manage of the agents
Author: Zachary
Reserved Rights
"""

from mcts_agent_final import MCTS_Agent
from env_gobang.gomoku import Gomoku
from mcts_tree import Node
import numpy as np

max_mcts_episode = 1000 # the episode's number for agent to run the complete mcts
max_mcts_simulation = 18 # the steps that the simulation runs
board_size_ = 8
num4win_ = 5

""" init the env agent """
env = Gomoku(board_size=board_size_, num4win=num4win_)
agent = MCTS_Agent(env, max_mcts_episode=max_mcts_episode, max_mcts_simulation=max_mcts_simulation) # init the agent

""" show the example for player """
print('You should input data like this')
print('for example:3 3')
print('for example:5 3')

""" for, people play with the agent """
for j in range(50):
    env.reset()
    for i in range(60):

        print('your time to go', '*'*30)
        print('please input your action:')
        env.current_player_id = -1 # for 'o'
        str_action = input()
        action = int(str_action[0])*board_size_+int(str_action[2])
        data_return = env.step(action)
       
        """ show the board state and check if the game is over """
        env.visualize() 
        if data_return['terminal'] == True: break
        
        print('x time to go', '*'*30)
        print('please wait, the agent is calculating')
        env.current_player_id = 1 # for 'x'
        action = agent.act()
        data_return = env.step(action)

        """ show the board state and check if the game is over """
        env.visualize() 
        if data_return['terminal'] == True: break
