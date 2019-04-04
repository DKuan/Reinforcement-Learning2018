"""
This file use the Environment from chingzilla(github), If anything is wrong, please tell mt.
The agent uses Monte-Carlo method to search the optimal way in the environment.
Author: Zachary
Rights Reserved
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
import numpy as np

action_list = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]] # nine actions in the agent around

class MC_Agent(Agent_Base):
    """
    The agent is built according the algorithm in page 111, book RL2018
    off-policy MC control, for estimating pi = pi* 
    """ 
    def __init__(self, epsilon, num_action):
                """
                init the agent
                par:
                |num_action: int, the number of the actions
                |epsilon: float, the posibility of the greedy for target policy
                """
                Agent_Base.__init__(self)
                self.num_action = num_action
                self.epsilon = epsilon
                self.value_state_target = {} # store the target value
                self.value_state_behave = {} # store the behave value

 
    def act_behave(self, pos):
                """
                use the behaviour policy to ger the episodes.
                par:
                |return:
                |action list, [x, y]
                """
                if np.random.random() > self.epsilon:
                    _value_state = self.value_state_get(self.value_state_behave, pos) # get the array
                    action = np.where(_value_state.max() == _value_state)[0][0] # greedy state
                else:
                    action = np.random.randint(self.num_action) # epsilon state
                return action_list[action]

        
    def update_behave(self):
                """
                update the value_state of the behave, which will be the same as the target value_state
                """
                self.value_state_behave = self.value_state_target.copy()


    def value_state_get(self, value_dic, key):
                """
                this method is for the value_state dic to show it's value
                par:
                |value_dic:dictionary, the value_state dic need to be read
                """
                if not "{} {}".format(*key) in value_dic.keys(): #make sure the dic has the key
                        value_dic["{} {}".format(*key)] = np.random.random(self.num_action) # set the value
                return value_dic['{} {}'.format(*key)]

        
    def value_state_change(self, value_dic, key, id_th, value):
                """
                this method is for the value_state dic to change the ith value
                """
                if id_th >= self.num_action: # beyond range
                        return

                if "{} {}".format(*key) in value_dic.keys(): #make sure the dic has the key
                        value_dic["{} {}".format(*key)][id_th] = value # set the value
                else:
                        value_dic["{} {}".format(*key)] = np.random.random(self.num_action) # set the value
                        value_dic["{} {}".format(*key)][id_th] = value # change the value


