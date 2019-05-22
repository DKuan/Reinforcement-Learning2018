"""
self.alpha*(G- self.val_cal(state_tao, action_tao)""
This file is writen to model the approximation-Q-Learning
the same structure as the agents of opaigym
Authors:Zachary
Rights Reserved
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base
from tqdm import tqdm
import tile_coding as tc
import numpy as np
import time

MAX_SIZE = 2014
NUM_OF_TILINGS = 8
folder_path = 'data/'

class MCar_Agent(Agent_Base):
    """
    Episodic semi-gradient n-step sarsa
    """
    def __init__(self, alpha, gamma, epsilon, n_step, action_num):
        Agent_Base.__init__(self)
        self.action_num = action_num
        self.alpha = alpha 
        self.epsilon = epsilon
        #self.eps_gradient = eps_gradient
        self.gamma = gamma
        self.n_step = n_step
        #self.w = np.random.random(8) # the weight of the model
        #self.w = np.array([-2.81, 0.70, -0.076]) # the weight of the model
        self.weights = np.zeros(MAX_SIZE)
        self.episode_num = 10000 # the num of the episode train
        self.action_space_random = lambda: np.random.randint(action_num) # return the random number
        self.iht = tc.IHT(MAX_SIZE) 
        
    def act(self, state_new):
        """
        use the e-greedy policy to choose the action
        :arg
        |state_new: list, show the pos, speed of the car
        """

        pair_values = [self.val_cal(state_new, act_) \
            for act_ in range(self.action_num)] # cal the values

        """ e-greedy to choose new action """
        if np.random.random() < self.epsilon:
            try:
                action = np.where(np.max(pair_values) == pair_values)[0][0]
            except:
                print(pair_values, 'the pair values is')
        else:
            action = self.action_space_random() # choose action random
        return action

    def val_cal(self, state, action, gradient=False):
        """
        the feature is constructed by (tiles, action)
        this func has two procedure:
        1: get the feature
        2: cal the value of the (state, action) pair
        :par
        |state: instance for observation
        |action: int, for discrete number
        |gradient: bool, if true, then return the feature
        """
        #print('the state is', [8*(state[0]+1.2)/(0.6+1.2), 8*(state[1]+0.07)/(0.07+0.07)] )
        #print(action,[round(8*(state[0]+1.2)/(0.6+1.2), 3), 
        #            round(8*(state[1]+0.07)/(0.07+0.07), 3)], 'the val cal state is')
        feature = tc.tiles(self.iht, NUM_OF_TILINGS, [8*(state[0])/(0.6+1.2), 8*(state[1])/(0.07+0.07)], [action])
        #feature_gradient = tc.tiles(self.iht, 8, [8*(state[0]+self.eps_gradient[0])/(0.6+1.2), 
         #                       8*(state[1]+self.eps_gradient[1])/(0.07+0.07)], [action])
        #feature = tc.tiles(self.iht, 8, [round(8*(state[0]+1.2)/(0.6+1.2), 3), 
        #            round(8*(state[1]+0.07)/(0.07+0.07), 3)], [action])
        #feature = [8*(state[0]+1.2)/(0.6+1.2), 8*(state[1]+0.07)/(0.07+0.07)]
        if gradient: return feature
        val = sum(self.weights[feature])

        return val

    def e_greedy(self, state_new):
        """
        use e-greedy to get the action
        """
        pair_values = [self.val_cal(state_new, act_) \
                        for act_ in range(self.action_num)] # cal the values

        """ e-greedy to choose new action """
        if np.random.random() < self.epsilon:
            try:
                action = np.where(np.max(pair_values) == pair_values)[0][0]
            except:
                print(pair_values, 'the pair values is')
        else:
            action = self.action_space_random() # choose action random
        return action

    def train(self, env, episode_num):
        """
        use the principle of Episodic semi-gradient n-step sarsa to train the weight
        :arg
        |env: gym object, instance of the class environment
        |spisode_num: int, number of the train step run
        """
        time_now = time.ctime().replace(':', '')
        file_name = 'nstep' + time_now[8:10] + '_' + time_now[11:17] + '.txt'
        data_record = open(folder_path + file_name, 'w') 

        for i in tqdm(range(episode_num)): 
            """ init the par for a new episode """
            state = env.reset() # reset the env for a new episode
            action = self.action_space_random()
            t = 0 # init the time
            T = 50000 # you best done before this time
            tao = -100 # just low than the T is fine
            rewards_memory = [] # every time should update
            actions_memory = []
            states_memory = []

            """ do the episode """
            while(tao < T - 1):
                #if t % 300 == 0:
                #    print('the three is {}, {}, {}'.format(t, tao, T))

                """ game is not end """
                if t <= T:
                    state_new, r_new, done, _ = env.step(action) # do the action
                    if done: # if the game is end
                        T = t
                        print('done in {} time', t)
                        data_record.writelines(str(t)+'\n')
                    else:
                        action = self.e_greedy(state_new)
                    rewards_memory.append(round(r_new, 3)) # store the rewards to cal the return
                    actions_memory.append(action ) # store the action
                    states_memory.append((round(state_new[0], 4), round(state_new[1], 4)))

                """ update weight procedure"""     
                tao = t - self.n_step
                if tao >= 0: # the number of trajectory is bigger than n

                    """ cal the return G """
                    if t < T: # if the tao + n is still in trajectory
                        final_time = t # use the n_step
                        n_step_terminal = self.n_step
                    else: 
                        final_time = T - 1 # tao + n_step is big than T
                        n_step_terminal = T - tao

                    #G = np.sum([r \
                    G = np.sum([r * self.gamma ** idx \
                            for idx, r in enumerate(rewards_memory[1:].copy())]) #return
                    G = round(G, 4) # control the float
                    state_n = states_memory[-2]
                    action_n = actions_memory[-2]
                    G += (self.gamma ** (n_step_terminal - 2) * self.val_cal(state_n, action_n))

                    """ update the delta for weights """
                    state_tao = states_memory[0]
                    action_tao = actions_memory[0]
                    delta = self.alpha*(G - self.val_cal(state_tao, action_tao))
                    features = self.val_cal(state_tao, action_tao, True)
                    self.weights[features] += delta
                    
                    """ update the memory """ 
                    rewards_memory.remove(rewards_memory[0])
                    states_memory.remove(states_memory[0])
                    actions_memory.remove(actions_memory[0])

                    """ for test """
                    #print('the state is', states_memory)
                    #print('the actions is', actions_memory)
                    #print('the reward is', rewards_memory)

                """ update the time """
                t += 1
                #input()
        data_record.close()
