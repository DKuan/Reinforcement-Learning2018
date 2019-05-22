"""
self.alpha*(G- self.val_cal(state_tao, action_tao)""
This file is writen to model the approximation-Q-Learning
the same structure as the agents of opaigym
Authors:Zachary
Rights Reserved
"""
import sys
sys.path.append('../')

from tqdm import tqdm
import tile_coding as tc
import numpy as np
import time

MAX_SIZE = 2014
NUM_OF_TILINGS = 8
folder_path = 'data/'

class MCar_Agent():
    """
    Episodic semi-gradient one-step sarsa
    """
    def __init__(self, alpha, gamma, epsilon, n_step, action_num):
        self.action_num = action_num
        self.alpha = alpha 
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_step = n_step
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
        #print(feature, 'in cal val feature')
        val = sum(self.weights[feature]) # cal the sum 
        #return val[0]
        return val

    def e_greedy(self, state_new):
        """
        use e-greedy to get the action for new state
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
        file_name = 'one' + time_now[8:10] + '_' + time_now[11:17] + '.txt'
        data_record = open(folder_path + file_name, 'w')

        for i in tqdm(range(episode_num)): 
            t = 0

            """ init the par for a new episode """
            state = env.reset() # reset the env for a new episode
            action = self.action_space_random()

            """ do the episode """
            while(True):
                #print('the three is {}, {}, {}'.format(t, tao, T))
                """ game is not end """
                state_value = self.val_cal(state, action)
                state_new, r_new, done, _ = env.step(action) # do the action
                features = self.val_cal(state.copy(), action, True)
                if done: # if the game is end
                    delta = r_new - state_value
                    self.weights[features] += self.alpha * delta 
                    print('done in {} time', t)
                    data_record.writelines(str(t) + '\n')
                    break
                else:
                    action_new = self.e_greedy(state_new.copy())
                    state_new_value = self.val_cal(state_new, action_new)
                    delta = r_new + (self.gamma * state_new_value) - state_value
                    self.weights[features] += self.alpha * delta
                    state = state_new
                    action = action_new
                """ update the time """
                t += 1
                #input()
        data_record.close()
