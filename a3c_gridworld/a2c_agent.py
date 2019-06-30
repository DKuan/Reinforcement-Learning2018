#!/usr/bin/env python3
"""
The third version of the ffai-agent for A3C, the aim is to train the discrate action more efficiently.
time: 19/06/30
Author: FFAI_WD
Rights Reserved
"""
import os
import sys

""" add the path to the sys.path"""
sys.path.append(os.getcwd())

from collections import namedtuple
import gym
import time
import random
import numpy as np

import torch
import torch.nn as nn
from model import Model 
import torch.optim as optim
from torch.distributions import Categorical


action_num = 4
action_list = np.arange(action_num)

class RL_AGENT_A3C():
    """
    RL agent class
    """
    def __init__(self, lr, gamma, board, device, model_path, \
            r_memory_Fname, o_model_name, model_load=False ):
        self.step_now = 0 # record the step
        self.step_last_update = 0 # record the last update time 
        self.gamma = gamma
         
        self.device = device
        self.model_path = model_path
        self.mode_enjoy = model_load
        if model_load == False: 
            self.ac_model = Model(board[0], board[1], action_num).to(device)
            self.optimizer = optim.Adagrad(self.ac_model.parameters(), lr=lr)
            self.saved_log_probs = []
            self.saved_value = []
            self.saved_r = []
            self.saved_dones = []
            self.entropy = 0
        else:
            self.load(o_model_name) 
        self.obs_new = None
        self.obs_old = None
        self.action = None
        self.action_old = None
    
    def load(self, old_model):
        """
        load the trained model
        par:
        |old_model:str, the name of the old model
        """
        model_path_t = self.model_path + 't' + old_model
        self.target_net = torch.load(model_path_t, map_location=self.device)
        self.target_net.eval()
        print('target net par', self.target_net.state_dict())

    def save(self):
        """
        save the trained model
        """
        t = time.strftime('%m%d%H%M%S')
        self.model_path_p = self.model_path + t + '.pt'
        torch.save(self.ac_model, self.model_path_p)
    
    def save_trace(self, r, done):
        """
        save the s, r, done for learn, episode
        par:
        |s: numpy, state of the game now
        |r: float, the reward now
        |done: bool, if the game is end
        """
        self.saved_r.append(r)
        self.saved_dones.append(done)

    def learn(self):
        """
        This func is used to learn the agent
        par:
        |env: class-Environment, use it for nothing
        """
        R = 0 # the return accumulated
        a_loss_all = [] # actor loss
        c_loss_all = [] # critic loss
        for log_p, s_v, r, done in zip(self.saved_log_probs[::-1], self.saved_value[::-1], \
                self.saved_r[::-1], self.saved_dones[::-1]):
            R = r if done else r + self.gamma*R
            adv = R - s_v
            c_loss_all.append(adv.pow(2))
            a_loss_all.append(log_p * adv)
        c_loss = sum(c_loss_all)/c_loss_all.__len__()
        a_loss = sum(a_loss_all)/a_loss_all.__len__()
        loss = a_loss + 0.5 * c_loss - 0.001 * self.entropy

        """ update the par """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """ clear the list """
        self.entropy = 0
        for l in [self.saved_log_probs, self.saved_value, \
                self.saved_r, self.saved_dones]:
            l.clear()

    def select_action(self, state):
        """
        select the action by the prob caled by the actor-net 
        arg:
        |state: tensor, the state of game now 
        """
        value, probs = self.ac_model(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_value.append(value)
        self.entropy += m.entropy().mean()
        return action.item()

    def feature_combine(self, obs):
        """ 
        This file extract features from the obs.layers and 
        combine them into a new feature layer
        Used feature layers:    
        """
        """ combine all the layers """
        feature_c = obs.copy()
        feature_c = feature_c.astype(np.float32)
        feature_c = torch.tensor(feature_c, dtype=torch.float32, device=self.device)
        size = feature_c.shape
        feature_c = feature_c.resize_(1, 1, size[0], size[1])
        return feature_c

    def data_augment(self, transition):
        """
        use this func to flip the feature, to boost the experience,
        deal the problem of sparse reward
        par:
        |transition: tuple, with (feature_o, action, feature_n, reward) 
        """
        flip_ver_dim = 2
        feature_old = transition[0]
        action = transition[1]
        feature_new = transition[3]
        reward = transition[2]

        """ vertical flip """
        feature_o_aug = feature_old.flip([flip_ver_dim])
        feature_n_aug = feature_new.flip([flip_ver_dim])

        """ vertical :action flip """
        if action == 0:  action = 1
        elif action == 1: action = 0

        return feature_o_aug, action, reward, feature_n_aug

    def act(self, map):
        """ this func is interact with the competition func """
        state_old = self.feature_combine(map) # get the feature
        action = self.select_action(state_old)

        return action

    def act_enjoy(self, map):
        """ this func is interact with the competition func """
        state_old = self.feature_combine(map) # get the feature
        action = self.select_action(state_old)

        return action
