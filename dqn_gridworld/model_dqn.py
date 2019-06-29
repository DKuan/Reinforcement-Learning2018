import gym
import math
import random
import pickle
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler as WRSampler

Transition = namedtuple('Transition', 
        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object): 
    """ This class is for storing the memory showing in the interaction """
    def __init__(self, capacity, num_remove):
        self.capacity = capacity
        #self.memory = [set(), set()]
        self.memory = set()
        self.num_remove = num_remove
        self.index_no_reward = 0
        self.index_reward = 1

    def push(self, *args):
        transition_ = Transition(*args)
        self.memory.add(transition_)

        if len(self.memory) >= self.capacity:
            rm_data = self.sample(self.num_remove)
            for data_ in rm_data:
                self.memory.remove(data_)

    def sample(self,  batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemoryL(object): 
    """ This class is for storing the memory showing in the interaction """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.episode_memory = set()
        self.next = 0

    def push(self, t):
        if self.memory.__len__() < self.capacity:
            self.memory.append(t)
        else:
            self.memory[self.next] = t 
        self.next = (self.next + 1) % self.capacity

    def episode_add(self, *args):
        t = Transition(*args)
        if t not in self.episode_memory: 
            self.episode_memory.add(t)
    
    def update(self, done):
        while self.episode_memory.__len__() > 0 :
                t = self.episode_memory.pop()
                self.push(t)
        self.episode_memory.clear()

    def sample(self,  batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RewardMemory(object): 
    """ This class is for storing the memory showing in the interaction """
    def __init__(self, file_name):
        self.file_name = file_name
    
    def memory_read(self):
        E = list()
        try: 
            F = open(self.file_name, 'rb')
            E = pickle.load(F)
            F.close()
        except:
            F = open(self.file_name, 'wb')
            F.close()
        return E

    def memory_write(self, memory_new):
        F = open(self.file_name, 'wb')
        pickle.dump(memory_new, F)
        F.close()

    def push(self, transitions):
        memory = self.memory_read()
        memory += transitions
        self.memory_write(memory)

    def sample(self,  batch_size):
        memory = self.memory_read()
        if memory.__len__() > batch_size:
            return random.sample(memory, batch_size)
        else:
            return memory

    def __len__(self):
        E = self.memory_read()
        return E.__len__()

""" model make """
def conv2d_size_out(layer_num, size, padding=0, kernel_size=3, stride=1):
    for _ in range(layer_num):
       size = (size + padding*2 - (kernel_size - 1) - 1) // stride + 1
    return size

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        cells_layer1 = 64
        cells_layer2 = 32 
        self.fc3_out = 0
        self.lk_ReLU = torch.nn.LeakyReLU(0.01)
        self.fc_1 = nn.Linear(h * w, cells_layer1)
        self.fc_2 = nn.Linear(cells_layer1, cells_layer2)
        self.fc_3 = nn.Linear(cells_layer2, outputs)

    def forward(self, x):
        x = self.lk_ReLU(self.fc_1(x.flatten()))
        x = self.lk_ReLU(self.fc_2(x))
        #return self.fc(x.view(x.size(0), -1)).flatten()
        self.fc3_out = self.fc_3(x)
        return self.fc3_out
        #x = torch.sigmoid(self.fc_1(x.flatten()))
        #x = torch.sigmoid(self.fc_2(x))
        #return self.fc_3(x)

class DQN_S(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN_S, self).__init__()
        cells_layer1 = 128
        self.lk_ReLU = torch.nn.LeakyReLU(0.01)
        self.fc_1 = nn.Linear(h* w, cells_layer1)
        self.fc_2 = nn.Linear(cells_layer1, outputs)

    def forward(self, x):
        x = self.lk_ReLU(self.fc_1(x.flatten()))
        #return self.fc(x.view(x.size(0), -1)).flatten()
        return self.fc_2(x)

if __name__ == "__main__":
    path = 'model/112.pt'
    model = DQN(10, 10,  5)
    print(model.state_dict())
    input = torch.ones([64, 64])
    input = input.view(1, 1, 64, 64)
    result = model(input)
    loss = torch.nn.functional.mse_loss(result, result)
    loss.backward()
    #model.load_state_dict(torch.load(path))
    #for param_tensor in model.state_dict():
    #    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    #torch.save(model.state_dict(), path)
    
