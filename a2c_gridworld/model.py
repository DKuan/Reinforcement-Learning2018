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

""" model make """
def conv2d_size_out(layer_num, size, padding=0, kernel_size=3, stride=1):
    for _ in range(layer_num):
       size = (size + padding*2 - (kernel_size - 1) - 1) // stride + 1
    return size

class Model(nn.Module):
    def __init__(self, h, w, outputs):
        super(Model, self).__init__()
        cells_layer1 = 256 
        a_out = outputs 
        c_out = 1 
        self.lk_ReLU = torch.nn.LeakyReLU(0.01)
        self.softmax = torch.nn.Softmax()
        self.fc_1 = nn.Linear(h * w, cells_layer1)
        #self.fc_2 = nn.Linear(cells_layer1, cells_layer2)
        self.critic_o = nn.Linear(cells_layer1, c_out)
        self.actor_o = nn.Linear(cells_layer1, a_out)

    def forward(self, x):
        x = self.lk_ReLU(self.fc_1(x.flatten()))
        #x = self.lk_ReLU(self.fc_2(x))
        value = self.critic_o(x)
        prob = self.softmax(self.actor_o(x))
        return value, prob 

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
    
