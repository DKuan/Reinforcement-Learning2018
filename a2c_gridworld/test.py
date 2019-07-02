import torch
import numpy as np
import random
from collections import namedtuple

a = torch.tensor(1.0, requires_grad=True)
b = a + 1
c = ( 1 - b ) ** 2
loss = 1 - c
a.zero_grad()
loss.backward()
print(c.grad)
#data_t = [a, a, a]
#data_p = [a+1, a+2, a+3]
#data_torch_t = torch.tensor(data_t, requires_grad=True)
#data_torch_p = torch.tensor(data_p, requires_grad=True)
#loss = torch.nn.MSELoss()
#output = [loss(d_p, d_t) for d_p, d_t in zip(data_t, data_p)]
#loss_value = sum(output) / 3
#print(loss_value)

#Transition = namedtuple('Transition', ('state', 'action', 'done'))
#t1 = Transition(torch.ones((3,4)), 2, True)
#t2 = Transition(torch.ones((3,4)), 2, True)
#t3 = Transition(2, 4, True)
#t4 = Transition(26, 4, True)
