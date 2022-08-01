# import the required packages
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np
# define our data generation function
weight = 1000
height = 100 

A = np.random.rand(weight, height)
b = np.random.rand(weight)

lr = 0.003

def loss(x):
    return np.linalg.norm(np.dot(A, x) - b)

def gradient(x):
    return np.dot(np.dot(A.T, A), x) -   np.dot(A.T, b)

x = np.array([0.0 for i in range(height)])

for i in range(100):
    # print("iter={}out={}\n".format(i, model(x)))
    # print("loss={}\n".format(loss(x)))
    x = x - lr * gradient(x)
    print("iter={} loss={}\n".format(i, np.linalg.norm(loss(x))))