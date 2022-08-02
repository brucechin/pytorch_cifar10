from cProfile import label
from re import S
import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math
import matplotlib.pyplot as plt
import random

num_epochs = 100

def read_file(filename):
    res = []
    with open(filename, 'r') as f:
        while(True):
            data = f.readline()
            if(data == ''):
                break
            if('loss' in data):
                # print(data)
                data = float(data.split(' ')[3])
                res.append(data)
    # print(res[:num_epochs])
    return res[:num_epochs]

uniform_non_sketch = read_file('uniform_non_sketch.log')
uniform_sketch_ratio_01 = read_file('uniform_sketch_ration_0.1.log')
uniform_sketch_ratio_03 = read_file('uniform_sketch_ratio_0.3.log') 

normal_non_sketch = read_file('normal_non_sketch.log')
normal_sketch_ratio_01 = read_file('normal_sketch_ratio_0.1.log')
normal_sketch_ratio_03 = read_file('normal_sketch_ratio_0.3.log') 

ticks_size = 15
x = np.linspace(0, len(uniform_non_sketch), len(uniform_non_sketch))
plt.figure()
plt.plot(x, uniform_non_sketch, label = "No sketch", marker='o')
plt.plot(x, uniform_sketch_ratio_01, label = "Sketch compression ratio 0.1", marker='o')
plt.plot(x, uniform_sketch_ratio_03, label = "Sketch compression ratio 0.3", marker='o')
plt.xlabel("Training Epoch", fontsize= ticks_size)
plt.ylabel("Loss", fontsize= ticks_size)
plt.legend(loc='best', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("uniform.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')