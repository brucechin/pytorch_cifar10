# import the required packages

import numpy as np
# define our data generation function
weight = 10000
height = 1000
sketch_size = 10
A_global = np.random.rand(weight, height)
b = np.random.rand(weight)
num_clients = 100

def loss(x):
    tmp = np.linalg.norm(np.dot(A_global, x) - b) 
    return 0.5 * tmp * tmp 

def gradient(x, s):
    A = A_global[s,:]
    b_local = b[s]
    return np.dot(np.dot(A.T, A), x) -   np.dot(A.T, b_local)

x_clients = [np.array([0.0 for i in range(height)]) for i in range(num_clients)]

lr = 0.0000005

for i in range(20):
    sketch = np.random.normal(loc =0 , scale = np.sqrt(1/sketch_size), size = (sketch_size, height))
    sketch = np.eye(height)
    partition_size = int(weight/num_clients)
    sketch_grad_list = []
    for c in range(num_clients):
        c_range = range(c * partition_size, (c+1)*partition_size)
        sketched_grad = np.dot(sketch, gradient(x_clients[c], c_range))
        sketch_grad_list.append(sketched_grad)
    sketch_grad = np.array(sketch_grad_list)
    sketch_grad = np.mean(sketch_grad)
    desketched_grad = np.dot(sketch.T, sketched_grad)
    total_loss = 0
    for c in range(num_clients):
        x_clients[c] = x_clients[c] - lr * desketched_grad
        total_loss += loss(x_clients[c])

    print("iter={} loss={}\n".format(i, total_loss/num_clients))