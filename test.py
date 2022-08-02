'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import math
import os
import argparse
from torch.autograd import Variable

from models import *
from utils import progress_bar
import numpy as np 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--lr_global', default=0.2, type=float, help='global learning rate')
parser.add_argument('--num_clients', default=10, type=int, help='number of clients')
parser.add_argument('--K', default=1, type=int, help='how many epoches between two synchronize')
parser.add_argument('--enablesketch', default=False, type=bool, help='whether to enable sketch')
parser.add_argument('--sketchdim', default=0.1, type=float, help='the ratio of sketch dimension / gradient dimension. Lower ratio causes lower communication but lower convergence rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dimension', default=60, type=int, help='dimension of the function')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')




num_of_data = 1000

distribution = 'normal'

def generate_data(dimension, number_of_data):
    # generate data
    torch.manual_seed(0)

    data = torch.rand(number_of_data, dimension)
    label = torch.rand(number_of_data, 1)

    if(distribution == 'normal'):
        print("use normal distribution instead.")
        data = torch.normal( mean=0, std= 1, size=(number_of_data, dimension))
        label = torch.normal( mean=0, std= 1, size=(number_of_data, 1))

    return data, label

train_x, train_y = generate_data(args.dimension, num_of_data)





# Model
print('==> Building model..')



intermediate_layer_size = 30
class LeNetMine(nn.Module):
    def __init__(self):
        super(LeNetMine, self).__init__()
        self.fc1   = nn.Linear(args.dimension, intermediate_layer_size)
        self.fc2 = nn.Linear(intermediate_layer_size, 1)

    def forward(self, x):
        # print(x.shape)
        out = x.view(x.size(0), -1)
        # print(out.shape)
        out = F.relu(self.fc1(out.T))
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        return out

net = LeNetMine()

client_nets = [LeNetMine().to(device) for _ in range(args.num_clients)]

net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    for i in range(args.num_clients):
        client_nets[i] = torch.nn.DataParallel(client_nets[i])
    cudnn.benchmark = True

clients_last_sync_weights = []
for i in range(args.num_clients):
            copied_client_model = copy.deepcopy(client_nets[i])
            clients_last_sync_weights.append(copied_client_model.state_dict())

fedavg_gradients = []



# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss(size_average = False)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optimizer_clients = []
for i in range(args.num_clients):
    optimizer_clients.append(optim.SGD(client_nets[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4))


def update_function(param, grad, loss, learning_rate):
  return param - learning_rate * grad



if(args.enablesketch == True):
    print("enable sketch")




# Training
def train(epoch):
    global clients_last_sync_weights
    print('\nEpoch: %d' % epoch)
    for i in range(args.num_clients):
        client_nets[i].train()
    train_loss = 0
    correct = 0
    total = 0
    total_params_layers = 4
    batch_idx = 0
    total_loss = 0
    for ix in range(num_of_data):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(train_x[ix].cuda())
            labels = Variable(train_y[ix].cuda())
        else:
            inputs = Variable(train_x[ix])
            labels = Variable(train_y[ix])

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer_clients[ix % args.num_clients].zero_grad()

        # get output from the model, given the inputs
        outputs = client_nets[ix % args.num_clients](inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer_clients[ix % args.num_clients].step()

        with torch.no_grad():
            sketch1 = torch.normal(mean=0, std=math.sqrt(1.0/int(intermediate_layer_size * args.dimension * args.sketchdim)), size=(int(intermediate_layer_size * args.dimension * args.sketchdim), intermediate_layer_size * args.dimension)).to(device)
            sketch2 = torch.normal(mean=0, std=math.sqrt(1.0/int(intermediate_layer_size * args.sketchdim)), size=(int(intermediate_layer_size * args.sketchdim), intermediate_layer_size)).to(device)
            # sketch3 = torch.normal(mean=0, std=math.sqrt(1.0/int(256 * args.sketchdim)), size=(int(256 * args.sketchdim), 256)).to(device)

            sketches = [sketch1, sketch2]
            tmp_grad = [[] for _ in range(total_params_layers)]
            # for each client compute grad
            #client compute weight updates
            for i in range(args.num_clients):
                index = 0
                for (weight_name, p) in client_nets[i].state_dict().items():
                    old_p = clients_last_sync_weights[i][weight_name]
                    if(index  % 2 == 0 and index < 4 and args.enablesketch):
                        # print("enable sketched aggregation")
                        #for LC1, LC2 LC3 we compute sketched gradients
                        grad = p - old_p
                        grad  = grad.flatten()
                        # print(grad.shape)
                        sk = torch.matmul(sketches[int(index / 2)], grad)

                        # use below code to compare the diff between the original gradient and sketch then desketched gradient
                        tmp_grad[index].append(sk)
                    else:
                        # becase other layers are very small, we don't need to sketch them.
                        tmp_grad[index].append(p - old_p)
                    index+=1

            # compute fed avg
            for i in range(len(tmp_grad)):
                tmp = tmp_grad[i][0]
                for j in range(1, len(tmp_grad[i])):
                    tmp += tmp_grad[i][j]
                fedavg_gradients.append(tmp/args.num_clients)

            # now we have obtained the fed avg gradients. update each client model. 
            for i in range(args.num_clients):
                index = 0
                cur_dict =client_nets[i].state_dict()
                for (weight_name, old_weight) in clients_last_sync_weights[i].items():
                    # print(args.enablesketch)
                    if(index % 2 == 0 and index < 4 and args.enablesketch):
                        #for LC1, LC2 LC3 layer we compute desketched gradients and update the weights
                        desketch = torch.matmul(sketches[int(index / 2)].T,  fedavg_gradients[index])
                        # print("desketch ",index)
                        if(index == 0):
                            desketch = desketch.view(intermediate_layer_size,args.dimension)
                        elif(index == 2):
                            desketch = desketch.view(1,intermediate_layer_size)
                        new_val = old_weight  + args.lr_global * desketch
                        cur_dict[weight_name] = new_val
                    else:
                        new_val = old_weight + args.lr_global * fedavg_gradients[index]
                        cur_dict[weight_name] = new_val
                    index+=1
                client_nets[i].load_state_dict(cur_dict)

            #update the last sync client weights for compute next round fed avg gradients
            # clients_last_sync_weights = [client_nets[i].parameters() for i in range(args.num_clients)]

            clients_last_sync_weights.clear()
            for i in range(args.num_clients):
                copied_client_model = copy.deepcopy(client_nets[i])
                clients_last_sync_weights.append(copied_client_model.state_dict())

        total_loss += loss.item()
    print('epoch {}, loss {}'.format(epoch, total_loss/num_of_data))


    # if(epoch % args.K == 0):
    #     # sketch and desketch fedavg simulation
    #     print("server aggregation starts")




def test(epoch):
    global best_acc
    # net.eval()
    # for i in range(args.num_clients):
    client_nets[0].eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # for i in range(args.num_clients):
            outputs = client_nets[0](inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # correct /= float(args.num_clients)
            # test_loss /= float(args.num_clients)
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    # test(epoch)
    scheduler.step()

