'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import numpy as np 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--num_clients', default=10, type=int, help='number of clients')
parser.add_argument('--K', default=1, type=int, help='how many epoches between two synchronize')
parser.add_argument('--enablesketch', default=False, type=bool, help='whether to enable sketch')
parser.add_argument('--sketchdim', default=0.5, type=float, help='the ratio of sketch dimension / gradient dimension. Lower ratio causes lower communication but lower convergence rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = LeNet()

client_nets = [LeNet().to(device) for _ in range(args.num_clients)]

net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    for i in range(args.num_clients):
        client_nets[i] = torch.nn.DataParallel(client_nets[i])
    cudnn.benchmark = True

clients_last_sync_weights = [client_nets[i].parameters() for i in range(args.num_clients)]
fedavg_gradients = []


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# TODO(lqin) we need to support momentum and weight_decay
def update_function(param, grad, loss, learning_rate):
  return param - learning_rate * grad


sketch1 = torch.normal(mean=0, std=math.sqrt(1.0/int(1024 * args.sketchdim)), size=(int(1024 * args.sketchdim), 1024)).to(device)
sketch2 = torch.normal(mean=0, std=math.sqrt(1.0/int(512 * args.sketchdim)), size=(int(512 * args.sketchdim), 512)).to(device)
sketch3 = torch.normal(mean=0, std=math.sqrt(1.0/int(256 * args.sketchdim)), size=(int(256 * args.sketchdim), 256)).to(device)

sketches = [sketch1, sketch2, sketch3]

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
    total_params_layers = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = client_nets[batch_idx % args.num_clients](inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # print('gradients =', [x.grad.data.shape  for x in net.parameters()] )
        # optimizer.step()
        with torch.no_grad():
            index = 0
            for p in client_nets[batch_idx % args.num_clients].parameters():
                # print(p.grad.data.shape)
                if(index  % 2 == 0 and index < 6 and args.enablesketch):
                    sk = torch.matmul(sketches[int(index / 2)], p.grad.data)
                    desk = torch.matmul(sketches[int(index / 2)].T, sk)
                    new_val = update_function(p, desk, loss, args.lr)
                    p.copy_(new_val)
                    
                else:
                    new_val = update_function(p, p.grad, loss, args.lr)
                    p.copy_(new_val)
                index += 1
            total_params_layers = index
        # print('weights after backpropagation = ',   list(net.parameters())) 

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # if(epoch % args.K == 0):
    #     # sketch and desketch fedavg simulation
    #     print("server aggregation starts")
    #     with torch.no_grad():
    #         tmp_grad = [[] for _ in range(total_params_layers)]
    #         for i in range(args.num_clients):
    #             index = 0
    #             for (p, old_p) in zip(client_nets[i].parameters(), clients_last_sync_weights[i]):
    #                 tmp_grad[index].append(p - old_p)
    #                 index+=1
    #         for i in range(len(tmp_grad)):
    #             tmp = tmp_grad[i][0]
    #             for j in range(1, len(tmp_grad[i])):
    #                 tmp += tmp_grad[i][j]
    #             fedavg_gradients.append(tmp/args.num_clients)

    #         # now we have obtained the fed avg gradients. update each client model. 
    #         for i in range(args.num_clients):
    #             index = 0
    #             for (p, old_weight) in zip(client_nets[i].parameters(), clients_last_sync_weights[i]):
    #                 new_val = old_weight + fedavg_gradients[index]
    #                 p.copy_(new_val)
    #                 # print(p.is_leaf)
    #                 # print(fedavg_gradients[index].is_leaf)
    #                 index+=1

    #         #update the last sync client weights for compute next round fed avg gradients
    #         clients_last_sync_weights = [client_nets[i].parameters() for i in range(args.num_clients)]



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
    test(epoch)
    scheduler.step()
