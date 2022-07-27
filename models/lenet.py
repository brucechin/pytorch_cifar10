'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.fc1   = nn.Linear(3072, 1650)
        self.fc2   = nn.Linear(1650, 512)
        self.fc3   = nn.Linear(512, 84)
        self.fc4   = nn.Linear(84, 10)

    def forward(self, x):
        # print(x.shape)
        out = x.view(x.size(0), -1)
        # out = F.relu(self.conv1(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        return out
