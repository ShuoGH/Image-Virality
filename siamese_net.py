import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    '''
    Combine the STN and RN into a Siamese network for this experiment.
    '''

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


class RankerNet(nn.Module):
    '''
    Adopt th AlexNet for this experiment.

    Further work:
        Add more models and make them available.
    '''

    def __init__(self):
        super(RankerNet, self).__init__()

    def forward(self, x):
        return x


class STN(nn.Module):
    '''
    This is a spatial transformer network.
    '''

    def __init__(self):
        super(STN, self).__init__()

    def forward(self, x):
        return x
