import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

'''
- Localization network:
STN can be inserted into the pre-defined neural network.

In this summer project, I would use Alex-STN.

- Ranker Network:
Ranker Network can also adopted from other pre-defined network.

In this summer project, I would use AlexNet.
'''


class LocNet(nn.Module):
    '''
    This LocNet is built based on a spatial transformer network.

    In this project, as the paper said,
    - adopt the AlexNet and only use the layers in features (take out the classifier layers)
    - add extra conv layer
    - for the fc_loc: 4608 -> 128 -> 3 (only parameters for the transformation)
    - initilization( didn't do, it will initilize randomly automatically)
        -  See more weight initialization method: https://discuss.pytorch.org/t/parameters-initialisation/20001/10

    Note:
        the size of alexnet input is 3*224*224.
    '''

    def __init__(self):
        super(LocNet, self).__init__()

        # Spatial transformer localization-network
        self.alex_conv = models.alexnet(pretrained=True).features
        self.extra_conv = nn.Conv2d(256, 128, kernel_size=1)

        self.fc_loc = nn.Sequential(
            nn.Linear(4608, 128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        )

        # # Initialize the weights/bias with identity transformation.
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        '''
        Get the scale and transformation parameter, make affine transformation, sampling.
        '''
        xs = self.alex_conv(x)
        xs = self.extra_conv(xs)
        xs = xs.view(-1, 4608)
        theta = self.fc_loc(xs)

        # get the theta weights which is the output of localization net
        # reshape the N*3 theta into N*2*3 theta tensor
        theta_list = []
        for i in theta:
            temp = [[i[0], 0, i[1]], [0, i[0], i[2]]]
            theta_list.append(temp)
        theta = torch.tensor(theta_list
                             )
        # used to do the affine transformation
        # the size if x should N*C*H*W
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)  # sampler. Could be biinearsampler
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


class RankNet(nn.Module):
    '''
    This ranknet is built based on a AlexNet.
    The modification on AlexNet:
        remove the final fully connected layer(since we may need that for the combined branch)
    '''

    def __init__(self):
        super(RankNet, self).__init__()
        self.rank = models.alexnet(pretrained=True)
        self.rank.classifier = self.rank.classifier[:-1]

    def forward(self, x):
        xs = self.rank(x)
        return xs


class SiameseBranch(nn.Module):
    '''
    The branch without the global image in the ranknet
    '''

    def __init__(self):
        super(SiameseBranch, self).__init__()
        self.locnet = LocNet()
        self.ranknet = RankNet()
        self.final_fc = nn.Linear(4096, 1)

    def forward(self, x):
        xs = self.locnet(x)
        xs = self.ranknet(xs)
        xs = self.final_fc(xs)
        return xs


class SiameseCombinedBranch(nn.Module):
    '''
    The branch combined the global image
    '''

    def __init__(self):
        super(SiameseCombinedBranch, self).__init__()
        self.locnet = LocNet()
        self.ranknet = RankNet()
        self.final_fc = nn.Linear(8192, 1)

    def forward(self, x):
        xs = self.locnet(x)  # output: 4096
        x1 = self.ranknet(xs)
        x2 = self.ranknet(x)
        xs = torch.cat((x1, x2))  # it works, and can do the derivation
        xs = self.final_fc(xs)
        return xs


class SiameseNet(nn.Module):
    '''
    Combine the STN and RN into a Siamese network for this experiment.

    args:
        - model_type: 1 or 2. The 1 type represents the network just using local part of image. Type 2 combines the global image

    '''

    def __init__(self, model_type=1):
        '''
        If combined model, then concanate the global image and local part.
        '''
        super(SiameseNet, self).__init__()
        if model_type == 2:
            self.branch = SiameseCombinedBranch()
        else:
            self.branch = SiameseBranch()

    def forward(self, x):
        # return a tuple (v1,v2)
        x1 = self.branch(x[0])
        x2 = self.branch(x[1])
        return x1, x2
