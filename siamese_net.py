import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

'''
- Localization network:
STN can be inserted into the pre-defined neural network.
In this summer project, I would use DenseNet and AlexNet as the base model, inserted with STN

- Ranker Network:
Ranker Network can also adopted from other pre-defined network.
In this summer project, I would also try the DenseNet and AlexNet.
'''


class LocNet(nn.Module):
    '''
    This LocNet is built based on a spatial transformer network.

    In this project, as the paper said,
    - adopt the AlexNet/DenseNet (base model) and only use the layers in features (take out the classifier layers)
    - add extra conv layer
        - (256, 128, kernel_size=1) alexnet. the output size of the feature layers of alexnet is of 256 channels 6 * 6
        - (1024, 128, kernel_size=1) densenet. the output size of the features layers of densenet is of 1024 channels 7 * 7
            - add average pooling to make 128 * 6 * 6
    - Add the fc_loc:
        - 4608 -> 128 -> 3 (alexnet, only parameters for the transformation)
        - 4608 -> 128 -> 3 (densenet)

    Note:
        the size of alexnet input is 3*224*224.
    '''

    def __init__(self, model_name, device):
        super(LocNet, self).__init__()

        # Spatial transformer localization-network
        self.model_name = model_name
        self.device = device
        if self.model_name == 'alexnet':
            self.features_conv = models.alexnet(pretrained=True).features
            self.extra_conv = nn.Conv2d(256, 128, kernel_size=1)
        elif self.model_name == 'densenet':
            self.features_conv = models.densenet121(pretrained=True).features
            self.extra_conv = nn.Conv2d(1024, 128, kernel_size=1)
        else:
            raise Exception(
                "The model {} is not defined in the loc net.".format(self.model_name))

        self.fc_loc = nn.Sequential(
            nn.Linear(4608, 128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        )
    # Spatial transformer network forward function

    def stn(self, x):
        '''
        Get the scale and transformation parameter, make affine transformation, sampling.
        '''
        xs = self.features_conv(x)
        # print(xs.size())  # [64, 256, 6, 6]
        xs = self.extra_conv(xs)
        # print(xs.size())  # [64, 128, 6, 6]
        if self.model_name == 'densenet':
            m = nn.AdaptiveAvgPool2d((6, 6))
            xs = m(xs)
        xs = xs.view(-1, 4608)
        # print(xs.size())  # [64, 4608]
        theta = self.fc_loc(xs)
        # print the size of theta and the value of theta
        # print(theta.size(), theta)[64, 3]

        # get the theta weights which is the output of localization net
        # reshape the N*3 theta into N*2*3 theta tensor
        theta_list = []
        for i in theta:
            temp = [[i[0], 0, i[1]], [0, i[0], i[2]]]
            theta_list.append(temp)
        theta = torch.tensor(theta_list).to(self.device)
        # print(theta.size())  # [64, 2, 3]

        # used to do the affine transformation
        # the size if x should N*C*H*W
        grid = F.affine_grid(theta, x.size())
        # print(grid.size(), grid)
        x = F.grid_sample(x, grid)  # sampler. Could be biinearsampler
        # print(x.size(), x)
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

    def __init__(self, model_name):
        super(RankNet, self).__init__()
        self.model_name = model_name
        if self.model_name == 'alexnet':
            self.rank = models.alexnet(pretrained=True)
            self.rank.classifier = self.rank.classifier[:-1]
        elif self.model_name == 'densenet':
            self.rank = models.densenet121(pretrained=True).features
        else:
            raise Exception(
                "The model {} is not defined in the rank net.".format(self.model_name))

    def forward(self, x):
        xs = self.rank(x)
        return xs


class SiameseBranch(nn.Module):
    '''
    The branch without the global image in the ranknet
    '''

    def __init__(self, model_name, device):
        super(SiameseBranch, self).__init__()
        self.model_name = model_name
        self.device = device
        if model_name == 'densenet':
            self.extra_util_layer = nn.Conv2d(1024, 128, kernel_size=1)
            self.average_pooling = nn.AdaptiveAvgPool2d((6, 6))

        self.locnet = LocNet(self.model_name, self.device)
        self.ranknet = RankNet(self.model_name)
        self.final_fc = nn.Linear(4096, 1)

    def forward(self, x):
        xs = self.locnet(x)
        xs = self.ranknet(xs)
        if self.model_name == 'densenet':
            # add the average pooling to make the features output to 4096
            xs = self.extra_util_layer(xs)
            # add the view function to make the (64,128,6,6) to (64,4096)
            xs = self.average_pooling(xs).view(xs.size(0), 128 * 6 * 6)
        xs = self.final_fc(xs)
        return xs


class SiameseCombinedBranch(nn.Module):
    '''
    The branch combined the global image
    '''

    def __init__(self, model_name, device):
        super(SiameseCombinedBranch, self).__init__()
        self.model_name = model_name
        self.device = device
        if model_name == 'densenet':
            self.extra_util_layer = nn.Conv2d(1024, 128, kernel_size=1)
            self.average_pooling = nn.AdaptiveAvgPool2d((6, 6))

        self.locnet = LocNet(self.device)
        self.ranknet = RankNet()
        self.final_fc = nn.Linear(8192, 1)

    def forward(self, x):
        xs = self.locnet(x)  # output: 4096
        x1 = self.ranknet(xs)
        if self.model_name == 'densenet':
            # add the average pooling to make the features output to 4096
            x1 = self.extra_util_layer(x1)
            # add the view function to make the (64,128,6,6) to (64,4096)
            x1 = self.average_pooling(x1).view(x1.size(0), 128 * 6 * 6)
        x2 = self.ranknet(x)
        if self.model_name == 'densenet':
            # add the average pooling to make the features output to 4096
            x2 = self.extra_util_layer(x2)
            # add the view function to make the (64,128,6,6) to (64,4096)
            x2 = self.average_pooling(x2).view(x2.size(0), 128 * 6 * 6)

        xs = torch.cat((x1, x2), dim=1)  # it works, and can do the derivation
        xs = self.final_fc(xs)
        return xs


class SiameseNet(nn.Module):
    '''
    Combine the STN and RN into a Siamese network for this experiment.

    args:
        - model_type: 1 or 2. The 1 type represents the network just using local part of image. Type 2 combines the global image

    '''

    def __init__(self, model_type=1, model_name='alexnet', device='cpu', freeze_locnet=False, freeze_ranknet=False):
        '''
        If combined model, then concanate the global image and local part.
        '''
        super(SiameseNet, self).__init__()
        self.model_name = model_name
        self.device = device
        self.freeze_locnet_flag = freeze_locnet
        self.freeze_ranknet_flag = freeze_ranknet

        if model_type == 2:
            self.branch = SiameseCombinedBranch(self.model_name, self.device)
        else:
            self.branch = SiameseBranch(self.model_name, self.device)

        if self.freeze_locnet_flag:
            self.freeze_locnet()
        if self.freeze_ranknet_flag:
            self.freeze_ranknet()

    def forward(self, x):
        # return a tuple (v1,v2)
        x1 = self.branch(x[0])
        x2 = self.branch(x[1])
        return x1, x2

    def freeze_locnet(self):
        if self.model_name == 'alexnet':
            for para in self.branch.locnet.alex_conv.parameters():
                para.requires_grad = False
        elif self.model_name == 'densenet':
            for para in self.branch.locnet.features_conv.parameters():
                para.requires_grad = False
        else:
            raise Exception(
                'The model {} is not supported for freezing localisation net'.format(self.model_name))
        return

    def freeze_ranknet(self):
        if self.model_name == 'alexnet':
            for para in self.branch.ranknet.rank.features.parameters():
                para.requires_grad = False
        elif self.model_name == 'densenet':
            for para in self.branch.ranknet.rank.parameters():
                para.requires_grad = False
        else:
            raise Exception(
                'The model {} is not supported for freezing rank net'.format(self.model_name))
        return
