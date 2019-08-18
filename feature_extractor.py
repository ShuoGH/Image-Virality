import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models


'''
Implement a classifier which is used to do the multi class classification problem.

Feature extractor:
    - AlexNet
    - VGG
    - ResNet
    - DenseNet

Classifier:
    - MLP
    - LightGBM
    - Multi class logistic regression

To connect the classification problem and the siamese net, we can say, the classification is used to find the best feature extractor
so that I can embedded into the siamese network and imply the spatial network for our work.
'''


class Identity(nn.Module):
    '''
    Create this Identity class for the replacement of certain layer
    '''

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class AlexNetExtractor(nn.Module):
    '''
    If the batch_size == 64, the extracted feature is of size 64 256 6 6.
    '''

    def __init__(self):
        super(AlexNetExtractor, self).__init__()
        self.alex_conv = models.alexnet(pretrained=True).features

    def forward(self, x):
        out = self.alex_conv(x)
        return out


class VGGExtractor(nn.Module):
    '''
    Batch size 64, input 3 224 224, the output is 64 512 7 7. if use vgg16
    '''

    def __init__(self):
        super(VGGExtractor, self).__init__()
        self.vgg_conv = models.vgg16(pretrained=True).features

    def forward(self, x):
        out = self.vgg_conv(x)
        return out


class ResNetExtractor(nn.Module):
    '''
    Resnet18:
        input 64 3 224 224 output 64 512

    Resnet34:
        output: 64 512
    '''

    def __init__(self):
        super(ResNetExtractor, self).__init__()
        self.resnet_conv = models.resnet18(pretrained=True)
        self.resnet_conv.fc = Identity()

    def forward(self, x):
        out = self.resnet_conv
        return out


class DenseNetExtractor(nn.Module):
    '''
    DenseNet121: output is 64 1024
    DenseNet161: output is 64 2208

    '''

    def __init__(self):
        super(DenseNetExtractor, self).__init__()
        self.densenet_conv = models.densenet(pretrained=True)

    def forward(self, x):
        out = self.densenet_conv(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_name='AlexNet'):
        super(FeatureExtractor, self).__init__()
        if model_name == 'AlexNet':
            self.extractor = AlexNetExtractor()
        elif model_name == 'VGG':
            self.extractor = VGGExtractor()
        elif model_name == 'ResNet':
            self.extractor = ResNetExtractor()
        elif model_name == 'DenseNet':
            self.extractor = DenseNetExtractor()
        else:
            raise Exception(
                "The {} extractor is not defined.".format(model_name))

    def forward(self, x):
        return self.extractor(x)
