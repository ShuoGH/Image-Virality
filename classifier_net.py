import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models


class AlexNet_model(nn.Module):
    def __init__(self, num_classes=5, pretrained_flag=True):
        super(AlexNet_model, self).__init__()
        self.num_classes = num_classes
        self.model = models.alexnet(pretrained=pretrained_flag)
        self.model.classifier[-1] = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def freeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = False


class VGG_model(nn.Module):
    def __init__(self, num_classes=5, pretrained_flag=True):
        super(VGG_model, self).__init__()
        self.num_classes = num_classes
        self.model = models.vgg16(pretrained=pretrained_flag)
        self.model.classifier[-1] = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def freeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = False


class ResNet_model(nn.Module):
    def __init__(self, num_classes=5, pretrained_flag=True):
        super(ResNet_model, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet101(pretrained=pretrained_flag)
        self.model.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def freeze(self):
        '''
        Only leave the fc layer open for grad.
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True


class DenseNet_model(nn.Module):
    def __init__(self, num_classes=5, pretrained_flag=True):
        super(DenseNet_model, self).__init__()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=pretrained_flag)
        self.model.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def freeze(self):
        '''
        Only leave the classifier layer open for grad.
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True


def cnn_classifier(model_name, num_classes=5):
    num_classes = num_classes
    pretrained_flag = True
    if model_name == 'alexnet':
        return AlexNet_model(num_classes, pretrained_flag)

    elif model_name == 'vgg':
        return VGG_model(num_classes, pretrained_flag)

    elif model_name == 'resnet':
        return ResNet_model(num_classes, pretrained_flag)

    elif model_name == 'densenet':
        return DenseNet_model(num_classes, pretrained_flag)

    else:
        raise Exception(
            "The {} model is not defined here.".format(model_name))
