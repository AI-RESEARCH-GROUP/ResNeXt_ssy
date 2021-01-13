import torch.nn as nn
from torch.nn import Conv2d, AvgPool2d, MaxPool2d, BatchNorm2d, ReLU
import torch
from torchvision.models import resnet

def createModel(opt):
    depth = opt.depth
    shortcutType = opt.shortcutType or 'B'
    iChannels = opt.iChannels

    def shortcut(nInputPlane, nOutputPlane, stride):
        useConv = shortcutType == 'C' or (shortcutType == 'B' and nInputPlane != nOutputPlane)
        if useConv:
            return nn.Sequential(Conv2d(nInputPlane, nOutputPlane, 1, 1, stride, stride),
                                 BatchNorm2d(nOutputPlane))
        elif nInputPlane != nOutputPlane:
            return nn.Sequential(AvgPool2d(1, 1, stride, stride))
        else:
            return nn.Identity()


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = Conv2d(inplanes, width, 1, 1, 0)
        self.bn1 = norm_layer(width)
        self.conv2 = Conv2d(width, width, 3, stride, 1)
        self.bn2 = norm_layer(width)
        self.conv3 = Conv2d(width, planes * self.expansion, 1, 1, 0)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

