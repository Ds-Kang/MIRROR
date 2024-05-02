#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   AugmentCE2P.py
@Time    :   8/4/19 3:35 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
# Note here we adopt the nn.BatchNorm2d implementation from https://github.com/mapillary/inplace_abn
# By default, the nn.BatchNorm2d module contains a BatchNorm Layer and a LeakyReLu layer

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


BatchNorm2d = nn.BatchNorm2d


class LeakyReluLayer(nn.Module):
    def __init__(self,inplace=False, negative_slope=0.1):
        super(LeakyReluLayer, self).__init__()
        self.negative_slope=negative_slope


    def forward(self, x):
        neg_x=self.negative_slope*x

        return torch.max(neg_x, x)


class InPlaceABNSync(BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(InPlaceABNSync, self).__init__(*args, **kwargs)
        self.act = LeakyReluLayer()

    def forward(self, input):
        output = super(InPlaceABNSync, self).forward(input)
        output = self.act(output)
        return output

affine_par = True

pretrained_settings = {
    'resnet101': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
    'mobilenet': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
}


CORRESP_NAME = {
    # layer1
    "features.0.0.weight": "layer1.0.weight",
    "features.0.1.weight": "layer1.1.weight",
    "features.0.1.bias": "layer1.1.bias",
    "features.0.1.running_mean": "layer1.1.running_mean",
    "features.0.1.running_var": "layer1.1.running_var",

    "features.1.conv.0.weight": "layer1.2.conv.0.weight",
    "features.1.conv.1.weight": "layer1.2.conv.1.weight",
    "features.1.conv.1.bias": "layer1.2.conv.1.bias",
    "features.1.conv.1.running_mean": "layer1.2.conv.1.running_mean",
    "features.1.conv.1.running_var": "layer1.2.conv.1.running_var",
    "features.1.conv.3.weight": "layer1.2.conv.3.weight",
    "features.1.conv.4.weight": "layer1.2.conv.4.weight",
    "features.1.conv.4.bias": "layer1.2.conv.4.bias",
    "features.1.conv.4.running_mean": "layer1.2.conv.4.running_mean",
    "features.1.conv.4.running_var": "layer1.2.conv.4.running_var",
    # layer2
    "features.2.conv.0.weight": "layer2.0.conv.0.weight",
    "features.2.conv.1.weight": "layer2.0.conv.1.weight",
    "features.2.conv.1.bias": "layer2.0.conv.1.bias",
    "features.2.conv.1.running_mean": "layer2.0.conv.1.running_mean",
    "features.2.conv.1.running_var": "layer2.0.conv.1.running_var",
    "features.2.conv.3.weight": "layer2.0.conv.3.weight",
    "features.2.conv.4.weight": "layer2.0.conv.4.weight",
    "features.2.conv.4.bias": "layer2.0.conv.4.bias",
    "features.2.conv.4.running_mean": "layer2.0.conv.4.running_mean",
    "features.2.conv.4.running_var": "layer2.0.conv.4.running_var",
    "features.2.conv.6.weight": "layer2.0.conv.6.weight",
    "features.2.conv.7.weight": "layer2.0.conv.7.weight",
    "features.2.conv.7.bias": "layer2.0.conv.7.bias",
    "features.2.conv.7.running_mean": "layer2.0.conv.7.running_mean",
    "features.2.conv.7.running_var": "layer2.0.conv.7.running_var",

    "features.3.conv.0.weight": "layer2.1.conv.0.weight",
    "features.3.conv.1.weight": "layer2.1.conv.1.weight",
    "features.3.conv.1.bias": "layer2.1.conv.1.bias",
    "features.3.conv.1.running_mean": "layer2.1.conv.1.running_mean",
    "features.3.conv.1.running_var": "layer2.1.conv.1.running_var",
    "features.3.conv.3.weight": "layer2.1.conv.3.weight",
    "features.3.conv.4.weight": "layer2.1.conv.4.weight",
    "features.3.conv.4.bias": "layer2.1.conv.4.bias",
    "features.3.conv.4.running_mean": "layer2.1.conv.4.running_mean",
    "features.3.conv.4.running_var": "layer2.1.conv.4.running_var",
    "features.3.conv.6.weight": "layer2.1.conv.6.weight",
    "features.3.conv.7.weight": "layer2.1.conv.7.weight",
    "features.3.conv.7.bias": "layer2.1.conv.7.bias",
    "features.3.conv.7.running_mean": "layer2.1.conv.7.running_mean",
    "features.3.conv.7.running_var": "layer2.1.conv.7.running_var",
    # layer3
    "features.4.conv.0.weight": "layer3.0.conv.0.weight",
    "features.4.conv.1.weight": "layer3.0.conv.1.weight",
    "features.4.conv.1.bias": "layer3.0.conv.1.bias",
    "features.4.conv.1.running_mean": "layer3.0.conv.1.running_mean",
    "features.4.conv.1.running_var": "layer3.0.conv.1.running_var",
    "features.4.conv.3.weight": "layer3.0.conv.3.weight",
    "features.4.conv.4.weight": "layer3.0.conv.4.weight",
    "features.4.conv.4.bias": "layer3.0.conv.4.bias",
    "features.4.conv.4.running_mean": "layer3.0.conv.4.running_mean",
    "features.4.conv.4.running_var": "layer3.0.conv.4.running_var",
    "features.4.conv.6.weight": "layer3.0.conv.6.weight",
    "features.4.conv.7.weight": "layer3.0.conv.7.weight",
    "features.4.conv.7.bias": "layer3.0.conv.7.bias",
    "features.4.conv.7.running_mean": "layer3.0.conv.7.running_mean",
    "features.4.conv.7.running_var": "layer3.0.conv.7.running_var",

    "features.5.conv.0.weight": "layer3.1.conv.0.weight",
    "features.5.conv.1.weight": "layer3.1.conv.1.weight",
    "features.5.conv.1.bias": "layer3.1.conv.1.bias",
    "features.5.conv.1.running_mean": "layer3.1.conv.1.running_mean",
    "features.5.conv.1.running_var": "layer3.1.conv.1.running_var",
    "features.5.conv.3.weight": "layer3.1.conv.3.weight",
    "features.5.conv.4.weight": "layer3.1.conv.4.weight",
    "features.5.conv.4.bias": "layer3.1.conv.4.bias",
    "features.5.conv.4.running_mean": "layer3.1.conv.4.running_mean",
    "features.5.conv.4.running_var": "layer3.1.conv.4.running_var",
    "features.5.conv.6.weight": "layer3.1.conv.6.weight",
    "features.5.conv.7.weight": "layer3.1.conv.7.weight",
    "features.5.conv.7.bias": "layer3.1.conv.7.bias",
    "features.5.conv.7.running_mean": "layer3.1.conv.7.running_mean",
    "features.5.conv.7.running_var": "layer3.1.conv.7.running_var",

    "features.6.conv.0.weight": "layer3.2.conv.0.weight",
    "features.6.conv.1.weight": "layer3.2.conv.1.weight",
    "features.6.conv.1.bias": "layer3.2.conv.1.bias",
    "features.6.conv.1.running_mean": "layer3.2.conv.1.running_mean",
    "features.6.conv.1.running_var": "layer3.2.conv.1.running_var",
    "features.6.conv.3.weight": "layer3.2.conv.3.weight",
    "features.6.conv.4.weight": "layer3.2.conv.4.weight",
    "features.6.conv.4.bias": "layer3.2.conv.4.bias",
    "features.6.conv.4.running_mean": "layer3.2.conv.4.running_mean",
    "features.6.conv.4.running_var": "layer3.2.conv.4.running_var",
    "features.6.conv.6.weight": "layer3.2.conv.6.weight",
    "features.6.conv.7.weight": "layer3.2.conv.7.weight",
    "features.6.conv.7.bias": "layer3.2.conv.7.bias",
    "features.6.conv.7.running_mean": "layer3.2.conv.7.running_mean",
    "features.6.conv.7.running_var": "layer3.2.conv.7.running_var",

    # layer4
    "features.7.conv.0.weight": "layer4.0.conv.0.weight",
    "features.7.conv.1.weight": "layer4.0.conv.1.weight",
    "features.7.conv.1.bias": "layer4.0.conv.1.bias",
    "features.7.conv.1.running_mean": "layer4.0.conv.1.running_mean",
    "features.7.conv.1.running_var": "layer4.0.conv.1.running_var",
    "features.7.conv.3.weight": "layer4.0.conv.3.weight",
    "features.7.conv.4.weight": "layer4.0.conv.4.weight",
    "features.7.conv.4.bias": "layer4.0.conv.4.bias",
    "features.7.conv.4.running_mean": "layer4.0.conv.4.running_mean",
    "features.7.conv.4.running_var": "layer4.0.conv.4.running_var",
    "features.7.conv.6.weight": "layer4.0.conv.6.weight",
    "features.7.conv.7.weight": "layer4.0.conv.7.weight",
    "features.7.conv.7.bias": "layer4.0.conv.7.bias",
    "features.7.conv.7.running_mean": "layer4.0.conv.7.running_mean",
    "features.7.conv.7.running_var": "layer4.0.conv.7.running_var",

    "features.8.conv.0.weight": "layer4.1.conv.0.weight",
    "features.8.conv.1.weight": "layer4.1.conv.1.weight",
    "features.8.conv.1.bias": "layer4.1.conv.1.bias",
    "features.8.conv.1.running_mean": "layer4.1.conv.1.running_mean",
    "features.8.conv.1.running_var": "layer4.1.conv.1.running_var",
    "features.8.conv.3.weight": "layer4.1.conv.3.weight",
    "features.8.conv.4.weight": "layer4.1.conv.4.weight",
    "features.8.conv.4.bias": "layer4.1.conv.4.bias",
    "features.8.conv.4.running_mean": "layer4.1.conv.4.running_mean",
    "features.8.conv.4.running_var": "layer4.1.conv.4.running_var",
    "features.8.conv.6.weight": "layer4.1.conv.6.weight",
    "features.8.conv.7.weight": "layer4.1.conv.7.weight",
    "features.8.conv.7.bias": "layer4.1.conv.7.bias",
    "features.8.conv.7.running_mean": "layer4.1.conv.7.running_mean",
    "features.8.conv.7.running_var": "layer4.1.conv.7.running_var",

    "features.9.conv.0.weight": "layer4.2.conv.0.weight",
    "features.9.conv.1.weight": "layer4.2.conv.1.weight",
    "features.9.conv.1.bias": "layer4.2.conv.1.bias",
    "features.9.conv.1.running_mean": "layer4.2.conv.1.running_mean",
    "features.9.conv.1.running_var": "layer4.2.conv.1.running_var",
    "features.9.conv.3.weight": "layer4.2.conv.3.weight",
    "features.9.conv.4.weight": "layer4.2.conv.4.weight",
    "features.9.conv.4.bias": "layer4.2.conv.4.bias",
    "features.9.conv.4.running_mean": "layer4.2.conv.4.running_mean",
    "features.9.conv.4.running_var": "layer4.2.conv.4.running_var",
    "features.9.conv.6.weight": "layer4.2.conv.6.weight",
    "features.9.conv.7.weight": "layer4.2.conv.7.weight",
    "features.9.conv.7.bias": "layer4.2.conv.7.bias",
    "features.9.conv.7.running_mean": "layer4.2.conv.7.running_mean",
    "features.9.conv.7.running_var": "layer4.2.conv.7.running_var",

    "features.10.conv.0.weight": "layer4.3.conv.0.weight",
    "features.10.conv.1.weight": "layer4.3.conv.1.weight",
    "features.10.conv.1.bias": "layer4.3.conv.1.bias",
    "features.10.conv.1.running_mean": "layer4.3.conv.1.running_mean",
    "features.10.conv.1.running_var": "layer4.3.conv.1.running_var",
    "features.10.conv.3.weight": "layer4.3.conv.3.weight",
    "features.10.conv.4.weight": "layer4.3.conv.4.weight",
    "features.10.conv.4.bias": "layer4.3.conv.4.bias",
    "features.10.conv.4.running_mean": "layer4.3.conv.4.running_mean",
    "features.10.conv.4.running_var": "layer4.3.conv.4.running_var",
    "features.10.conv.6.weight": "layer4.3.conv.6.weight",
    "features.10.conv.7.weight": "layer4.3.conv.7.weight",
    "features.10.conv.7.bias": "layer4.3.conv.7.bias",
    "features.10.conv.7.running_mean": "layer4.3.conv.7.running_mean",
    "features.10.conv.7.running_var": "layer4.3.conv.7.running_var",

    "features.11.conv.0.weight": "layer4.4.conv.0.weight",
    "features.11.conv.1.weight": "layer4.4.conv.1.weight",
    "features.11.conv.1.bias": "layer4.4.conv.1.bias",
    "features.11.conv.1.running_mean": "layer4.4.conv.1.running_mean",
    "features.11.conv.1.running_var": "layer4.4.conv.1.running_var",
    "features.11.conv.3.weight": "layer4.4.conv.3.weight",
    "features.11.conv.4.weight": "layer4.4.conv.4.weight",
    "features.11.conv.4.bias": "layer4.4.conv.4.bias",
    "features.11.conv.4.running_mean": "layer4.4.conv.4.running_mean",
    "features.11.conv.4.running_var": "layer4.4.conv.4.running_var",
    "features.11.conv.6.weight": "layer4.4.conv.6.weight",
    "features.11.conv.7.weight": "layer4.4.conv.7.weight",
    "features.11.conv.7.bias": "layer4.4.conv.7.bias",
    "features.11.conv.7.running_mean": "layer4.4.conv.7.running_mean",
    "features.11.conv.7.running_var": "layer4.4.conv.7.running_var",

    "features.12.conv.0.weight": "layer4.5.conv.0.weight",
    "features.12.conv.1.weight": "layer4.5.conv.1.weight",
    "features.12.conv.1.bias": "layer4.5.conv.1.bias",
    "features.12.conv.1.running_mean": "layer4.5.conv.1.running_mean",
    "features.12.conv.1.running_var": "layer4.5.conv.1.running_var",
    "features.12.conv.3.weight": "layer4.5.conv.3.weight",
    "features.12.conv.4.weight": "layer4.5.conv.4.weight",
    "features.12.conv.4.bias": "layer4.5.conv.4.bias",
    "features.12.conv.4.running_mean": "layer4.5.conv.4.running_mean",
    "features.12.conv.4.running_var": "layer4.5.conv.4.running_var",
    "features.12.conv.6.weight": "layer4.5.conv.6.weight",
    "features.12.conv.7.weight": "layer4.5.conv.7.weight",
    "features.12.conv.7.bias": "layer4.5.conv.7.bias",
    "features.12.conv.7.running_mean": "layer4.5.conv.7.running_mean",
    "features.12.conv.7.running_var": "layer4.5.conv.7.running_var",

    "features.13.conv.0.weight": "layer4.6.conv.0.weight",
    "features.13.conv.1.weight": "layer4.6.conv.1.weight",
    "features.13.conv.1.bias": "layer4.6.conv.1.bias",
    "features.13.conv.1.running_mean": "layer4.6.conv.1.running_mean",
    "features.13.conv.1.running_var": "layer4.6.conv.1.running_var",
    "features.13.conv.3.weight": "layer4.6.conv.3.weight",
    "features.13.conv.4.weight": "layer4.6.conv.4.weight",
    "features.13.conv.4.bias": "layer4.6.conv.4.bias",
    "features.13.conv.4.running_mean": "layer4.6.conv.4.running_mean",
    "features.13.conv.4.running_var": "layer4.6.conv.4.running_var",
    "features.13.conv.6.weight": "layer4.6.conv.6.weight",
    "features.13.conv.7.weight": "layer4.6.conv.7.weight",
    "features.13.conv.7.bias": "layer4.6.conv.7.bias",
    "features.13.conv.7.running_mean": "layer4.6.conv.7.running_mean",
    "features.13.conv.7.running_var": "layer4.6.conv.7.running_var",

    # layer5
    "features.14.conv.0.weight": "layer5.0.conv.0.weight",
    "features.14.conv.1.weight": "layer5.0.conv.1.weight",
    "features.14.conv.1.bias": "layer5.0.conv.1.bias",
    "features.14.conv.1.running_mean": "layer5.0.conv.1.running_mean",
    "features.14.conv.1.running_var": "layer5.0.conv.1.running_var",
    "features.14.conv.3.weight": "layer5.0.conv.3.weight",
    "features.14.conv.4.weight": "layer5.0.conv.4.weight",
    "features.14.conv.4.bias": "layer5.0.conv.4.bias",
    "features.14.conv.4.running_mean": "layer5.0.conv.4.running_mean",
    "features.14.conv.4.running_var": "layer5.0.conv.4.running_var",
    "features.14.conv.6.weight": "layer5.0.conv.6.weight",
    "features.14.conv.7.weight": "layer5.0.conv.7.weight",
    "features.14.conv.7.bias": "layer5.0.conv.7.bias",
    "features.14.conv.7.running_mean": "layer5.0.conv.7.running_mean",
    "features.14.conv.7.running_var": "layer5.0.conv.7.running_var",

    "features.15.conv.0.weight": "layer5.1.conv.0.weight",
    "features.15.conv.1.weight": "layer5.1.conv.1.weight",
    "features.15.conv.1.bias": "layer5.1.conv.1.bias",
    "features.15.conv.1.running_mean": "layer5.1.conv.1.running_mean",
    "features.15.conv.1.running_var": "layer5.1.conv.1.running_var",
    "features.15.conv.3.weight": "layer5.1.conv.3.weight",
    "features.15.conv.4.weight": "layer5.1.conv.4.weight",
    "features.15.conv.4.bias": "layer5.1.conv.4.bias",
    "features.15.conv.4.running_mean": "layer5.1.conv.4.running_mean",
    "features.15.conv.4.running_var": "layer5.1.conv.4.running_var",
    "features.15.conv.6.weight": "layer5.1.conv.6.weight",
    "features.15.conv.7.weight": "layer5.1.conv.7.weight",
    "features.15.conv.7.bias": "layer5.1.conv.7.bias",
    "features.15.conv.7.running_mean": "layer5.1.conv.7.running_mean",
    "features.15.conv.7.running_var": "layer5.1.conv.7.running_var",

    "features.16.conv.0.weight": "layer5.2.conv.0.weight",
    "features.16.conv.1.weight": "layer5.2.conv.1.weight",
    "features.16.conv.1.bias": "layer5.2.conv.1.bias",
    "features.16.conv.1.running_mean": "layer5.2.conv.1.running_mean",
    "features.16.conv.1.running_var": "layer5.2.conv.1.running_var",
    "features.16.conv.3.weight": "layer5.2.conv.3.weight",
    "features.16.conv.4.weight": "layer5.2.conv.4.weight",
    "features.16.conv.4.bias": "layer5.2.conv.4.bias",
    "features.16.conv.4.running_mean": "layer5.2.conv.4.running_mean",
    "features.16.conv.4.running_var": "layer5.2.conv.4.running_var",
    "features.16.conv.6.weight": "layer5.2.conv.6.weight",
    "features.16.conv.7.weight": "layer5.2.conv.7.weight",
    "features.16.conv.7.bias": "layer5.2.conv.7.bias",
    "features.16.conv.7.running_mean": "layer5.2.conv.7.running_mean",
    "features.16.conv.7.running_var": "layer5.2.conv.7.running_var",
    
    "features.17.conv.0.weight": "layer5.3.conv.0.weight",
    "features.17.conv.1.weight": "layer5.3.conv.1.weight",
    "features.17.conv.1.bias": "layer5.3.conv.1.bias",
    "features.17.conv.1.running_mean": "layer5.3.conv.1.running_mean",
    "features.17.conv.1.running_var": "layer5.3.conv.1.running_var",
    "features.17.conv.3.weight": "layer5.3.conv.3.weight",
    "features.17.conv.4.weight": "layer5.3.conv.4.weight",
    "features.17.conv.4.bias": "layer5.3.conv.4.bias",
    "features.17.conv.4.running_mean": "layer5.3.conv.4.running_mean",
    "features.17.conv.4.running_var": "layer5.3.conv.4.running_var",
    "features.17.conv.6.weight": "layer5.3.conv.6.weight",
    "features.17.conv.7.weight": "layer5.3.conv.7.weight",
    "features.17.conv.7.bias": "layer5.3.conv.7.bias",
    "features.17.conv.7.running_mean": "layer5.3.conv.7.running_mean",
    "features.17.conv.7.running_var": "layer5.3.conv.7.running_var",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=((16, 16), (8, 8), (5, 6), (2, 6))):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AvgPool2d(stride=size[0],kernel_size=size[1])
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle



class Edge_Module(nn.Module):
    """
    Edge Learning Branch
    """

    def __init__(self, in_fea_1=256, in_fea_2=512, in_fea_3=1024, mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea_1, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea_2, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea_3, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class Decoder_Module(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes, in_channels_1=512, in_channels_2=256):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return fusion_result


def initialize_pretrained_model(model, settings, pretrained='./models/resnet101-imagenet.pth'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)


def resnet101(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def initialize_mobilenet(model, settings, pretrained='./pretrain_model/mobilenet_v2.pth.tar'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    
    if pretrained:
        corresp_name = CORRESP_NAME
        new_params = model.state_dict().copy()
        saved_state_dict = torch.load(pretrained)
        for name in saved_state_dict:
            if name not in corresp_name:
                continue
            if corresp_name[name] not in new_params.keys():
                continue
            if name == "features.0.0.weight":
                model_weight = new_params[corresp_name[name]]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = saved_state_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                new_params[corresp_name[name]] = model_weight
            else:
                new_params[corresp_name[name]] = saved_state_dict[name]
        
        model.load_state_dict(new_params)



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=7, input_size=512, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280        
        interverted_residual_setting_layer1 = [
            # t, c, n, s
            [1, 16, 1, 1]
        ]
        interverted_residual_setting_layer2 = [
            # t, c, n, s
            [6, 24, 2, 2]
        ]
        interverted_residual_setting_layer3 = [
            # t, c, n, s
            [6, 32, 3, 2]
        ]
        interverted_residual_setting_layer4 = [
            # t, c, n, s
            [6, 64, 4, 2],
            [6, 96, 3, 1]
        ]
        interverted_residual_setting_layer5 = [
            # t, c, n, s
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.layer1 = [conv_bn(3, input_channel, 2)]
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting_layer1:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layer1.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layer1.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in interverted_residual_setting_layer2:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layer2.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layer2.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in interverted_residual_setting_layer3:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layer3.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layer3.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in interverted_residual_setting_layer4:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layer4.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layer4.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in interverted_residual_setting_layer5:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layer5.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layer5.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.layer5.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)
        self.layer5 = nn.Sequential(*self.layer5)


        self.context_encoding = PSPModule(1280, 320)

        self.edge = Edge_Module(24,32,96)
        self.decoder = Decoder_Module(num_classes, 320,24)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

        # self._initialize_weights()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)     

        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return fusion_result


def mobilenetv2(num_classes=7, pretrained='./pretrain_model/mobilenet_v2.pth.tar'):
    # model = MobileNetV2(num_classes)
    model = MobileNetV2(7)
    
    settings = pretrained_settings['mobilenet']['imagenet']
    
    initialize_mobilenet(model, settings, pretrained)
    # model.load_state_dict(torch.load(pretrained))
    # model.load_state_dict(torch.load(pretrained), strict=False)
    return model
