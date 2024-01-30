import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
from myutils.builder import get_builder

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class ResBasicBlock(nn.Module):
    def __init__(self, builder, in_planes, out_planes, stride, dropRate=0.0):
        super(ResBasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        #self.bn1 = MetaBatchNorm2d(in_planes)
        #self.relu1 = nn.LeakyReLU(0.1)
        #self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)

        self.conv1 = builder.conv3x3(in_planes, out_planes, stride=stride)
        self.bn1 = builder.batchnorm(in_planes)
        self.relu1 = builder.activation()
        self.bn2 = builder.batchnorm(out_planes)
        self.relu2 = builder.activation()
        self.conv2 = builder.conv3x3(out_planes, out_planes, stride=1)

        #self.bn2 = MetaBatchNorm2d(out_planes)
        #self.relu2 = nn.LeakyReLU(0.1)
        #self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1,
        #                       padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        #self.convShortcut = (not self.equalInOut) and MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        #                       padding=0, bias=False) or None
        self.convShortcut = (not self.equalInOut) and builder.conv2d(kernel_size=1, in_channels=in_planes, out_channels=out_planes, stride=stride,
                                                                 padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, builder, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(builder, block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, builder, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(builder, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, builder, depth=28, widen_factor=2, n_classes=10, dropRate=0.0, transform_fn=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = ResBasicBlock
        # 1st conv before any network block

        self.conv1 = builder.conv3x3(3, nChannels[0], stride=1)

        # 1st block
        self.block1 = NetworkBlock(builder, n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(builder, n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(builder, n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier

        self.bn1 = builder.batchnorm(nChannels[3])
        self.relu = builder.activation()
        self.fc = nn.Linear(nChannels[3], n_classes)
        #self.fc = builder.conv1x1_fc(nChannels[3], n_classes)

        self.nChannels = nChannels[3]

        self.transform_fn = transform_fn

    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        return self.fc(out)
    def update_batch_stats(self, flag, builder):
        for m in self.modules():
            if isinstance(m, builder.batchnorm):
                m.update_batch_stats = flag


def wideresnet_cifar10_6(**kwargs):
    print(get_builder())
    return WideResNet(get_builder(), n_classes=6, **kwargs)

def wideresnet_cifar10_10(**kwargs):
    print(get_builder())
    return WideResNet(get_builder(), n_classes=10, **kwargs)

def wideresnet_cifar10_50(**kwargs):
    print(get_builder())
    return WideResNet(get_builder(), n_classes=50, **kwargs)

def wideresnet_TinyImagenet_100(**kwargs):
    print(get_builder())
    return WideResNet(get_builder(), n_classes=100, **kwargs)