'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.mnn as mnn


class MNNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MNNBasicBlock, self).__init__()
        self.morph1 = mnn.MNN(
            in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.morph2 = mnn.MNN(planes, planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                mnn.MNN(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.morph1(x)))
        out = self.bn2(self.morph2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MNNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(MNNBottleneck, self).__init__()
        self.morph1 = mnn.MNN(in_planes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.morph2 = mnn.MNN(planes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.morph3 = mnn.MNN(planes, self.expansion *
                               planes, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                mnn.MNN(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.morph1(x)))
        out = F.relu(self.bn2(self.morph2(out)))
        out = self.bn3(self.morph3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MNNResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(MNNResNet, self).__init__()
        self.in_planes = 32

        self.morph1 = mnn.MNN(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(4096*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.morph1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MNNResNet18():
    return MNNResNet(MNNBasicBlock, [2, 2, 2, 2], num_classes=6)


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = MNNResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
