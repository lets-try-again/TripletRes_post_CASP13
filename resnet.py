# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:40:25 2017

@author: lee
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.droprate = 0.2
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bn2(out)


        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        #########################input for cov on GPU0#######################
        self.conv1 = nn.Conv2d(441, 64, kernel_size=1,
                               bias=False)#.cuda()
        self.bn1 = nn.InstanceNorm2d(64)#.cuda()
        self.relu = nn.ReLU(inplace=True)#.cuda()
        self.layer1 = self._make_layer(block, 64, layers[0])#.cuda()
        self.layer2 = self._make_layer(block, 64, layers[1])#.cuda()
        #########################input for cov on GPU0 end###################



        #########################input for plm on GPU1#######################
        self.conv2=nn.Conv2d(441, 64, kernel_size=1,
                               bias=False)#.cuda()
        self.bn2 = nn.InstanceNorm2d(64)#.cuda()
        #self.relu = nn.ReLU(inplace=True)
        self.layer3 = self._make_layer(block, 64, layers[0])#.cuda()
        self.layer4 = self._make_layer(block, 64, layers[1])#.cuda()
        #########################input for plm on GPU1 end###################

        #########################input for PRE on GPU2#######################
        self.conv3=nn.Conv2d(441, 64, kernel_size=1,
                               bias=False)#.cuda()
        self.bn3 = nn.InstanceNorm2d(64)#.cuda()
        #self.relu = nn.ReLU(inplace=True)
        self.layer5 = self._make_layer(block, 64, layers[0])#.cuda()
        self.layer6 = self._make_layer(block, 64, layers[1])#.cuda()
        #########################input for PRE on GPU2 end###################

        #########################input for all on GPU3#######################
        self.conv4=nn.Conv2d(192, 64, kernel_size=1,
                               bias=False)#.cuda()
        self.bn4 = nn.InstanceNorm2d(64)#.cuda()
        #self.relu = nn.ReLU(inplace=True)
        self.layer7 = self._make_layer(block, 64, layers[2])#.cuda()
        self.layer8 = self._make_layer(block, 64, layers[3])#.cuda()
        #########################input for all on GPU3 end###################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.lastlayer=nn.Conv2d(self.inplanes,12,3,padding=1)#.cuda()
        self.sig=nn.Sigmoid()
        self.soft=nn.Softmax(1)
    def _make_layer(self, block, planes, blocks):

        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1,x2,x3):

        #x1 in cuda(0), x2 in cuda(1), x3 in cuda(2), x in cuda(3)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)

        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.layer5(x3)
        x3 = self.layer6(x3)

        x=torch.cat((x1, x2,x3), 1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.lastlayer(x)
        x = self.soft(x)
        x=0.5*(x+torch.transpose(x, -1, -2))

        return x,torch.sum(x[:,:4,:,:],1,True)

def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 3, 3, 2])
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet46(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 6, 10, 3])
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet86():
    model=ResNet(BasicBlock,[12,12,12,12])
    return model
def resnet86_triple():
    model=ResNet(BasicBlock,[12,12,12,12])
    return model