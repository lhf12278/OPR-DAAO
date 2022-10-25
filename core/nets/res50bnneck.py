import torch
import torch.nn as nn
import torchvision
from .bnneck import BNClassifier
from .non_local import _NonLocalBlockND
from torch.nn import Module,Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F


class Res50BNNeck(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Res50BNNeck, self).__init__()

        self.class_num = class_num
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        # classifier
        self.classifier = BNClassifier(2048, self.class_num)


    def forward(self, x,_NonLocalBlockND):#need
    # def forward(self, x):


        x = self.conv1(x)
        x = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        z,W_y= _NonLocalBlockND(out4)


        mixgapfeatures = self.gap(z).squeeze(dim=2).squeeze(dim=2)
        mixgap_bned_features, mixgap_cls_score = self.classifier(mixgapfeatures)

        mixgmpfeatures = self.gmp(z).squeeze(dim=2).squeeze(dim=2)
        mixgmp_bned_features, mixgmp_cls_score = self.classifier(mixgmpfeatures)

        mixsumfeatures = mixgapfeatures + mixgmpfeatures
        mixsum_features, mixsum_cls_score = self.classifier(mixsumfeatures)

        out3gap = self.gap(out3).squeeze(dim=2).squeeze(dim=2)
        out3gmp = self.gmp(out3).squeeze(dim=2).squeeze(dim=2)
        out3sum = out3gap + out3gmp

        out2gap = self.gap(out2).squeeze(dim=2).squeeze(dim=2)
        out2gmp = self.gmp(out2).squeeze(dim=2).squeeze(dim=2)
        out2sum = out2gap + out2gmp

        if self.training:
            return mixgapfeatures,mixgmpfeatures, mixsumfeatures,mixgap_cls_score,mixgmp_cls_score,mixsum_cls_score, out2 , out3, out4,out3sum, out2sum
        else:
            return mixsumfeatures



