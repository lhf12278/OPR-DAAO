import torch
import torch.nn as nn
import torchvision
from .bnneck import BNClassifier
from .non_local import _NonLocalBlockND
from torch.nn import Module,Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F


class Res50BNNeckk(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Res50BNNeckk, self).__init__()

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

    def forward(self, x):


        x = self.conv1(x)
        x = self.layer1(x)
        out2_t = self.layer2(x)
        out3_t = self.layer3(out2_t)
        out4_t = self.layer4(out3_t)



        gapfeatures_t = self.gap(out4_t).squeeze(dim=2).squeeze(dim=2)
        gap_bned_features_t, gap_cls_score_t = self.classifier(gapfeatures_t)

        gmpfeatures_t = self.gmp(out4_t).squeeze(dim=2).squeeze(dim=2)
        gmp_bned_features_t, gmp_cls_score_t = self.classifier(gmpfeatures_t)

        sumfeatures_t = gapfeatures_t + gmpfeatures_t
        sum_features_t, sum_cls_score_t = self.classifier(sumfeatures_t)

        out3gap = self.gap(out3_t).squeeze(dim=2).squeeze(dim=2)
        out3gmp = self.gmp(out3_t).squeeze(dim=2).squeeze(dim=2)
        out3sum = out3gap + out3gmp

        out2gap = self.gap(out2_t).squeeze(dim=2).squeeze(dim=2)
        out2gmp = self.gmp(out2_t).squeeze(dim=2).squeeze(dim=2)
        out2sum = out2gap + out2gmp

        if self.training:
            return gapfeatures_t, gap_cls_score_t, gmpfeatures_t, gmp_cls_score_t, sumfeatures_t, sum_cls_score_t, out2_t, out2gap, out3_t, out4_t
        else:
            return sumfeatures_t


