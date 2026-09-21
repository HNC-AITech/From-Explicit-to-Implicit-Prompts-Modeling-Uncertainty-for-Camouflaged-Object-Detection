import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from Models.pvtv2_our import pvt_v2_b2
import torchvision
#from models.SwinTransformers import SwinTransformer
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

def conv3x3_bn_relu(channel1, channel2, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(channel2),
            nn.GELU(),
            )

class Conv1(nn.Module):
    def __init__(self, channel1, channel2):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=1, padding=0, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        return x
    def initialize(self):
        weight_init(self)

class Conv3(nn.Module):
    def __init__(self, channel1, channel2):
        super(Conv3, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        return x
    def initialize(self):
        weight_init(self)


class Convbnrelu1(nn.Module):
    def __init__(self, channel1, channel2):
        super(Convbnrelu1, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channel2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
    def initialize(self):
        weight_init(self)

class Convbnrelu3(nn.Module):
    def __init__(self, channel1, channel2):
        super(Convbnrelu3, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(channel2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
    def initialize(self):
        weight_init(self)

class Conv_dil(nn.Module):
    def __init__(self, channel1, channel2, dil):
        super(Conv_dil, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=3, padding=dil, dilation=dil)
        self.bn = nn.BatchNorm2d(channel2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Pre(nn.Module):
    def __init__(self, channel1):
        super(Pre, self).__init__()
        self.conv1 = Conv1(channel1=channel1, channel2=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
    def initialize(self):
        weight_init(self)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    def initialize(self):
        weight_init(self)

class DecoderNet(nn.Module):
    def __init__(self, channel=32):
        super(DecoderNet, self).__init__()

        self.Translayer1 = RFB_modified(64, channel)
        self.Translayer2 = RFB_modified(128, channel)
        self.Translayer3 = RFB_modified(320, channel)
        self.Translayer4 = RFB_modified(512, channel)

    def forward(self, fea, image_shape):
        x1 = self.Translayer1(fea[0])
        x2 = self.Translayer2(fea[1])
        x3 = self.Translayer3(fea[2])
        x4 = self.Translayer4(fea[3])

        '''de4 = x4
        de3 = F.interpolate(de4, size= x3.size()[2:], mode='bilinear')*x3+x3
        de2 = F.interpolate(de3, size= x2.size()[2:], mode='bilinear')*x2+x2
        de1 = F.interpolate(de2, size= x1.size()[2:], mode='bilinear')*x1+x1'''

        x1 = F.interpolate(x1, size=image_shape, mode='bilinear')
        x2 = F.interpolate(x2, size=image_shape, mode='bilinear')
        x3 = F.interpolate(x3, size=image_shape, mode='bilinear')
        x4 = F.interpolate(x4, size=image_shape, mode='bilinear')

        fea_mul = x1*x2*x3*x4
        fea_add = x1+x2+x3+x4
        fea_fuse = torch.cat([fea_mul, fea_add], 1)
        # x1 = F.interpolate(x1, size=image_shape, mode='bilinear')
        # x2 = F.interpolate(x2, size=image_shape, mode='bilinear')
        # x3 = F.interpolate(x3, size=image_shape, mode='bilinear')
        # x4 = F.interpolate(x4, size=image_shape, mode='bilinear')



        return fea_fuse

    def initialize(self):
        weight_init(self)

class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):  # r: reduction ratio=16
        super(ChannelAttention, self).__init__()

        hidden_channels = channels // ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False),  # 1x1conv代替全连接，根据原文公式没有偏置项
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False)  # 1x1conv代替全连接，根据原文公式没有偏置项
        )
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        return self.sigmoid(
            self.mlp(x_avg) + self.mlp(x_max)
        )

class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上进行avgpool，(B,C,H,W)->(B,1,H,W)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 在通道维度上进行maxpool，(B,C,H,W)->(B,1,H,W)
        return self.sigmoid(
            self.conv(torch.cat([x_avg, x_max], dim=1))
        )  # Ms(F) = σ(f7×7([AvgP ool(F);MaxPool(F)])) = σ(f7×7([Fsavg;Fsmax]))，对应原文公式(3)

class REM1(nn.Module):
    def __init__(self, channel):
        super(REM1, self).__init__()

        self.CA_f = ChannelAttention(channel)
        self.CA_b = ChannelAttention(channel)

        self.SA_f = SpatialAttention()
        self.SA_b = SpatialAttention()

        self.conv_dil = nn.Sequential(Conv_dil(channel,channel, dil=2),
                                      Conv_dil(channel,channel, dil=2))


        self.fuse_f = BasicConv2d(channel*2, channel, 3, padding=1)
        self.fuse_b = BasicConv2d(channel*2, channel, 3, padding=1)
        self.fuse = BasicConv2d(channel*2, channel, 3, padding=1)

        #self.relu = nn.ReLU(inplace=True)
    def forward(self, fea, prompt):
        fea_shape = fea.size()[2:]
        prompt = F.interpolate(prompt, size=fea_shape, mode='nearest')

        w_f = torch.where(prompt == 1.0, 1.0, 0.0)
        w_u = torch.where(prompt == 0.5, 1.0, 0.0)
        w_b = torch.where(prompt == 0.0, 1.0, 0.0)

        fea_en = self.conv_dil(fea)+fea

        w_fu = w_f+w_u
        w_bu = w_b+w_u

        fea_fu = w_fu * fea_en
        fea_bu = w_bu * fea_en

        focus_f = self.SA_f(self.CA_f(fea_fu) * fea_fu)
        focus_b = self.SA_b(self.CA_b(fea_bu) * fea_bu)

        en_f = self.fuse_f(torch.cat([focus_f * fea_fu, (1 - focus_b) * fea_bu], 1))
        en_b = self.fuse_b(torch.cat([focus_b * fea_bu, (1 - focus_f) * fea_fu], 1))

        out = self.fuse(torch.cat([en_f, en_b], 1))

        return out

class REM(nn.Module):
    def __init__(self, channel):
        super(REM, self).__init__()

        self.CA_f = ChannelAttention(channel)
        self.CA_b = ChannelAttention(channel)

        self.SA_f = SpatialAttention()
        self.SA_b = SpatialAttention()

        self.conv_dil = nn.Sequential(Conv_dil(channel, channel, dil=2),
                                      Conv_dil(channel, channel, dil=2))
        self.cp = Convbnrelu1(channel, channel*3)

        self.fuse_f = BasicConv2d(channel*2, channel, 3, padding=1)
        self.fuse_b = BasicConv2d(channel*2, channel, 3, padding=1)
        self.fuse = BasicConv2d(channel*2, channel, 3, padding=1)

        #self.relu = nn.ReLU(inplace=True)
    def forward(self, fea, prompt):
        fea_shape = fea.size()[2:]
        prompt = F.interpolate(prompt, size=fea_shape, mode='nearest')

        w_f = torch.where(prompt == 1.0, 1.0, 0.0)
        w_u = torch.where(prompt == 0.5, 1.0, 0.0)
        w_b = torch.where(prompt == 0.0, 1.0, 0.0)

        fea_en = self.cp(self.conv_dil(fea)+fea)

        fea_chunk = torch.chunk(fea_en, chunks=3, dim=1)
        fea_f = fea_chunk[0]
        fea_u = fea_chunk[1]
        fea_b = fea_chunk[2]


        fea_fu = w_f * fea_f + w_u * fea_u
        fea_bu = w_b * fea_b + w_u * fea_u

        focus_f = self.SA_f(self.CA_f(fea_fu) * fea_fu+fea_fu)
        focus_b = self.SA_b(self.CA_b(fea_bu) * fea_bu+fea_bu)

        en_f = self.fuse_f(torch.cat([focus_f * fea_fu, (1 - focus_b) * fea_bu], 1))
        en_b = self.fuse_b(torch.cat([focus_b * fea_bu, (1 - focus_f) * fea_fu], 1))

        out = self.fuse(torch.cat([en_f, en_b], 1))

        return out


class DeformConv_input1(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_input1, self).__init__()

        self.conv_channel_adjust = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size[0] * kernel_size[1],
                                             kernel_size=(1, 1))

        self.offset_net = nn.Conv2d(in_channels=2 * kernel_size[0] * kernel_size[1],
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    groups=2 * kernel_size[0] * kernel_size[1],
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        x_chan = self.conv_channel_adjust(x)
        offsets = self.offset_net(x_chan)
        out = self.deform_conv(x, offsets)
        return out


class DeformConv_input2(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_input2, self).__init__()

        self.conv_channel_adjust = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size[0] * kernel_size[1],
                                             kernel_size=(1, 1))

        self.offset_net = nn.Conv2d(in_channels=2 * kernel_size[0] * kernel_size[1],
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    groups=2 * kernel_size[0] * kernel_size[1],
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x, y):
        y_chan = self.conv_channel_adjust(y)
        offsets = self.offset_net(y_chan)
        out = self.deform_conv(x, offsets)
        return out

class FTM1(nn.Module):    ########## dis
    def __init__(self, channel):
        super(FTM1, self).__init__()
        self.proj_liner1 = nn.Conv2d(channel, channel, 1)
        self.activation = nn.GELU()
        self.DWConv1 = nn.Conv2d(channel, channel, kernel_size=(5, 5), padding=2, groups=channel)
        self.DWConv2 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding=3, groups=channel)

        self.DFConv1 = DeformConv_input2(channel, groups=channel)
        self.DFConv2 = DeformConv_input2(channel, groups=channel)

        self.proj_21 = nn.Conv2d(channel, channel, 1)
        #self.proj_2 = BasicConv2d(in_channel, out_channel, 1)
        #self.proj_2 = Convbnrelu1(channel//2, channel)
    def forward(self, fea):
        fea_pro = self.activation(self.proj_liner1(fea))

        fea_d1 = self.DWConv1(fea_pro)
        fea_sample1 = self.DFConv1(fea_pro, fea_d1)

        fea_d2 = self.DWConv2(fea_d1)
        fea_sample2 = self.DFConv2(fea_sample1, fea_d2)

        out = self.proj_21(fea_sample2)+fea
        #out = fea_en+fea
        return out

class FTM2(nn.Module):  ########## dis
    def __init__(self, channel):
        super(FTM2, self).__init__()
        self.proj_liner1 = nn.Conv2d(channel, channel, 1)
        self.activation = nn.GELU()
        self.DWConv1 = nn.Conv2d(channel, channel, kernel_size=(5, 5), padding=2, groups=channel)
        self.DWConv2 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding=3, groups=channel)

        self.DFConv1 = DeformConv_input2(channel, groups=channel)
        self.DFConv2 = DeformConv_input2(channel, groups=channel)

        self.proj_21 = nn.Conv2d(channel, channel, 1)
        #self.proj_2 = BasicConv2d(in_channel, out_channel, 1)
        #self.proj_2 = Convbnrelu1(channel//2, channel)

    def forward(self, fea):
        fea_pro = self.activation(self.proj_liner1(fea))

        fea_d1 = self.DWConv1(fea_pro)
        fea_sample1 = self.DFConv1(fea_pro, fea_d1)

        fea_d2 = self.DWConv2(fea_d1)
        fea_sample2 = self.DFConv2(fea_pro, fea_d2)

        out = self.proj_21(fea_sample1+fea_sample2)+fea
        #out = fea_en+fea
        return out


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

class FTM(nn.Module):  ########## dis
    def __init__(self, channel):
        super(FTM, self).__init__()
        self.proj_liner1 = nn.Conv2d(channel, channel, 1)
        self.activation = nn.ReLU(inplace=True)
        self.DWConv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=(5, 5), padding=2, groups=channel),
                       nn.BatchNorm2d(channel),
                       nn.ReLU(inplace=True))


        self.DWConv2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=(7, 7), padding=3, groups=channel),
                       nn.BatchNorm2d(channel),
                       nn.ReLU(inplace=True))

        self.DFConv1 = DeformConv_input2(channel, groups=channel)
        self.DFConv2 = DeformConv_input2(channel, groups=channel)

        self.proj_21 = nn.Conv2d(channel, channel, 1)
        #self.proj_2 = BasicConv2d(in_channel, out_channel, 1)
        #self.proj_2 = Convbnrelu1(channel//2, channel)


    def forward(self, fea):
        fea_pro = self.activation(self.proj_liner1(fea))

        fea_d1 = self.DWConv1(fea_pro)
        fea_sample1 = self.DFConv1(fea_pro, fea_d1)

        fea_d2 = self.DWConv2(fea_d1)
        fea_sample2 = self.DFConv2(fea_sample1, fea_d2)

        out = self.proj_21(fea_sample2)+fea
        #out = fea_en+fea
        return out


class OurNet(nn.Module):
    def __init__(self, norm_layer=None, channel=32, en_channel=[64, 128, 320, 512]):
        super(OurNet, self).__init__()

        self.backbone1 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone2 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.REM1 = REM(en_channel[0])
        self.REM2 = REM(en_channel[1])
        self.REM3 = REM(en_channel[2])
        self.REM4 = REM(en_channel[3])

        self.FTM1 = FTM(en_channel[0])
        self.FTM2 = FTM(en_channel[1])
        self.FTM3 = FTM(en_channel[2])
        self.FTM4 = FTM(en_channel[3])

        self.decoder = DecoderNet(channel)
        self.decoder2 = DecoderNet(channel)
        self.pre = nn.Conv2d(channel*2, 1, kernel_size=3, stride=1, padding=1)
        self.pre2 = nn.Conv2d(channel*2, 1, kernel_size=3, stride=1, padding=1)

        self.input = BasicConv2d(4, 3, 1)

    def forward(self, inputs, mode):
        image_shape = inputs[0].size()[2:]
        if mode == 0:
            x1 = self.backbone1.forward_features_stage1(inputs[0])
            x2 = self.backbone1.forward_features_stage2(x1)
            x3 = self.backbone1.forward_features_stage3(x2)
            x4 = self.backbone1.forward_features_stage4(x3)

            fea_fuse = self.decoder([x1, x2, x3, x4], image_shape)
            pre = self.pre(fea_fuse)
            pre = F.interpolate(pre, size=image_shape, mode='bilinear')
            return pre

        if mode == 1:
            en1 = self.backbone2.forward_features_stage1(self.input(torch.cat([inputs[0], inputs[1]], 1)))
            en1 = self.REM1(en1, inputs[1])
            en2 = self.backbone2.forward_features_stage2(en1)
            en2 = self.REM2(en2, inputs[1])
            en3 = self.backbone2.forward_features_stage3(en2)
            en3 = self.REM3(en3, inputs[1])
            en4 = self.backbone2.forward_features_stage4(en3)
            en4 = self.REM4(en4, inputs[1])
            fea_fuse = self.decoder2([en1, en2, en3, en4], image_shape)
            pre_seg = self.pre2(fea_fuse)
            pre_seg = F.interpolate(pre_seg, size=image_shape, mode='bilinear')
            return pre_seg

        if mode == 2:
            ori1 = self.backbone1.forward_features_stage1(inputs[0])
            ori1 = self.FTM1(ori1)
            ori2 = self.backbone1.forward_features_stage2(ori1)
            ori2 = self.FTM2(ori2)
            ori3 = self.backbone1.forward_features_stage3(ori2)
            ori3 = self.FTM3(ori3)
            ori4 = self.backbone1.forward_features_stage4(ori3)
            ori4 = self.FTM4(ori4)
            fea_fin = self.decoder([ori1, ori2, ori3, ori4], image_shape)
            pre_fin = self.pre(fea_fin)
            pre_fin = F.interpolate(pre_fin, size=image_shape, mode='bilinear')

            return pre_fin

    def load_pre(self, pre_model):
        self.backbone1.load_state_dict(torch.load(pre_model),strict=False)
        self.backbone2.load_state_dict(torch.load(pre_model),strict=False)
        #self.backbone.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"PVT loading pre_model ${pre_model}")

