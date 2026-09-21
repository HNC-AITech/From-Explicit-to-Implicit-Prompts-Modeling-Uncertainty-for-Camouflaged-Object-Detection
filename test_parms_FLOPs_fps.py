import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from setproctitle import setproctitle

setproctitle('yy')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
from torch.autograd import Variable
import torchvision.models as models
#from thop import profile


def print_model_parm_nums(model):
    # for param in model.parameters():
    #    print(param.nelement())
    total = sum([param.numel() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def print_model_parm_flops(model, crop_size):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = "cuda"
        dtype = torch.cuda.FloatTensor
    else:
        device = "cpu"
        dtype = torch.FloatTensor

    # input = Variable(torch.rand(3,crop_size,crop_size).unsqueeze(0), requires_grad = True).to(device)
    x = torch.rand(1, 3, crop_size, crop_size).type(dtype).to(device)
    # out = model(input).to(device)
    out = model(x).to(device)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))


def compute_fps(net, crop_size):
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = "cuda"
        dtype = torch.cuda.FloatTensor
    else:
        device = "cpu"
        dtype = torch.FloatTensor

    model = net.to(device)

    input = torch.rand(1, 3, crop_size, crop_size).type(dtype).to(device)
    N = 10
    model.eval()
    with torch.no_grad():
        fpss = list()
        for i in range(10):
            start_time = time.time()
            for n in range(N):
                # print("run: {}/{}".format(n + 1, i + 1))
                xxx = model(input)
            fpss.append(N / (time.time() - start_time))
        fps = np.mean(fpss)

    print("  + FPS = %.2f with %s." % (fps, device))

    return fps


if __name__ == '__main__':
    from Models.speedtest_implicit import OurNet

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = "cuda"
        dtype = torch.cuda.FloatTensor
    else:
        device = "cpu"
        dtype = torch.FloatTensor

    # 修改model和crop_size即可
    model = OurNet().to(device)
    crop_size = 384

    # ----------------------------------------FLOPs、 Params------------------------------------------------
    print_model_parm_flops(model, crop_size)            # 如果测试报错，请使用下面代码 macs即为FLOPs
    print_model_parm_nums(model)

    '''x1 = torch.rand(1, 3, crop_size, crop_size).type(dtype).to(device)
    x2 = torch.rand(1, 3, crop_size, crop_size).type(dtype).to(device)
    macs, params = profile(model, inputs=(x,))
    print("use thop test...")
    print("  + macs: %.2fG" % (macs / 1e9))
    print("  + params: %.2fM" % (params / 1e6))'''

    # ----------------------------------------fps------------------------------------------------
    fps = compute_fps(model, crop_size)