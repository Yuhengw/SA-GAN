import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# import common
# Convolution layers
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,
                               dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


# irnn layer
class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


# Attention
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


# SAM
class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        input = x
        input_x = self.conv_in(input)
        if self.attention:
            attention = self.attention_layer(input)
        x = self.relu1(input_x)
        top = self.irnn1(x)
        top_sum = self.conv1(x) + sum(top)
        bottom = self.irnn2(x)
        bottom_sum = self.conv1(x) + sum(bottom)
        top_feature = self.relu2(torch.cat([x, top_sum], 1))
        bottom_feature = self.relu2(torch.cat([x, bottom_sum], 1))
        top_feature = self.conv2(torch.cat([top_feature, self.conv1(x)], 1))
        bottom_feature = self.conv2(torch.cat([bottom_feature, self.conv1(x)], 1))
        result = self.conv_out(self.conv3(torch.cat([top_feature, bottom_feature], 1)))
        result = self.sigmod(result)
        if self.attention:
            result = result * attention
        return result, attention


# SPANet
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet, self).__init__()
        self.in_channels = 6
        self.out_channels = 64
        self.mid_channels = 128

        self.conv_in = conv3x3(self.in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.block1 = SAM(self.out_channels, self.mid_channels)
        self.block2 = SAM(self.mid_channels, self.mid_channels)
        self.block3 = SAM(self.mid_channels, self.mid_channels)

        self.relu2 = nn.ReLU(True)
        self.conv_out = conv3x3(self.mid_channels, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        input = self.conv_in(x)
        input = self.relu1(input)
        out, attention1 = self.block1(input)
        out, attention2 = self.block2(out)
        out, attention3 = self.block3(out)
        out = self.relu2(out)
        out = self.conv_out(out)
        out = self.tanh(out)
        return [attention1, attention2, attention3], out


# Generators
class Generator(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gen = SPANet()

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)


# GeneratorGroup
class GeneratorGroup(nn.Module):
    def __init__(self, num_gens, gpu_ids):
        super().__init__()
        self.num_gens = num_gens
        self.generators = nn.ModuleList([Generator(gpu_ids) for _ in range(num_gens)])

    def forward(self, x_list):
        assert len(x_list) == self.num_gens, "The length of the input list must match the number of generators."
        return [gen(x) for gen, x in zip(self.generators, x_list)]
