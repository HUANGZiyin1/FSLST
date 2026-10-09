import torch
import torch.nn as nn
import torch.nn.functional as F
from octconv import *
import common

# ==========
# Spatio-temporal deformable fusion module
# ==========



class HAWT(nn.Module):
    def __init__(self, in_channel=3, in_channel_long = 5,nf=32, nb=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(HAWT, self).__init__()
        # u-shape backbone
        self.conv_local= nn.Sequential(
            nn.Conv2d(in_channel, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
        )

        self. MSRB1 = MSRB(nf)
        self. MSRB2 = MSRB(nf)
        self. MSRB3 = MSRB(nf)

        self.conv_global= nn.Sequential(
            nn.Conv3d(1, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ResN1 = ResBlock(nf)
        self.ResN2 = ResBlock(nf)
        self.ResN3 = ResBlock(nf)

        self.conv_trans3 = nn.MaxPool3d((in_channel_long, 1, 1), stride=1, padding=0)

        self. SA_l = SpatialGate()
        self. SA_g = SpatialGate()
        self.sigmoid = nn.Sigmoid()
        self. CA = ChannelAttention(nf*2)

        self.fusion= nn.Conv2d(2, 1, 5, padding=5 // 2, bias=False)

        self.qenet = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 1, 3, padding=3//2),
        )


    def forward(self, inputs, input_long, TFrm):

        input3d = input_long.unsqueeze(1)
        # Feature Extraction

        out_local = self.conv_local(inputs)#1,32,128,128
        out_1 = self.MSRB1(out_local)#1,32,128,128
        out_2 = self.MSRB2(out_1)#1,32,128,128
        out_3 = self.MSRB3(out_2)#1,32,128,128

        out_global = self.conv_global(input3d) # 1,32,3,128,128
        out_g1 = self.ResN1(out_global)  # 1,32,5,128,128
        out_g2 = self.ResN2(out_g1)      #1, 32, 3, 128, 128
        out_g3 = self.ResN3(out_g2)
        out_g3 = self.conv_trans3(out_g3) #1,32,1,128,128
        out_g3 = out_g3.squeeze(2) #1,32,128,128
        l_sw = self.SA_l(out_3)
        g_sw = self.SA_g(out_g3)
        sw = self.sigmoid(self.fusion(torch.cat([l_sw,g_sw],1)))
        o = torch.cat([out_3,out_g3],1)#[1, 64, 128, 128]
        cw = self.CA(o)
        o = o*sw*cw
        o = self.qenet(o) #1,1,128,128
        o = TFrm+o

        return o




# ==========
# Quality enhancement module
# ==========


class MSRB(nn.Module):
    def __init__(self, nf, k1 = 3, k2 = 5):
        super(MSRB, self).__init__()

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k1, padding=k1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*2, k1, padding=k1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k2, padding=k2//2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*2, k2, padding=k2//2),
            nn.ReLU(inplace=True),
        )

        self.confusion = nn.Conv2d(nf * 4, nf, 1, padding=0, stride=1)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(input_1)
        output_5_1 = self.conv_5_1(input_1)
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv3d(channel, channel, 3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 3, padding=3 // 2),
        )

    def forward(self, x):
        res = self.conv_bt(x)
        res = res+x
        return res


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1) // 2, bias=False),
            nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return scale


if __name__ == '__main__':
    model = HAWT()
    net_input = torch.randn(1, 3, 128, 128)
    net_input_long = torch.randn(1, 5, 128, 128)
    TFrm = torch.randn(1, 1, 128, 128)
    net_output= model(net_input,net_input_long,TFrm)
    print(net_output.shape)