import torch
from torch import nn
from torch.nn import functional as F
import warnings
from einops import rearrange

warnings.filterwarnings("ignore")


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        self.in_channels = in_channels
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            pass
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = F.avg_pool2d(x, 2)
        c2 = self.conv2(c2)
        c3 = F.avg_pool2d(c2, 2)
        c3 = self.conv3(c3)
        c4 = F.avg_pool2d(c3, 2)
        c4 = self.conv4(c4)

        p4 = c4
        p3 = c3 + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = c2 + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = c1 + F.interpolate(p2, scale_factor=2, mode='nearest')

        return p1

class GLPM(nn.Module):
    def __init__(self, in_channel):
        super(GLPM, self).__init__()
        self.in_channel = in_channel
        self.fpn = FPN(in_channel, in_channel)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)

    def forward(self, x):
        output = self.fpn(x)
        output = self.activation(self.conv(torch.cat((x,output),1)))
        return output


class CRM(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CRM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MPFM(nn.Module):
    def __init__(self, levels, pool_type='max_pool'):
        super(MPFM, self).__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        B, C, H, W = x.size()
        pooling_output = []
        for level in self.levels:
            kernel_size = (H + level - 1) // level, (W + level - 1) // level
            stride = kernel_size
            #
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
            #
            tensor = F.interpolate(tensor, size=(H, W), mode='bilinear', align_corners=False)
            pooling_output.append(tensor)

        return torch.cat(pooling_output, dim=1)

class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        output = y + x
        return output

class E3DCM(nn.Module):
    def __init__(self, cin, cout):
        super(E3DCM, self).__init__()

        self.Conv_mixdence = nn.Conv3d(cin, cout, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.reconv = nn.ModuleList([
            BasicConv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            BasicConv3d(cout, cout, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),
            BasicConv3d(cout, cout, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        ])
        self.aggregation1 = nn.Conv3d(3 * cout, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.aggregation2 = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, x):
        x = torch.cat(x, 1)
        # print(x.shape)
        x = self.relu(self.Conv_mixdence(x))

        hsi_feas = []
        for layer in self.reconv:
            fea = layer(x)
            hsi_feas.append(fea)
        hsi_feas = torch.cat(hsi_feas, dim=1)
        output = self.aggregation1(hsi_feas) + x
        output = self.aggregation2(output) + output

        return output

class RSMB(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='lrelu', norm='none', sn=False, spp_levels=[1, 2, 4]):
        super(RSMB, self).__init__()

        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv2 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv3 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)

        self.spp = MPFM(spp_levels,'avg_pool')

        self.reconv_3d = E3DCM(1, 1)

        spp_output_channels = latent_channels * len(spp_levels)  # 假设每个level输出latent_channels通道
        self.conv4 = Conv2dLayer(spp_output_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv5 = Conv2dLayer(latent_channels * 3, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 3, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)

        self.se1 = CRM(in_channels)
        self.se2 = CRM(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x1_3d = self.reconv_3d([x.unsqueeze(1)])
        x2_3d = self.reconv_3d([x1_3d])
        x3_3d = self.reconv_3d([x2_3d])

        spp_output = self.spp(x)
        x3_2 = self.se1(x)

        x4 = self.conv4(torch.cat((spp_output, x3, x3_2, x1_3d.squeeze(1)), 1))
        x5 = self.conv5(torch.cat((x4, x2, x2_3d.squeeze(1)),1))
        x6 = self.conv6(torch.cat((x5, x1, x3_3d.squeeze(1)), 1)) + self.se2(x3_2)

        return x6


class P2D3D_HRNet(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=100, n_RSMBs=10):
        super(P2D3D_HRNet, self).__init__()
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.PReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)
        self.denosing = GLPM(channels)
        self.backbone = nn.ModuleList(
            [RSMB(channels, channels) for _ in range(n_RSMBs)])
        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        out = self.denosing(out)

        for i, block in enumerate(self.backbone):
            out = block(out)

        out = self.tail_conv2D(out)
        out = self.output_conv2D(self.output_prelu2D(out))
        return out
