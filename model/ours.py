import torch.nn as nn
import torch
from .ffc import FFCResnetBlock
from torchvision import models
import torch.nn.functional as F
import functools


def init_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch, bias=True)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            activation
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            activation,
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
            activation
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GlobalWeightPredictor(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, activation=nn.ReLU(inplace=True)):

        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_channels, hidden_dim)
        self.relu = activation
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = activation

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, in_dim, out_dim, flag=1):
        super().__init__()
        self.flag = flag
        self.conv1 = conv_layer(in_dim,out_dim,kernel_size=3,groups=in_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv_layer(out_dim, out_dim,kernel_size=7,groups=in_dim)
        self.act2 = nn.LeakyReLU()
        self.conv3 = conv_layer(out_dim, out_dim,kernel_size=3,groups=in_dim)
        self.act3 = nn.LeakyReLU()
        self.conv4 = conv_layer(out_dim, out_dim,kernel_size=3)
        self.act4 = nn.LeakyReLU()

    def forward(self, x):
        if self.flag == 1:
            x1 = self.conv1(x)
            x1 = self.act1(x1)
            x2 = self.conv2(x + x1)
            x2 = self.act2(x2)
            x3 = self.conv3(x + x1 + x2)
            x3 = self.act3(x3)
            out = self.conv4(x + x1 + x2 + x3)
            out = self.act4(out)
        elif self.flag == 2:
            x1 = self.conv1(x) + x
            x1 = self.act1(x1)
            x2 = self.conv2(x1) + x1
            x2 = self.act2(x2)
            x3 = self.conv3(x2) + x2
            x3 = self.act3(x3)
            out = self.conv4(x3) + x3
            out = self.act4(out)
        elif self.flag == 3:
            x1 = self.conv1(x)
            x1 = self.act1(x1)
            x2 = self.conv2(x1)
            x2 = self.act2(x2)+x1
            x3 = self.conv3(x2)
            x3 = self.act3(x3)+x
            out = self.conv4(x3)
            out = self.act4(out)
        else:
            x1 = self.conv1(x)
            x1 = self.act1(x1)
            x2 = self.conv2(x1)
            x2 = self.act2(x2)
            x3 = self.conv3(x2)
            x3 = self.act3(x3)
            out = self.conv4(x3)
            out = self.act4(out)

        return out

class AdaptiveGammaAdjustment(nn.Module):
    def __init__(self, num_gamma=4, input_channel = 3, basechannels=64, scale = 4):
        super().__init__()
        self.num_gamma = num_gamma
        hidden_dim = basechannels * scale

        self.conv = nn.Conv2d(input_channel, basechannels, kernel_size=3, stride=1, padding=1, bias=True)
        self.fe = FeatureExtraction(basechannels, basechannels,1)

        self.mlp = GlobalWeightPredictor(basechannels,hidden_dim,num_gamma)
        self.mlp2 = SpatialAttention(basechannels, hidden_dim, 3*num_gamma)
        self.mlp3 = GlobalWeightPredictor(3*num_gamma, scale*(3*num_gamma), num_gamma, nn.LeakyReLU())


    def forward(self, A):
        B, C, H, W = A.shape
        A1 = self.fe(self.conv(A))
        gammas = self.mlp(A1)
        lineafactors = self.mlp2(A1)
        noisebisa = self.mlp3(lineafactors)

        A_expanded = A.unsqueeze(2).expand(-1, -1, self.num_gamma, -1, -1)
        gamma_expanded = gammas.view(B, 1, self.num_gamma, 1, 1)
        lineafactors_expanded = lineafactors.view(B, 3, self.num_gamma, H, W)
        noisebisa = noisebisa.view(B, 1, self.num_gamma, 1, 1)

        A3 = noisebisa + lineafactors_expanded * torch.pow(A_expanded, gamma_expanded)

        A3 = A3.permute(0, 2, 1, 3, 4).reshape(B, 3 * self.num_gamma, H, W)
        return A3


class AdaptiveFusion(nn.Module):
    def __init__(self, num_exposures=4, basechannels=64, scale=4):
        super().__init__()
        self.num_exposures = num_exposures
        hidden_dim = basechannels * scale

        self.conv = nn.Conv2d(3*(num_exposures+1), basechannels, kernel_size=3, stride=1, padding=1,bias=True)
        self.feature_extract = FeatureExtraction(basechannels,basechannels,1)

        self.spatial_attention = GlobalWeightPredictor(basechannels,hidden_dim,3*(num_exposures + 1), nn.LeakyReLU())
        self.mlp = GlobalWeightPredictor(3 +basechannels, scale * (3 +basechannels), 3, nn.LeakyReLU())

    def forward(self, A, A_multi):
        B, C, H, W = A.shape
        N = self.num_exposures

        Q = torch.cat([A, A_multi], dim=1)

        features = self.feature_extract(self.conv(Q))

        spatial_weights = self.spatial_attention(features)
        spatial_weights = spatial_weights.view(B, 3, N+1, 1, 1).float()

        noisebisa = self.mlp(torch.cat([A, features], dim=1))
        noisebisa = noisebisa.view(B, 3, 1, 1).float()

        imgs = Q.view(B, 3, N + 1, H, W)
        fused = noisebisa + (imgs * spatial_weights).sum(dim=2)

        return fused

class SecondProcessModel(nn.Module):
    def __init__(self, dim=16, num_blocks=3, in_channels=3):

        super(SecondProcessModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, 3, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fft_blocks = nn.ModuleList([FFCResnetBlock(dim) for _ in range(num_blocks)])
        self.upconv1 = nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True)
        self.upconv_last = nn.Conv2d(dim, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def _upsample(self, x, skip, upconv):
        x = self.lrelu(self.pixel_shuffle(upconv(torch.cat((x, skip), dim=1))))
        return x

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        fft_features = x3
        for fft_block in self.fft_blocks:
            fft_features = fft_block(fft_features)
        out_noise = self._upsample(fft_features, x3, self.upconv1)
        out_noise = self._upsample(out_noise, x2, self.upconv2)
        out_noise = self.upconv3(torch.cat((out_noise, x1), dim=1))
        out_noise = self.upconv_last(out_noise)
        out_noise = out_noise + x
        B, C, H, W = x.size()
        out_noise = out_noise[:, :, :H, :W]
        return out_noise


class MYMODEL(nn.Module):
    def __init__(self):
        super(MYMODEL, self).__init__()
        self.basechannles = 64
        self.basechannles2 = 16
        self.num_gamma = 6
        self.aga = AdaptiveGammaAdjustment(num_gamma=self.num_gamma, basechannels=self.basechannles, scale = 4)
        self.af = AdaptiveFusion(num_exposures=self.num_gamma, basechannels=self.basechannles)
        self.refinenet = SecondProcessModel(dim=self.basechannles2, num_blocks=3, in_channels=3)

    def forward(self, input):
        x = self.aga(input)
        x1 = self.af(input, x)
        out = self.refinenet(x1)

        return out