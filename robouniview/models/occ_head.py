
import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
from robouniview.models.pycls_model import regnetx
from einops import rearrange, repeat


class Upsample2d_3d_tiny(nn.Module):
    def __init__(self, in_channels=1024, out_channels=128, z=10):
        super().__init__()

        self.out_channels = out_channels
        self.z = z

        self.Up2d_3d = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels*z, kernel_size=1, padding=0, bias=False),
                            nn.BatchNorm2d(out_channels*z),
        )
        self.Conv3d_1 = nn.Sequential(
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x1):
        x1 = self.Up2d_3d(x1)
        x1 = rearrange(x1, " B (C Z) H W ->B C Z H W", C=self.out_channels, Z=self.z)
        x1 = self.Conv3d_1(x1)
        return  x1, x1

class Decoder_3d_tiny(nn.Module):
    def __init__(self, in_channels=128, out_channels=4):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
       
        self.Conv3d_1 = nn.Sequential(
                            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(in_channels, int(in_channels/2), kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(int(in_channels/2)),       
        )
       
        self.Upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.Conv3d_3 = nn.Sequential(
                            nn.Conv3d(int(in_channels/2), int(in_channels/4), kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(int(in_channels/4)),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(int(in_channels/4), out_channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1):
        x1 = self.Conv3d_1(x1)
        x2 = self.Upsample(x1)
        x3 = self.Conv3d_3(x2)
        return  x3


class Upsample2d_3d(nn.Module):
    def __init__(self, in_channels=1024, out_channels=128, z=10):
        super().__init__()

        self.out_channels = out_channels
        self.z = z

        self.Up2d_3d = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels*z, kernel_size=1, padding=0, bias=False),
                            nn.BatchNorm2d(out_channels*z),
        )
        self.Conv3d_1 = nn.Sequential(
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
        )

        self.MaxPool3d_1 = nn.MaxPool3d(2, stride=2)

        self.Conv3d_2 = nn.Sequential(
                            nn.Conv3d(out_channels, out_channels*2, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels*2),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(out_channels*2, out_channels*2, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels*2),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
        )

        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.Conv3d_3 = nn.Sequential(
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(out_channels),
                
        )

    def forward(self, x1):
        x1 = self.Up2d_3d(x1)
        x1 = rearrange(x1, " B (C Z) H W ->B C Z H W", C=self.out_channels, Z=self.z)
        x1 = self.Conv3d_1(x1)
        x2 = self.MaxPool3d_1(x1)
        x2 = self.Conv3d_2(x2)
        x3 = self.Upsample(x2) + x1
        x4 = self.Conv3d_3(x3) + x3

        return  x4, x2
    

class Decoder_3d(nn.Module):
    def __init__(self, in_channels=128, out_channels=4):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
       
        self.Conv3d_1 = nn.Sequential(
                            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(in_channels),
                            nn.ReLU(inplace=True),
                            # nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                            # nn.BatchNorm3d(in_channels),  
                            # nn.ReLU(inplace=True),
                            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(in_channels),       
        )
        self.Conv3d_2 = nn.Sequential(
                            nn.Conv3d(in_channels, int(in_channels/2), kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(int(in_channels/2)),
                            nn.ReLU(inplace=True),
                            # nn.Conv3d(int(in_channels/2), int(in_channels/2), kernel_size=3, padding=1, bias=False),
                            # nn.BatchNorm3d(int(in_channels/2)),
                            # nn.ReLU(inplace=True),
                            nn.Conv3d(int(in_channels/2), int(in_channels/2), kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(int(in_channels/2)),
        )
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Conv3d_3 = nn.Sequential(
                            nn.Conv3d(int(in_channels/2), int(in_channels/4), kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm3d(int(in_channels/4)),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(int(in_channels/4), out_channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1):
        x1 = self.Conv3d_1(x1)
        x2 = self.Upsample(x1)
        x2 = self.Conv3d_2(x2)
        x3 = self.Upsample(x2)
        x3 = self.Conv3d_3(x3)
        return  x3
    







class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class FastEncoderHead(nn.Module):
    def __init__(self,
                 layers_config
                 ):
        # explicitly define out channels in config
        self.layers_config = layers_config

        super().__init__()
       

    # def _setup(self):
        upsample_ratio = self.layers_config.get('upsample', 1)
        upsample_layers = []
        while upsample_ratio > 1:
            upsample_layers.append(
                nn.ConvTranspose2d(self.layers_config['in_channels'], self.layers_config['in_channels'], 4, 2, 1,
                                   bias=False))
            upsample_layers.append(nn.BatchNorm2d(self.layers_config['in_channels']))
            upsample_layers.append(nn.ReLU(True))
            upsample_ratio /= 2

        self.upsample = nn.Sequential(*upsample_layers)
        self.conv1 = nn.Conv2d(self.layers_config['in_channels'], 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        head_module = self.layers_config.get('head_module', 'regnet')
        if head_module == 'regnet800MF':
            trunk = regnetx("800MF", pretrained=False)
            self.layer1 = trunk.s2
            self.layer2 = trunk.s3
            self.outps = 288
        elif head_module == 'regnet950MF':
            trunk = regnetx("950MF", pretrained=False)
            self.layer1 = trunk.s2
            self.layer2 = trunk.s3
            self.outps = 288
        else:
            trunk = resnet18(pretrained=False, zero_init_residual=True)
            self.layer1 = trunk.layer2
            self.layer2 = trunk.layer3
            self.outps = 256

        self.up2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, self.layers_config['out_channels'], kernel_size=1, padding=0),
        )
        self.cbup1 = nn.Sequential(
            nn.Conv2d(self.outps, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.cbup2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.cb3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        # Apply upsampling
        x = self.upsample(x)

        # Apply Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x3 = self.cbup1(x3)
        x3 = x2 + x3
        x3 = self.cbup2(x3)
        x4 = self.cb3(x1)
        x3 = x4 + x3
        # Apply upsampling
        x = self.up2(x3)

        return x