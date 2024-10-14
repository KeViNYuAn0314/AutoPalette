import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from .layercam import LayerCAM
import copy


class DnCNN(nn.Module):
    def __init__(self, in_channels, num_of_layers=17, feature_dim=64):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(feature_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.out_channel = feature_dim

    def forward(self, x):
        out = self.dncnn(x)
        return out
    

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out_channel = 64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
    


class ColorCNN(nn.Module):
    def __init__(self, arch, num_colors, layer_ids=[3, 7, 11], in_channel=3, im_size=(32, 32), soften=1, color_norm=1, color_jitter=0):
        super().__init__()
        self.num_colors = num_colors
        self.soften = soften
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        # self.base = UNet(3)
        self.color_mask = nn.Sequential(nn.Conv2d(3, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()
        self.layer_ids = layer_ids
        self.im_size = im_size

    # def assign_backbone(self, backbone):
    #     self.backbone = backbone

    def forward(self, img, activation_maps=None, training=True):
        # feat = self.base(img)
        # # print('feat shape is ', feat.shape)
        # m = self.color_mask(feat)

        '''Make layer cam activation maps'''
        # if backbone is not None:
        #     img_cp = copy.deepcopy(img.detach())
        #     layer_cams = []
        #     # with torch.no_grad():
        #     for i in range(len(self.layer_ids)):
        #         layer_name = 'features_' + str(self.layer_ids[i])
        #         model_dict = dict(type='ConvNetD3', arch=backbone, layer_name=layer_name, input_size=self.im_size)
        #         layercam = LayerCAM(model_dict)
        #         layercam_map = layercam(img_cp).squeeze(0)
        #         layer_cams.append(layercam_map)
        #     layer_cams = torch.stack(layer_cams, dim=1)
        #     out = torch.cat([img, layer_cams], dim=1)
        #     # self.backbone = None
        #     m = self.color_mask(out)
        # else:
        #     m = self.color_mask(img)
        
        if activation_maps is not None:
            out = torch.cat([img, activation_maps], dim=1)
            m = self.color_mask(out)
        else:
            m = self.color_mask(img)

        # m = self.color_mask(img)
        # print('mask shape is ', m.shape)
        m = self.mask_softmax(self.soften * m)  # softmax output
        # print('after softmax shape is ', m.shape)
        M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
        indicator_M = torch.zeros_like(m).scatter(1, M, 1)
        # print(img.unsqueeze(2).shape)
        # print(m.unsqueeze(1).shape)
        # print((img.unsqueeze(2) * m.unsqueeze(1)).shape)
        if training:
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (   # (1, 3, 1, 32, 32) * (1, 1, 64, 32, 32) = (1, 3, 64, 32, 32)
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            jitter_color_palette = color_palette + self.color_jitter * np.random.randn()
            transformed_img = (m.unsqueeze(1) * jitter_color_palette).sum(dim=2)
        else:
            color_palette = (img.unsqueeze(2) * indicator_M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (indicator_M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette
