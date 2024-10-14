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
    


class ColorCNNPlus(nn.Module):
    def __init__(self, num_colors, layer_ids=[3, 7, 11], in_channel=3, im_size=(32, 32), soften=1, color_norm=1, color_jitter=0, temp=1.0, topk=4, agg='mean'):
        super().__init__()
        self.topk = topk
        self.temp = temp
        self.agg = agg
        self.num_colors = num_colors
        self.soften = soften
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        # self.base = UNet(3) if arch == 'unet' else DnCNN(3)
        # self.base = UNet(3)
        self.color_mask = nn.Sequential(nn.Conv2d(in_channel, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()
        self.layer_ids = layer_ids
        self.im_size = im_size

    # def assign_backbone(self, backbone):
    #     self.backbone = backbone

    def forward(self, img, activation_maps=None, training=True):
        # feat = self.base(img)
        # print('feat shape is ', feat.shape)
        # m = self.color_mask(feat)
        B, _, H, W = img.shape

        '''Make layer cam activation maps'''
        if activation_maps is not None:
            out = torch.cat([img, activation_maps], dim=1)
            m = self.color_mask(out)
        else:
            # out = self.base(img)
            # m = self.color_mask(out)
            m = self.color_mask(img)

        m = m.view([B, -1, self.num_colors, H, W]).mean(dim=1)
        topk, idx = torch.topk(m, min(self.topk, self.num_colors), dim=1)
        m = torch.scatter(torch.zeros_like(m), 1, idx, F.softmax(topk / self.temp, dim=1)) + 1e-16  # softmax output
        color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / \
                        m.unsqueeze(1).sum(dim=[3, 4], keepdim=True)

        if training:
            transformed_img = (m.unsqueeze(1) * color_palette).sum(dim=2)
        else:
            M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
            M = torch.zeros_like(m).scatter(1, M, 1)
            transformed_img = (M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette
    
    
class PixelSimLoss(nn.Module):
    def __init__(self, sample_ratio=0.1, normalize=True):
        super(PixelSimLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.normalize = normalize

    def forward(self, featmap_src, featmap_tgt, visualize=False):
        B, C, H, W = featmap_src.shape
        sample_idx = [np.random.choice(H * W, int(H * W * self.sample_ratio), replace=False) for _ in range(B)]
        sample_idx = np.stack(sample_idx, axis=0).reshape([B, 1, int(H * W * self.sample_ratio)]).repeat(C, axis=1)
        f_src, f_tgt = featmap_src.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_src.device)), \
            featmap_tgt.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_tgt.device))
        A_src, A_tgt = torch.bmm(f_src.permute([0, 2, 1]), f_src), torch.bmm(f_tgt.permute([0, 2, 1]), f_tgt)
        if self.normalize:
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=[1, 2]) ** 2 / sample_idx.shape[-1])
        else:
            # loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) / sample_idx.shape[-1])
            loss_semantic = F.binary_cross_entropy(A_src, A_tgt)
        if visualize:
            import matplotlib.pyplot as plt
            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # fig, ax = plt.subplots(figsize=(10, 2))
            # im = ax.imshow(f_src[0].detach().cpu().numpy(), cmap='GnBu', vmin=0, vmax=1)
            # divider = make_axes_locatable(ax)
            # cax = divider.new_horizontal(size="5%", pad=1, pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(im, cax=cax, orientation="vertical")
            # plt.show()

            fig, ax = plt.subplots(figsize=(10, 2))
            ax.imshow(f_src[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('f_src.png')
            plt.show()
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.imshow(f_tgt[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('f_tgt.png')
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(A_src[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('A_src.png')
            plt.show()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(A_tgt[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('A_tgt.png')
            plt.show()

        return loss_semantic
