import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from io import BytesIO

from .palette import Palette
from .dithering import error_diffusion_dithering

import torchvision.transforms as transforms

class PNGCompression(object):
    def __init__(self, ):
        pass

    def __call__(self, img):
        png_buffer = BytesIO()
        img.save(png_buffer, "PNG")
        return img
    
    
class MedianCut(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            if not self.dither:
                sampled_img = img.quantize(colors=self.num_colors, method=0).convert('RGB')
            else:
                palette = Palette(img.quantize(colors=self.num_colors, method=0))
                sampled_img = error_diffusion_dithering(img, palette).convert('RGB')
        else:
            sampled_img = img
        return sampled_img

def get_data(data_path, args):
    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform = transforms.Compose([MedianCut(args.num_colors,), PNGCompression(), transforms.ToTensor(), normalize])
        
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        
    return dst_train
