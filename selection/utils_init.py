from torchvision import datasets, transforms
from torch import tensor, long

import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from io import BytesIO

sys.path.append('../utils')

from palette import Palette
from dithering import error_diffusion_dithering


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
    
    
def CIFAR10(data_path, n_colors):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def CIFAR10_Medcut(data_path, n_colors):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([MedianCut(n_colors,), PNGCompression(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def CIFAR100_Medcut(data_path, n_colors):
    channel = 3
    im_size = (32, 32)
    num_classes = 100
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([MedianCut(n_colors,), PNGCompression(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def Tiny_Medcut(data_path, n_colors):
    channel = 3
    im_size = (64, 64)
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([MedianCut(n_colors,), PNGCompression(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
    dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform_test)
    # print(dst_train.targets)
    print(np.min(dst_train.targets))
    print(np.max(dst_train.targets))
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    class_names = dst_train.classes
    
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def SVHN_Medcut(data_path, n_color):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([MedianCut(n_color,), PNGCompression(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
    dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform_test)
    class_names = [str(c) for c in range(num_classes)]
    # dst_train.targets = tensor(dst_train.targets, dtype=long)
    # dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def get_dataset(dataset, data_path, n_colors):
    datasets = {
        'CIFAR10_og': CIFAR10,
        'CIFAR10_Medcut': CIFAR10_Medcut,
        'CIFAR100_Medcut': CIFAR100_Medcut,
        'SVHN_Medcut': SVHN_Medcut,
        'Tiny_Medcut': Tiny_Medcut,
    }
    
    return datasets[dataset](data_path, n_colors)
    
    