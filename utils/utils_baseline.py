# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP, ResNet18BN, ResNet18_Tiny, ResNet18BN_Tiny,VGG11_Tiny
import math
from torch.utils.data import Subset
import torchvision
from sklearn.metrics import f1_score
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class WarehouseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load all image paths and their corresponding labels
        for _, class_name in enumerate(os.listdir(root_dir)):
            label = CLASSES.index(class_name)
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png') or img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        # print(f'there are {len(self.labels)} labels')
        # print(f'there are {len(self.image_paths)} images')
        # print(self.image_paths[0])
        # print(self.labels[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



# Custom dataset class to load the resized images and labels
class ResizedImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = [os.path.join(root, filename) for root, _, files in os.walk(root_dir) for filename in files if filename.endswith('.pt')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = torch.load(image_path)
        
        class_name = os.path.basename(os.path.dirname(image_path))
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()


# rate: data reduction rate
# class_num: number of classes of the reduced dataset
def reduce_dataset(train_set, rate=1, class_num=10, num_per_class = 5000):
    indx = np.argsort(np.array(train_set.targets)).tolist()
    print('dataset contain {} classes'.format(class_num))
    ind = []
    for i in range(class_num):
        mid = [indx[m] for m in range(i*num_per_class,int(i*num_per_class+num_per_class*rate))]
        ind.extend(mid)
    train_set=Subset(train_set,ind)
    return train_set


def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None, baseline=False):

    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if baseline:
            transform =transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
    elif dataset == 'ImbalancedCIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if baseline:
            transform =transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = ImbalanceCIFAR10(data_path, train=True, download=True, transform=transform, imb_type=args.imbalance_type, imb_factor=args.imbalance_rate)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if baseline:
            transform =transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
        
    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
        # class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        config.img_net_classes = config.dict[subset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])

        dst_train = datasets.ImageNet(data_path, split="train", transform=transform) # no augmentation
        dst_train_dict = {c : torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c])))) for c in range(len(config.img_net_classes))}
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
        loader_train_dict = {c : torch.utils.data.DataLoader(dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16) for c in range(len(config.img_net_classes))}
        dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
        dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
        for c in range(len(config.img_net_classes)):
            dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
            dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c
        print(dst_test.dataset)
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
        class_names = None


    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if baseline:
            transform =transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
        
    elif dataset == 'CIFAR_Warehouse':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'test')
        print('get dataset ...')
        dst_train = WarehouseDataset(train_path, transform=transform)
        dst_test = WarehouseDataset(test_path, transform=transform)
        class_names = CLASSES
        class_map = {x: x for x in range(num_classes)}

    elif dataset == 'ImageNet1K':
        channel = 3
        im_size = (64, 64)
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(im_size),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.Resize(im_size),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create datasets and data loaders for training and testing
        dst_train = ResizedImageNetDataset(root_dir=os.path.join(data_path, "train"), transform=data_transforms['train'])
        dst_test = ResizedImageNetDataset(root_dir=os.path.join(data_path, "val"), transform=data_transforms['val'])

        # dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=data_transforms['train']) # no augmentation
        # dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
    

    else:
        exit('unknown dataset: %s'%dataset)



    if args.zca:
        images = []
        labels = []
        print("Train ZCA")
        for i in tqdm.tqdm(range(len(dst_train))):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)

        images = []
        labels = []
        print("Test ZCA")
        for i in tqdm.tqdm(range(len(dst_test))):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)

        print(type(zca))


        args.zca_trans = zca


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)


    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), dist=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'MLP_Tiny':
        net = MLP(channel=channel, num_classes=num_classes,res=64)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'LeNet_Tiny':
        net = LeNet(channel=channel, num_classes=num_classes, res=64)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet_Tiny':
        net = AlexNet(channel=channel, num_classes=num_classes, res=64)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11_Tiny':
        net = VGG11_Tiny( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_Tiny':
        net = ResNet18_Tiny(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_Tiny':
        net = ResNet18BN_Tiny(channel=channel, num_classes=num_classes)
    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4BN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')


    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False, If_Float = False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    
    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()
    net = net.to(device)
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        if If_Float:
            lab = datum[1].float().to(device)
        else:
            lab = datum[1].long().to(device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=device)
        img = img.to(device)
        if args.dataset == "ImageNet" and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)


        if If_Float:
            acc = 1.
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def epoch_imbalance(mode, dataloader, net, optimizer, criterion, args, aug, texture=False, If_Float = False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    
    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()
    net = net.to(device)
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        if If_Float:
            lab = datum[1].float().to(device)
        else:
            lab = datum[1].long().to(device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=device)
        img = img.to(device)
        if args.dataset == "ImageNet" and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)


        if If_Float:
            acc = 1.
        else:
            predictions = np.argmax(output.cpu().data.numpy(), axis=-1)
            labels = lab.cpu().data.numpy()
            # acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
            f1_macro = f1_score(labels, predictions, average='macro')
            f1_weighted = f1_score(labels, predictions, average='weighted')
            f1_micro = f1_score(labels, predictions, average='micro')

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg




def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False, train_criterion=None, Preciser_Scheduler=False, type=1):
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    net = net.to(device)
    images_train.to(device)
    labels_train.to(device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if Preciser_Scheduler:
        LR_begin=0.0000000001
        LR_End = float(args.lr_net)
        if type==0:
            t=0
        else:
            t=500
        T=Epoch
        lambda1 = lambda epoch: ((LR_End-LR_begin)*epoch / t) if epoch < t else  LR_End * (1+math.cos(math.pi*(epoch - t)/(T-t)))/2.
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_End, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    '''test'''
    test_criterion = nn.CrossEntropyLoss().to(device)
    If_Float = True
    if train_criterion == None:
        train_criterion = nn.CrossEntropyLoss().to(device)
        If_Float = False

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, train_criterion, args, aug=True, texture=texture, If_Float=If_Float)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, test_criterion, args, aug=False, If_Float = False)
        
        # with torch.no_grad():
        #     acc_test_list.append(acc_test)
        
        if Preciser_Scheduler:
            scheduler.step()
        else:
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test




def evaluate_synset_imbalance(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False, train_criterion=None, Preciser_Scheduler=False, type=1):
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    net = net.to(device)
    images_train.to(device)
    labels_train.to(device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if Preciser_Scheduler:
        LR_begin=0.0000000001
        LR_End = float(args.lr_net)
        if type==0:
            t=0
        else:
            t=500
        T=Epoch
        lambda1 = lambda epoch: ((LR_End-LR_begin)*epoch / t) if epoch < t else  LR_End * (1+math.cos(math.pi*(epoch - t)/(T-t)))/2.
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_End, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    '''test'''
    test_criterion = nn.CrossEntropyLoss().to(device)
    If_Float = True
    if train_criterion == None:
        train_criterion = nn.CrossEntropyLoss().to(device)
        If_Float = False

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch_imbalance('train', trainloader, net, optimizer, train_criterion, args, aug=True, texture=texture, If_Float=If_Float)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch_imbalance('test', testloader, net, optimizer, test_criterion, args, aug=False, If_Float = False)
        
        # with torch.no_grad():
        #     acc_test_list.append(acc_test)
        
        if Preciser_Scheduler:
            scheduler.step()
        else:
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train f1 = %.4f, test f1 = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test




def evaluate_baseline(it_eval, net, trainloader, testloader, args, return_loss=False, texture=False, train_criterion=None, Preciser_Scheduler=False, type=1):
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    net = net.to(device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if Preciser_Scheduler:
        LR_begin=0.0000000001
        LR_End = float(args.lr_net)
        if type==0:
            t=0
        else:
            t=500
        T=Epoch
        lambda1 = lambda epoch: ((LR_End-LR_begin)*epoch / t) if epoch < t else  LR_End * (1+math.cos(math.pi*(epoch - t)/(T-t)))/2.
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_End, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    '''test'''
    test_criterion = nn.CrossEntropyLoss().to(device)
    If_Float = True
    if train_criterion == None:
        train_criterion = nn.CrossEntropyLoss().to(device)
        If_Float = False

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, train_criterion, args, aug=True, texture=texture, If_Float=If_Float)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, test_criterion, args, aug=False, If_Float = False)
        if Preciser_Scheduler:
            scheduler.step()
        else:
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test



def evaluate_synset_cross(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False, train_criterion=None, Preciser_Scheduler=False, type=1):
    if args.parall_eva==False:
        device = torch.device("cuda:0")
    else:
        device = args.device
    net = net.to(device)
    images_train.to(device)
    labels_train.to(device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if Preciser_Scheduler:
        LR_begin=0.0000000001
        LR_End = float(args.lr_net)
        if type==0:
            t=0
        else:
            t=500
        T=Epoch
        lambda1 = lambda epoch: ((LR_End-LR_begin)*epoch / t) if epoch < t else  LR_End * (1+math.cos(math.pi*(epoch - t)/(T-t)))/2.
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_End, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    '''test'''
    test_criterion = nn.CrossEntropyLoss().to(device)
    If_Float = True
    if train_criterion == None:
        train_criterion = nn.CrossEntropyLoss().to(device)
        If_Float = False

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, train_criterion, args, aug=True, texture=texture, If_Float=If_Float)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep % 100 == 0:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, test_criterion, args, aug=False, If_Float = False)
                acc_test_list.append(acc_test)
        
        # with torch.no_grad():
        #     acc_test_list.append(acc_test)
        
        if Preciser_Scheduler:
            scheduler.step()
        else:
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test, acc_test_list



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['AlexNet', 'VGG11', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model, ]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    elif eval_mode == 'BN':
        model_eval_pool = ['ConvNet','ConvNetBN','ResNet18','ResNet18BN','AlexNet', 'VGG11', 'ResNet18_AP']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}
