import os
import torch.nn as nn
import argparse

# from torchvision import transforms
from utils_init import *
from datetime import datetime
from time import sleep
import torch

from utils_init import get_dataset
from graphcut import Submodular
import math



def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10_Medcut', help='dataset')
    parser.add_argument('--n_color', type=int, default=64)
    parser.add_argument('--model', type=str, default='ConvNetD3', help='model')
    parser.add_argument('--selection', type=str, default="GraphCut", help="selection method")
    # parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--ipc', default=1, type=int, help='image per class')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of workers')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument("--nesterov", action='store_true', help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', "-b", default=1024, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch",  default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", action='store_true',
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--imbalance', action='store_true',
                        help="whether balance selection is performed per class")

    parser.add_argument('--save_path', type=str, default='./result', help='path to save results (default: do not save)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    

    checkpoint = {}
    start_exp = 0
    start_epoch = 0


    if args.save_path != "":
        checkpoint_name = "{dst}_{net}_{mtd}_feature_{ncolor}_exp{exp}_epoch_{ipc}_".format(dst=args.dataset,
                                                                                        net=args.model,
                                                                                        mtd=args.selection,
                                                                                        ncolor=args.n_color,
                                                                                        dat=datetime.now(),
                                                                                        exp=start_exp,
                                                                                        ipc=args.ipc)

    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
    #     (args.data_path)
    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv \
    #       = get_dataset(args.dataset, args.data_path, args.batch, args.subset, args=args)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, args.data_path, args.n_color)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    
    ''' Calculate budget'''
    args.ipc = args.ipc * (2**(8-math.ceil(math.log2(args.n_color))))

    torch.random.manual_seed(args.seed)

    
    # selection_args = dict(epochs=args.selection_epochs,
    #                         selection_method=args.uncertainty,
    #                         balance=args.balance,
    #                         greedy=args.submodular_greedy,
    #                         function=args.submodular
    #                         )
    # method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
    method = Submodular(dst_train, args, args.ipc, args.seed, epochs=args.selection_epochs, balance=not args.imbalance)
    subset = method.select()
    print(len(subset["indices"]))

    torch.save({'subset': subset}, os.path.join(args.save_path, checkpoint_name))

    


if __name__ == '__main__':
    main()
