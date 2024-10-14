import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.utils
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
sys.path.append("../")
# from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule

from utils.cfg import CFG as cfg
import warnings
import yaml

from color_networks.color_cnn import ColorCNN
from color_networks.color_cnn_plus import ColorCNNPlus, PixelSimLoss
from color_networks.palette_cnn import PaletteCNN
from color_networks.layercam import LayerCAM
from PIL import Image

from utils.kmean_init import kmeans_sample
from utils.median_cut import get_data
import math

warnings.filterwarnings("ignore", category=DeprecationWarning)



def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
 
 
 
def get_index_map(image, num_colors):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    to_tensor = transforms.ToTensor()
    image = copy.deepcopy(image.detach())
    B, C, H, W = image.shape
    for ch in range(3):
        image[:, ch] = image[:, ch]  * std[ch] + mean[ch]
        
    image = image.clamp(0.0, 1.0)
    to_pil = transforms.Compose([transforms.ToPILImage()])
    quant_image = [to_pil(img).quantize(colors=num_colors, method=0).convert('RGB') for img in image]
    index_maps = []
    for img in quant_image:
        # H, W, C = np.array(img).shape
        palette, index_map = np.unique(np.array(img).reshape([H * W, C]), axis=0, return_inverse=True)
        index_map = Image.fromarray(index_map.reshape(H, W).astype(np.uint8))
        index_map = (to_tensor(index_map) * 255).round().long()
        index_maps.append(index_map)
    index_maps = torch.stack(index_maps, dim=0)
    return index_maps


def main(args):
    manual_seed()

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    print('current cuda device is ', args.device)
    args.dsa_param = ParamDiffAug()
    
    if args.dsa:
        args.dc_aug_param = None

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.im_size = im_size
    
    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    
    wandb.init(sync_tensorboard=False,
               project=args.project,
               job_type="CleanRepo",
               config=args,
               )
    
    
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])
        
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans
        
    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    print(f'there are {torch.cuda.device_count()} gpus')
    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    ''' Calculate budget'''
    args.ipc = args.ipc * (2**(8-math.ceil(math.log2(args.num_colors))))

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    ''' initialize the synthetic data '''
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    
    
    # expert_dir = os.path.join(args.buffer_path, args.dataset)
    expert_dir = os.path.join(args.buffer_path, 'CIFAR10')
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]

        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)
    
    
    
    if args.pix_init == 'graphcut' or args.pix_init == 'medcut_graphcut':
        # assign labels
        subset_indices = torch.load(args.subset_ckpt)['subset']['indices']
        for i, idx in enumerate(subset_indices):
            label_syn[i] = dst_train[idx][1]

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
    elif args.pix_init == 'graphcut' or args.pix_init == 'medcut_graphcut':
        if args.parall_eva==False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        label_expert_files = expert_files
        subset_indices = torch.load(args.subset_ckpt)['subset']['indices']
        print('there are ', len(subset_indices), ' subset')
        for i, idx in enumerate(subset_indices):
            # image_syn.data[i] = dst_train[idx][0].detach().data
            image_syn.data[i] = images_all[idx].detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    optimizer_lr.zero_grad()
    
    
    if args.color_model == 'color_cnn':
        quant_model = ColorCNN('None', args.num_colors, in_channel=3).to(args.device)
    elif args.color_model == 'color_cnn_plus':
        quant_model = ColorCNNPlus(args.num_colors, in_channel=3, ).to(args.device)
    elif args.color_model == 'color_palette':
        quant_model = PaletteCNN(args.num_colors, in_channel=3, ).to(args.device)
        
    optimizer_quant = torch.optim.SGD(quant_model.color_mask.parameters(),
                          lr=args.lr_quant, momentum=args.momentum_quant, weight_decay=args.weight_decay_quant)
    optimizer_quant.zero_grad()
    
    if args.distributed:
        quant_model = torch.nn.DataParallel(quant_model)
        
        
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    criterion = SoftCrossEntropy
    
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    
    if args.pix_init == 'medcut_graphcut':
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        batch_size = 256
        for i in range(len(label_expert_files)):
            Temp_Buffer = torch.load(label_expert_files[i])
            for j in Temp_Buffer:
                temp_logits = None
                for select_times in range((len(image_syn)+batch_size-1)//batch_size):
                    current_data_batch = image_syn[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                    Temp_params = j[args.Label_Model_Timestamp]
                    Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
                    if args.distributed:
                        Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    Initialized_Labels = Temp_net(current_data_batch, flat_param=Initialize_Labels_params)
                    if temp_logits == None:
                        temp_logits = Initialized_Labels.detach()
                    else:
                        temp_logits = torch.cat((temp_logits, Initialized_Labels.detach()),0)
                logits.append(temp_logits.detach().cpu())
        logits_tensor = torch.stack(logits)
        true_labels = label_syn.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image
        del Temp_net

    elif args.pix_init == "real" or args.pix_init == 'noise' or args.pix_init == 'graphcut' or args.pix_init == 'medcut_kmeans':
        Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        Temp_params = buffer[0][-1]
        Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
        if args.distributed:
            Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        Initialized_Labels = Temp_net(image_syn, flat_param=Initialize_Labels_params)
        del Temp_net
    # else:
    #     Initialized_Labels = label_syn.clone().to(torch.float64)

    acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), label_syn.cpu().data.numpy()))
    print('InitialAcc:{}'.format(acc/len(label_syn)))

    label_syn = copy.deepcopy(Initialized_Labels.detach()).to(args.device).requires_grad_(True)
    label_syn.requires_grad=True
    label_syn = label_syn.to(args.device)
    

    optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)
    
    
    # palette nn warmup
    if args.use_warmup:
        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
        entropy = lambda x: (-x * torch.log(x + 1e-16)).sum(dim=1)
        for i in range(args.warmup_step):
            loss_avg, num_exp = 0, 0
            for i_batch, datum in enumerate(train_loader):
                img = datum[0].float().to(device)
                B, _, H, W = img.shape
                output, prob, color_palette = quant_model(img, training=True)

                index_map = get_index_map(img, args.num_colors).to(args.device)
                M = torch.zeros_like(prob).scatter(1, index_map, 1)
                pixel_loss = PixelSimLoss(sample_ratio=0.3)
                pixsim_loss = pixel_loss(prob, M)

                info_loss = -entropy(prob.mean(dim=[2, 3])).mean()
                color_appear_loss = -prob.view([B, -1, H * W]).max(dim=2)[0].mean()
                color_loss = args.colormax_ratio * color_appear_loss + args.info_ratio * info_loss + args.pixsim_ratio * pixsim_loss

                optimizer_quant.zero_grad()
                color_loss.backward()
                optimizer_quant.step()
                n_b = datum[1].shape[0]
                loss_avg += color_loss.item()*n_b
                num_exp += n_b
            print('warmup loss is ', loss_avg / num_exp)
    
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                print('DSA augmentation strategy: \n', args.dsa_strategy)
                print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                accs_test = []
                accs_train = []

                for it_eval in range(args.num_eval):
                    if args.parall_eva==False:
                        device = torch.device("cuda:0")
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
                    else:
                        device = args.device
                        net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True).to(device) # get a random model

                    eval_labs = label_syn.detach().to(device)
                    # layer_cams = get_activation_map(act_model, layer_ids, copy.deepcopy(image_syn.detach()), im_size)
                    with torch.no_grad():
                        image_save = image_syn.to(device)
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()).to(device), copy.deepcopy(eval_labs.detach()).to(device) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, copy.deepcopy(net_eval).to(device), image_syn_eval.to(device), label_syn_eval.to(device), testloader, args, texture=False, train_criterion=criterion)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)

                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                save_dir = os.path.join(".", "logged_files", args.dataset, str(args.ipc), args.model, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir,'Normal'))
                    
                torch.save(image_syn.cpu(), os.path.join(save_dir, 'Normal',"images_{}.pt".format(it)))
                torch.save(quant_model, os.path.join(save_dir, 'Normal', "model_{}".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_{}.pt".format(it)))

                # if save_this_it:
                #     torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_best.pt".format(it)))
                #     torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_best.pt".format(it)))
                #     torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc <= 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        # upsampled = image_save[:args.ipc].to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()
                        torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_zca_{}.pt".format(it)))
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        # if args.distributed:
        #     net = torch.nn.DataParallel(net)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        optimizer_y.zero_grad()
        optimizer_quant.zero_grad()

        ''' update synthetic data '''
        if 'BN' not in args.model: # for ConvNet
            loss = torch.tensor(0.0).to(args.device)
            
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                if args.batch_syn < args.ipc:
                    indices = torch.randperm(args.ipc)[:args.batch_syn]
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))[indices]
                else:
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                syn_images, prob, color_palette = quant_model(img_syn, training=True)

                if args.dsa:
                    img_real = DiffAugment(img_real, args.dsa_strategy, param=args.dsa_param)
                    syn_images = DiffAugment(syn_images, args.dsa_strategy, param=args.dsa_param)

                output_real = embed(img_real).detach()
                output_syn = embed(syn_images)

                # loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                
                loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                loss = loss / num_classes
                loss.backward()
                
                loss_avg += loss.item()
                
                
        else: # for ConvNetBN
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)

            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)


        # compute palette network loss
        sub_indices = torch.randperm(len(image_syn))[:64]
        syn_images, prob, color_palette = quant_model(copy.deepcopy(image_syn.detach())[sub_indices], training=True)
        entropy = lambda x: (-x * torch.log(x + 1e-16)).sum(dim=1)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        B, _, H, W = syn_images.shape
        
        color_appear_loss = -prob.view([B, -1, H * W]).max(dim=2)[0].mean()
        conf_loss = entropy(prob).mean()
        info_loss = -entropy(prob.mean(dim=[2, 3])).mean()

        index_map = get_index_map(copy.deepcopy(image_syn.detach())[sub_indices], args.num_colors).to(args.device)
        M = torch.zeros_like(prob).scatter(1, index_map, 1)
        pixel_loss = PixelSimLoss(sample_ratio=0.3)
        pixsim_loss = pixel_loss(prob, M)
        color_loss = args.colormax_ratio * color_appear_loss + args.conf_ratio * conf_loss + args.info_ratio * info_loss + args.pixsim_ratio * pixsim_loss

        torch.cuda.empty_cache()
        
        optimizer_y.step()
        optimizer_img.step()
        optimizer_lr.step()
        color_loss.backward()
        optimizer_quant.step()
        
        wandb.log({"Grand_Loss": loss_avg,
                    'conf_loss': args.conf_ratio * conf_loss.detach().cpu(), 
                    'info_loss': args.info_ratio * info_loss.detach().cpu(), 
                    'colormax_loss': args.colormax_ratio * color_appear_loss.detach().cpu(), 
                    'imitate_loss': args.pixsim_ratio * pixsim_loss.detach().cpu(),
                    'palette_loss': (color_loss + loss).detach().cpu(),
                    })

        if it%10 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

        # if it == args.Iteration: # only record the final results
        #     data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
        #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    # print('\n==================== Final Results ====================\n')
    # for key in model_eval_pool:
    #     accs = accs_all_exps[key]
    #     print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument('--num_colors', type=int, default=64)
    parser.add_argument('--lr_quant', type=float, default=0.1)
    parser.add_argument('--momentum_quant', type=float, default=0.5)
    parser.add_argument('--weight_decay_quant', type=float, default=5e-4)
    parser.add_argument('--pixsim_ratio', type=float, default=3.0, help='similarity towards the KMeans result')
    parser.add_argument('--conf_ratio', type=float, default=1.0,
                        help='softmax more like argmax (one-hot), reduce entropy of per-pixel color distribution')
    parser.add_argument('--info_ratio', type=float, default=1.0,
                        help='even distribution among all colors, increase entropy of entire-image color distribution')
    parser.add_argument('--colormax_ratio', type=float, default=1.0, help='ensure all colors present')

    parser.add_argument('--patch_ratio', type=float, default=1.0,)
    parser.add_argument('--background_mask', action='store_true')
    parser.add_argument('--color_model', type=str, default='color_cnn')
    parser.add_argument('--diverse_loss', action='store_true')
    parser.add_argument('--diverse_type', type=str, default='nuclear')
    parser.add_argument('--diverse_loss_ratio', type=float, default=0.1)
    
    parser.add_argument('--subset_ckpt', type=str, help='checkpoint for subset selection')
    parser.add_argument('--use_warmup', action='store_true')
    parser.add_argument('--warmup_step', type=int, default=2)
    
    
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    main(args)

