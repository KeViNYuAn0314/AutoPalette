
import numpy as np
import torch
# from ..nets.nets_utils import MyDataParallel
from copy import deepcopy
import torch.nn as nn

import sys
sys.path.append('../')

from utils.utils_baseline import get_network


def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res


class StochasticGreedy(object):
    def __init__(self, args, index, budget:int, already_selected=[], epsilon: float=0.9):
        # super(StochasticGreedy).__init__(args, index, budget, already_selected)
        self.args = args
        self.index = index
        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected
        self.epsilon = epsilon

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        sample_size = max(round(-np.log(self.epsilon) * self.n / self.budget), 1)

        greedy_gain = np.zeros(len(self.index))
        all_idx = np.arange(self.n)
        for i in range(sum(selected), self.budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, self.budget))

            # Uniformly select a subset from unselected samples with size sample_size
            subset = np.random.choice(all_idx[~selected], replace=False, size=min(sample_size, self.n - i))

            if subset.__len__() == 0:
                break

            greedy_gain[subset] = gain_function(subset, selected, **kwargs)
            current_selection = greedy_gain[subset].argmax()
            selected[subset[current_selection]] = True
            greedy_gain[subset[current_selection]] = -np.inf
            if update_state is not None:
                update_state(np.array([subset[current_selection]]), selected, **kwargs)
        return self.index[selected]


class GraphCut(object):
    def __init__(self, index, lam: float = 1., similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
        # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
        # calculate similarities incrementally at a later time if required.
        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]
        self.lam = lam

        if similarity_matrix is not None:
            self.sim_matrix_cols_sum = np.sum(self.similarity_matrix, axis=0)
        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.sim_matrix_cols_sum = np.zeros(self.n, dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.sim_matrix_cols_sum[not_calculated] = np.sum(self.sim_matrix[:, not_calculated], axis=0)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):

        gain = -2. * np.sum(self.similarity_kernel(selected, idx_gain), axis=0) + self.lam * self.sim_matrix_cols_sum[idx_gain]

        return gain

    def update_state(self, new_selection, total_selected, **kwargs):
        pass
    
    
class CoresetMethod(object):
    def __init__(self, dst_train, args, ipc=10, random_seed=None, **kwargs):
        self.dst_train = dst_train
        # self.num_classes = len(dst_train.classes)
        self.num_classes = len(args.class_names)
        self.ipc = ipc
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        # self.coreset_size = round(self.n_train * fraction)
        self.coreset_size = self.args.num_classes * ipc

    def select(self, **kwargs):
        return    



class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args, ipc=10, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None,
                 **kwargs):
        super().__init__(dst_train, args, ipc, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = self.args.num_classes * ipc
        self.specific_model = specific_model

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or \
                    'num_classes' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes.')
            if dst_pretrain_dict['im_size'][0] != args.im_size[0] or dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['num_classes'] != args.num_classes:
                self.num_classes_mismatch()
        self.args = args
        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        # if torchvision_pretrain:
        #     # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current
        #     # datasets to 224*224.
        #     if args.im_size[0] != 224 or args.im_size[1] != 224:
        #         self.dst_train = deepcopy(dst_train)
        #         self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
        #         if self.if_dst_pretrain:
        #             self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
        #             self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
        #                 [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = dst_test

    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()

        print('\n=> Training Epoch #%d' % epoch)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args.workers, pin_memory=True)

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        # self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
        #     self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
        #     pretrained=self.torchvision_pretrain,
        #     im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        self.model = get_network(self.args.model, channel=self.args.channel, num_classes=self.args.num_classes, im_size=self.args.im_size, dist=False).to(self.args.device)

        # if self.args.device == "cpu":
        #     print("Using CPU.")
        # elif self.args.gpu is not None:
        #     torch.cuda.set_device(self.args.gpu[0])
        #     # self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        #     self.model = get_network(self.model, channel=self.args.channel, num_classes=self.args.num_classes, im_size=self.args.im_size, )
        # elif torch.cuda.device_count() > 1:
        #     self.model = nets.nets_utils.MyDataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                   momentum=self.args.selection_momentum,
                                                   weight_decay=self.args.selection_weight_decay,
                                                   nesterov=self.args.selection_nesterov)
        elif self.args.selection_optimizer == "Adam":
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                    weight_decay=self.args.selection_weight_decay)
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                       lr=self.args.selection_lr,
                                                                       momentum=self.args.selection_momentum,
                                                                       weight_decay=self.args.selection_weight_decay,
                                                                       nesterov=self.args.selection_nesterov)

        self.before_run()

        for epoch in range(self.epochs):
            list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                 self.n_pretrain_size, replace=False)
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            if self.dst_test is not None and self.args.selection_test_interval > 0 and (
                    epoch + 1) % self.args.selection_test_interval == 0:
                self.test(epoch)
            self.after_epoch()

        return self.finish_run()

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction == 1. else
                                                  torch.utils.data.Subset(self.dst_test, np.random.choice(
                                                      np.arange(len(self.dst_test)),
                                                      round(len(self.dst_test) * self.args.selection_test_fraction),
                                                      replace=False)),
                                                  batch_size=self.args.selection_batch, shuffle=False,
                                                  num_workers=self.args.workers, pin_memory=True)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            output = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
                    epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                        self.args.selection_batch) + 1, loss.item(),
                    100. * correct / total))

        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result


class Submodular(EarlyTrain):
    def __init__(self, dst_train, args, ipc=10, random_seed=None, epochs=200, specific_model=None, balance=False,
                 greedy="ApproximateLazyGreedy", metric="cossim", **kwargs):
        super(Submodular, self).__init__(dst_train, args, ipc, random_seed, epochs, specific_model, **kwargs)
        self._greedy = greedy
        self._metric = metric

        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))
            
    def calc_feature(self, index=None):
        self.model.eval()
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers
        )
        sample_num = self.n_train if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features
        features = []
        
        for i, (input, targets) in enumerate(batch_loader):
            batch_num = targets.shape[0]
            feature = self.model.embed(input.to(self.args.device))
            feature = feature.view(batch_num, -1)
            features.append(feature.detach().cpu().numpy())
        features = np.concatenate(features, axis=0)
        return features
            
        

    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)
        sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features

        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs.requires_grad_(True),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1,
                                        self.embedding_dim).repeat(1, self.args.num_classes, 1) *\
                                        bias_parameters_grads.view(batch_num, self.args.num_classes,
                                        1).repeat(1, 1, self.embedding_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())

        # print('')
        # print(len(gradients))
        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def finish_run(self):
        # if isinstance(self.model, MyDataParallel):
        #     self.model = self.model.module

        # Turn on the embedding recorder and the no_grad flag
        with self.model.embedding_recorder:
            self.model.no_grad = True
            self.train_indx = np.arange(self.n_train)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[self.dst_train.targets == c]
                    # Calculate gradients into a matrix
                    # gradients = self.calc_gradient(index=c_indx)
                    gradients = self.calc_feature(index=c_indx) ####### note this is to calculate feature experiments
                    # Instantiate a submodular function
                    submod_function = GraphCut(index=c_indx,
                                        similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))
                    submod_optimizer = StochasticGreedy(args=self.args,
                                        index=c_indx, budget=self.ipc, already_selected=[])

                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                                 update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, c_selection_result)
            else:
                # Calculate gradients into a matrix
                # gradients = self.calc_gradient()
                gradients = self.calc_feature(index=c_indx) ####### note this is to calculate feature experiments
                # Instantiate a submodular function
                submod_function = GraphCut(index=self.train_indx,
                                            similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = StochasticGreedy(args=self.args, index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)

            self.model.no_grad = False
        return {"indices": selection_result}

    def select(self, ):
        selection_result = self.run()
        return selection_result

    def select_n_model(self,):
        selection_result = self.run()
        return selection_result, self.model
