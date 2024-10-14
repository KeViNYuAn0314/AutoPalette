import torch.nn as nn
import torch.nn.functional as F
import torch

def find_layer(arch, target_layer_name):
    """Find target layer to calculate CAM.

        : Args:
            - **arch - **: Self-defined architecture.
            - **target_layer_name - ** (str): Name of target class.

        : Return:
            - **target_layer - **: Found layer. This layer will be hooked to get forward/backward pass information.
    """

    if target_layer_name.split('_')[0] not in arch._modules.keys():
        raise Exception("Invalid target layer name.")
    target_layer = arch.features[int(target_layer_name.split('_')[1])]
    return target_layer


class BaseCAM(object):
    """ Base class for Class activation mapping.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """

    def __init__(self, model_dict):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        
        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        if torch.cuda.is_available():
          self.model_arch.cuda()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value'] = grad_output[0].cuda()
            else:
              self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
              self.activations['value'] = output.cuda()
            else:
              self.activations['value'] = output
            return None

        
        self.target_layer = find_layer(self.model_arch, layer_name)

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
    
    
    
class LayerCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        #logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class = predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()
        
        one_hot_output = torch.FloatTensor(len(input), logit.size()[-1]).zero_()
        # one_hot_output[0][predicted_class] = 1
        for i in range(len(input)): 
            one_hot_output[i][predicted_class[i]] = 1 
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        self.model_arch.zero_grad()
        # Backward pass with specified target
        # for p in self.model_arch.parameters():
        #     print('require grad ', p.requires_grad)
        logit.backward(gradient=one_hot_output, retain_graph=True)
        # logit.backward(gradient=one_hot_output, retain_graph=False)
        activations = self.activations['value'].clone().detach()
        gradients = self.gradients['value'].clone().detach()
        b, k, u, v = activations.size()
        
        with torch.no_grad():
            activation_maps = activations * F.relu(gradients)
            cam = torch.sum(activation_maps, dim=1).unsqueeze(0)    
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)      
            cam_min, cam_max = cam.min(), cam.max()
            norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data

        return norm_cam

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)