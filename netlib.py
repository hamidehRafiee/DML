import torch, os, numpy as np

import torch.nn as nn
import pretrainedmodels as ptm

import pretrainedmodels.utils as utils
import torchvision.models as models

import googlenet
import torch.nn.functional as F


"""============================================================="""
def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.

    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()



"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == 'googlenet':
        network =  GoogLeNet(opt)
    elif opt.arch == 'resnet50':
        network =  ResNet50(opt)

    elif opt.arch == 'modify_resnet50':
        network =  NetworkSuperClass(opt)

    elif opt.arch == 'backbone':
        network =  ResNet50_2lastLCE(opt)

    elif opt.arch == "attention":
        network = AttentionResNet50_2lastL(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))
    return network




"""=================================================================================================================================="""
class GoogLeNet(nn.Module):
    """
    Container for GoogLeNet s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt):
        """
        Args:
            opt: argparse.Namespace, contains all training-specific parameters.
        Returns:
            Nothing!
        """
        super(GoogLeNet, self).__init__()

        self.pars = opt

        self.model = googlenet.googlenet(num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else False)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        rename_attr(self.model, 'fc', 'last_linear')

        self.layer_blocks = nn.ModuleList([self.model.inception3a, self.model.inception3b, self.model.maxpool3,
                                           self.model.inception4a, self.model.inception4b, self.model.inception4c,
                                           self.model.inception4d, self.model.inception4e, self.model.maxpool4,
                                           self.model.inception5a, self.model.inception5b, self.model.avgpool])

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)


    def forward(self, x):
        ### Initial Conv Layers
        x = self.model.conv3(self.model.conv2(self.model.maxpool1(self.model.conv1(x))))
        x = self.model.maxpool2(x)

        ### Inception Blocks
        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = x.view(x.size(0), -1)
        x = self.model.dropout(x)

        mod_x = self.model.last_linear(x)

        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)



"""============================================================="""
class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, 1024)
        self.model.last_linear1 = torch.nn.Linear(1024, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        print(x.size())


        mod_x0 = self.model.last_linear(x)
        mod_x1 = self.model.last_linear1(mod_x0)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return torch.nn.functional.normalize(mod_x1, dim=-1)




class NetworkSuperClass(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass, self).__init__()
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        self.input_space, self.input_range, self.mean, self.std = self.model.input_space, self.model.input_range, self.model.mean, self.model.std

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)

        self.shared_norm = opt.shared_norm

        if self.shared_norm:
            self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        else:
            print("use 2 last layer ...")

            self.model.last_linear_class = torch.nn.Linear(self.model.last_linear.in_features, opt.classembed)
            self.model.last_linear_res = torch.nn.Linear(self.model.last_linear.in_features, opt.intraclassembed)
            self.model.last_linear = None

    def forward(self, x, feat='embed'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        print(x.size())

        x_class = self.model.last_linear_class(x)
        x_class = torch.nn.functional.normalize(x_class, dim=-1)

        x_res = self.model.last_linear_res(x)
        x_res = torch.nn.functional.normalize(x_res, dim=-1)

        x = torch.cat([x_class, x_res], dim=1)

        return x




class ResNet50_2lastLCE(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt):
        super(ResNet50_2lastLCE, self).__init__()

        self.pars = opt

        print(f'Getting pretrained weights  resnet50 {opt.embed_dim}...')
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        print('Done.')

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, 256)
        self.model.last_1 = torch.nn.Linear(256, 128 )#opt.embed_dim
        self.model.last_2 = torch.nn.Linear(256, 128)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        # print(x.size())
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())

        mod_x = self.model.last_linear(x)  # 256

        back_bone = torch.nn.functional.normalize(mod_x, dim=-1)

        last_l1 = self.model.last_1(mod_x)
        last_l1 = torch.nn.functional.normalize(last_l1, dim=-1)

        last_l2 = self.model.last_2(mod_x)
        last_l2 = torch.nn.functional.normalize(last_l2, dim=-1)

        return back_bone, last_l1, last_l2





class AttentionResNet50_2lastL(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning with an attention mechanism.
    """
    def __init__(self, opt):
        super(AttentionResNet50_2lastL, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print(f'Getting pretrained weights  resnet50 {opt.embed_dim}...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        self.model.last_1 = torch.nn.Linear(opt.embed_dim, opt.embed_dim)
        self.model.last_2 = torch.nn.Linear(opt.embed_dim, opt.embed_dim)

        torch.nn.init.kaiming_normal_(self.model.last_linear.weight, mode='fan_out')
        torch.nn.init.constant_(self.model.last_linear.bias, 0)

        torch.nn.init.kaiming_normal_(self.model.last_1.weight, mode='fan_out')
        torch.nn.init.constant_(self.model.last_1.bias, 0)

        torch.nn.init.kaiming_normal_(self.model.last_2.weight, mode='fan_out')
        torch.nn.init.constant_(self.model.last_2.bias, 0)

        # Attention mechanism added to last_l2
        self.attention1 = torch.nn.Linear(opt.embed_dim, 1)
        self.attention2 = torch.nn.Linear(opt.embed_dim, 1)
        # self.pool_base = torch.nn.AdaptiveAvgPool2d(1)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)    # 2048

        mod_x = self.model.last_linear(x)  # 256

        back_bone = torch.nn.functional.normalize(mod_x, dim=-1)
        # print(back_bone.size())  #torch.Size([112, 256])

        last_l1 = self.model.last_1(mod_x)
        # Attention mechanism applied to last_l1
        attention_weights1 = F.softmax(self.attention2(last_l1), dim=0)
        # print(attention_weights1.size())  #torch.Size([112, 1])
        attended_last_l1 = torch.mul(attention_weights1, last_l1)
        # print(attended_last_l1.size())
        attended_last_l1 = torch.nn.functional.normalize(attended_last_l1, dim=-1)
        # print(attended_last_l1.size())

        last_l2 = self.model.last_2(mod_x)
        # last_l2 = torch.nn.functional.normalize(last_l2, dim=-1)

        # Attention mechanism applied to last_l2
        attention_weights2 = F.softmax(self.attention2(last_l2), dim=0)
        attended_last_l2 = torch.mul(attention_weights2, last_l2)
        attended_last_l2 = torch.nn.functional.normalize(attended_last_l2, dim=-1)

        return back_bone, attended_last_l1, attended_last_l2

