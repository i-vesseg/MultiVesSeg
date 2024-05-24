import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

from models.encoders.iresnet import iresnet50

from torch.nn import Linear, Conv2d, PReLU
from utils.MyBatchNorm import MyBatchNorm2d, MySequential
def replace_batchnorm(model, DSBN):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module, DSBN)
        
        if isinstance(module, torch.nn.BatchNorm2d):
            model._modules[name] = MyBatchNorm2d(
                module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, domain_specific=DSBN
            )

    return model

def replace_sequential(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_sequential(module)
        
        if isinstance(module, torch.nn.Sequential):
            model._modules[name] = MySequential(module._modules)

    return model

from models.stylegan2.model import EqualLinear, PixelNorm

class fs_encoder_v2(nn.Module):
    def __init__(self, n_styles=18, arcface_model_path=None, residual=True, stride=(1, 1), DSBN=True):
        super(fs_encoder_v2, self).__init__()  

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(arcface_model_path))
        resnet50 = replace_batchnorm(resnet50, DSBN)
        resnet50 = replace_sequential(resnet50)

        resnet50_children = list(resnet50.children())

        # input conv layer
        self.conv = MySequential(
            Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            *resnet50_children[1:3]
        )
        
        # define layers
        self.block_1 = resnet50_children[3] # 15-18
        self.block_2 = resnet50_children[4] # 10-14
        self.block_3 = resnet50_children[5] # 5-9
        self.block_4 = resnet50_children[6] # 1-4
        
        self.residual = residual
        if self.residual:
            self.content_layers = nn.ModuleList([
                MySequential(
                    MyBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    MyBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.PReLU(num_parameters=256),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                    MyBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN)
                ),
                MySequential(
                    MyBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN)
                ),
                MySequential(
                    MyBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN)
                ),
                MySequential(
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                    MyBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, domain_specific=DSBN),
                )
            ])

        self.avg_pool = nn.AdaptiveAvgPool2d((3,3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

        self.pixel_norm = PixelNorm()
        self.label_embed = EqualLinear(2, 512*8) # TODO: pass n. of labels
        
        self.skiplayers = nn.ModuleList([
            DynamicSkipLayer(512, 512),
            DynamicSkipLayer(512, 512),
            DynamicSkipLayer(512, 256)
        ])
        
        self.residual = residual

    def forward(self, x, labels):
        is_swi = labels[:,1] > labels[:,0]
        is_tof = torch.logical_not(is_swi)
        
        latents = []
        features = []
        content = []
        
        x = self.conv(x, is_tof, is_swi)
        
        x = self.block_1(x, is_tof, is_swi)
        if self.residual:
            content += [self.content_layers[0](x, is_tof, is_swi)]
        features.append(self.avg_pool(x))
        
        x = self.block_2(x, is_tof, is_swi)
        if self.residual:
            content += [self.content_layers[1](x, is_tof, is_swi)]
        features.append(self.avg_pool(x))
        
        x = self.block_3(x, is_tof, is_swi)
        if self.residual:
            content += [self.content_layers[2](x, is_tof, is_swi)]
        features.append(self.avg_pool(x))
        
        x = self.block_4(x, is_tof, is_swi)
        if self.residual:
            content += [self.content_layers[3](x, is_tof, is_swi)]
        features.append(self.avg_pool(x))
        
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        
        labels = self.pixel_norm(self.label_embed(labels))
        labels = labels.reshape(len(out), out.shape[1]//2, out.shape[2]).repeat(1, 1, 2)
        out *= labels.reshape(out.shape)
        
        return (out, content) if self.residual else out
    
    def convertContent(self, content, init_state):
        new_fencs = []
        masks = []
        state = None
        for skiplayer, c in zip(self.skiplayers, content):
            mask, new_fenc, state = skiplayer(c, init_state if state is None else state)
            new_fencs += [new_fenc]
            masks += [1 - mask]
        
        residuals = [*new_fencs[::-1], init_state]
        masks = [*masks[::-1], 1]
        
        residuals = sum([
            [None]*5 + [(residuals[-1], masks[-1])],
            [None]*1 + [(residuals[-2], masks[-2])],
            [None]*1 + [(residuals[-3], masks[-3])],
            [None]*1 + [(residuals[-4], masks[-4])],
            [None]*5,
        ], start=[])
        return residuals
            
    
class DynamicSkipLayer(nn.Module):
    def __init__(self, hidden_nc, feature_nc, scale_factor=2):
        super(DynamicSkipLayer, self).__init__()  
        self.up = nn.Upsample(scale_factor=scale_factor)
        # Wh
        self.trans = nn.Conv2d(hidden_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        # Wr
        self.reset = nn.Conv2d(2*feature_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        # Wm
        self.mask = nn.Conv2d(2*feature_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        # WE
        self.update = nn.Conv2d(2*feature_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        # sigma
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fenc, s):
        # h^ = sigma(Wh * up(h))
        state = F.leaky_relu(self.trans(self.up(s)), 2e-1)
        # r = sigma(Wr * [h^, fE])
        reset_gate = self.sigmoid(self.reset(torch.cat((state,fenc),dim=1)))
        # h = rh^
        new_state = reset_gate * state
        # m = sigma(Wm * [h^, fE]) with sigma=None
        mask = self.mask(torch.cat((state,fenc),dim=1))
        # apply relu + tanh to set most of the elements to zeros
        mask = (F.relu(mask)).tanh()
        # fE^ = sigma(WE * [h, fE]) with sigma=None
        new_fenc = self.update(torch.cat((new_state,fenc),dim=1))
        # f = (1-m) * fG + m * fE^
        new_fenc *= mask#output = (1-mask) * fdec + mask * new_fenc
        return mask, new_fenc, new_state