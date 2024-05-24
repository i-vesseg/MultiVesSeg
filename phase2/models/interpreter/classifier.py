# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn

import gc

SQUEEZE = 4

class ListedLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        super(ListedLinear, self).__init__()
        assert out_features is not None, "Always specify the number of output features"
        self.in_features = in_features
        self.out_features = out_features
        
        if self.in_features is None:
            self.in_features = [512]*10 + [256]*2 + [128]*2 + [64]*2
        
        self.linear_layers = nn.Sequential(*[
            nn.Linear(self.in_features[i], self.out_features) for i in range(len(self.in_features))
        ])

    def forward(self, X):
        N = max([x.shape[1] for x in X])
        B = X[0].shape[0]
        output =  torch.zeros([B, N, N, self.out_features], dtype=X[0].dtype, device=X[0].device)
        for x, layer in zip(X, self.linear_layers):
            tmp_out = layer(x.reshape(-1, x.shape[-1]))
            output += torch.nn.functional.interpolate(
                  tmp_out.reshape([*x.shape[:-1], self.out_features]).permute(0, 3, 1, 2),
                  size = [N,N],
                  mode="bilinear", align_corners=True#nearest
            ).permute(0, 2, 3, 1)
        
        return output.reshape(-1, self.out_features)


class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        
        if numpy_class < 32:
            self.linear_layers = nn.Sequential(
                ListedLinear(None, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                nn.Softmax(dim=1)
            )
        else:
            self.linear_layers = nn.Sequential(
                ListedLinear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                nn.Softmax(dim=1)
            )
        
        self.dim=dim

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
    def forward(self, X, mask, detach=False, is_semi=None):
        if is_semi is None:
            return self.linear_layers([
                x[mask].permute(0, 2, 3, 1).detach() if detach else x[mask].permute(0, 2, 3, 1) for x in X
            ]).type(X[0].dtype)
        else:
            if detach:
                X = [torch.where(is_semi, x.permute(1, 2, 3, 0), x.detach().permute(1, 2, 3, 0)).permute(3, 0, 1, 2) for x in X]
            else:
                X = [torch.where(is_semi, x.detach().permute(1, 2, 3, 0), x.permute(1, 2, 3, 0)).permute(3, 0, 1, 2) for x in X]
            return self.linear_layers([x[mask].permute(0, 2, 3, 1) for x in X]).type(X[0].dtype)