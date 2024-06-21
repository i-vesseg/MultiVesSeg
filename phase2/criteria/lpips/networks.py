from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from criteria.lpips.utils import normalize_activation
from options import Options


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    else:
        raise NotImplementedError('choose net_type from [alex].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        #self.layers = models.alexnet(True).features
        alex_model = models.alexnet(pretrained=False)
        alex_model.load_state_dict(
            torch.load(Options().opts.alex_pretr_model, map_location=None if torch.cuda.is_available() else torch.device('cpu')
            ), strict=False
        )
        self.layers = alex_model.features
        
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)