import torch
from torch import nn
from models.encoders.iresnet import IBasicBlock

class MySequential(nn.Sequential):
    def forward(self, input, is_tof, is_swi):
        for module in self._modules.values():
            if type(module) == MyBatchNorm2d or type(module) == IBasicBlock:
                input = module(input, is_tof, is_swi)
            else:
                input = module(input)
        return input
    
class MyBatchNorm2d(nn.Module):
    def __init__(
        self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, domain_specific=True
    ):
        super(MyBatchNorm2d, self).__init__()
        self.domain_specific = domain_specific
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(2)
        ])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
    
    def domain_specific_forward(self, x, is_tof, is_swi):
        result = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        result[is_tof] = self.bns[0](x[is_tof])
        result[is_swi] = self.bns[1](x[is_swi])
        return result
    
    def forward(self, x, is_tof, is_swi):
        if self.domain_specific:
            return self.domain_specific_forward(x, is_tof, is_swi)
        else:
            return self.bns[0](x)
    
class MyBatchNorm1d(nn.Module):
    def __init__(
        self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, domain_specific=True
    ):
        super(MyBatchNorm1d, self).__init__()
        self.domain_specific = domain_specific
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for _ in range(2)
        ])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

    def domain_specific_forward(self, x, is_tof, is_swi):
        result = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        result[is_tof] = self.bns[0](x[is_tof])
        result[is_swi] = self.bns[1](x[is_swi])
        return result
    
    def forward(self, x, is_tof, is_swi):
        if self.domain_specific:
            return domain_specific_forward(x, is_tof, is_swi)
        else:
            return self.bns[0](x)