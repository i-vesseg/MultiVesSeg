import torch
from torch import nn
from options import Options

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
    
    def forward(self, prediction, target):
        prediction = torch.log(prediction + 1e-6)
        loss = prediction * target
        label_nc = loss.shape[1]
        if Options().opts.ce_weights is None:
            weights = [1/label_nc] * (label_nc)
        else:
            weights = Options().opts.ce_weights
        loss = torch.sum(torch.stack([w*loss[:,i] for w,i in zip(weights, range(label_nc))]), dim=0)
        return -1*torch.mean(loss)
