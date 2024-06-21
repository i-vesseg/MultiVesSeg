import torch
from torch import nn
from options import Options

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y):
        tp = torch.sum(x * y, dim=(0,2,3))
        fp = torch.sum(x*(1-y),dim=(0,2,3))
        fn = torch.sum((1-x)*y,dim=(0,2,3))
        nominator = 2*tp + 1e-05
        denominator = 2*tp + fp + fn + 1e-05
        dice_score = -(nominator / (denominator+1e-8))
        label_nc = len(dice_score)
        if Options().opts.dice_weights is None:
            weights = [1/label_nc] * (label_nc)
        else:
            weights = Options().opts.dice_weights
        dice_score = torch.mean(torch.stack([w*dice_score[i] for w,i in zip(weights, range(label_nc))]))
        return dice_score
