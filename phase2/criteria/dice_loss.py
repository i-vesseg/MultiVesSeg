import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    #def forward(self, y_hat, y):
        #THIS WAS ALREADY A COMMENT#y_hat, y = (y_hat + 1) / 2, (y + 1) / 2
        #y = (y + 1) / 2
        #tp = torch.sum(y_hat * y, dim=(0,2,3))
        #fp = torch.sum(y_hat*(1-y),dim=(0,2,3))
        #fn = torch.sum((1-y_hat)*y,dim=(0,2,3))
        #nominator = 2*tp + 1e-05
        #denominator = 2*tp + fp + fn + 1e-05
        #dice_score = -(nominator / (denominator+1e-8))[1:].mean()
        #return dice_score
    def forward(self, x, y):
        tp = torch.sum(x * y, dim=(0,2,3))
        fp = torch.sum(x*(1-y),dim=(0,2,3))
        fn = torch.sum((1-x)*y,dim=(0,2,3))
        nominator = 2*tp + 1e-05
        denominator = 2*tp + fp + fn + 1e-05
        dice_score = -(nominator / (denominator+1e-8))#[1:]
        label_nc = len(dice_score)
        weights = [1/label_nc] * (label_nc)
        dice_score = torch.mean(torch.stack([w*dice_score[i] for w,i in zip(weights, range(label_nc))]))
        """dice_score = torch.mean(torch.stack([
            .33*dice_score[0],#background#10#20#33#50
            .33*dice_score[1],#brain#10#20#33#50
            .34*dice_score[2],#vessels#80#60#34#00
        ]))"""
        return dice_score
