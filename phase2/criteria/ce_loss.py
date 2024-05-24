import torch
from torch import nn

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    #def forward(self, y_hat, y):
        #y = (y + 1) / 2
        #return -1 * torch.mean(torch.sum(torch.log(y_hat + 1e-6) * y, dim=1))
    
    def forward(self, prediction, target):
        prediction = torch.log(prediction + 1e-6)
        loss = prediction * target
        label_nc = loss.shape[1]
        weights = [(1-0.80)/(label_nc-1)] * (label_nc-1) + [0.80]
        loss = torch.sum(torch.stack([w*loss[:,i] for w,i in zip(weights, range(label_nc))]), dim=0)
        """loss = torch.sum(torch.stack([
            .10*loss[:,0],#background
            .10*loss[:,1],#brain
            .80*loss[:,2],#vessels
        ]), dim=0)"""
        return -1*torch.mean(loss)
