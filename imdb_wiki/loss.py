from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss

class LAloss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LAloss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        iota_list = tau * np.log(cls_probs)

        self.iota_list = torch.cuda.FloatTensor(iota_list)

    def forward(self, x, target):
        #print(" x shape is {} taegt shape is {} iota is {}" .format(x.shape, target.shape, self.iota_list))
        output = x + self.iota_list

        return F.cross_entropy(output, target)


class Weight_CE(nn.Module):
    def __init__(self):
        super(Weight_CE, self).__init__()

    def forward(self, x, y):
        # shape : batch, groups, binary      
        index_eq = x == y
        #
        index_finder = torch.sum(index_eq, dim=-1)
        #
        index_finder = index_finder - 1
        #
        index_finder[index_finder<0] = 0
        #
        total_sum = torch.sum(- y * torch.log(x), dim = -1)
        #
        loss = torch.sum(index_finder*total_sum)      
        #loss = torch.sum( -y * torch.log(x))
        return loss



class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss

'''
    This is the forward but not the loss 
    parameters to be added :
        self.weighted
        self.loss_mse = torch.nn.MSELoss()
        self.loss_la = LAloss(self.cls_num_list, tau=1.0)
        self.sigma
        self.output_strategy
            # output_strategy the output
                #   1) max possiblity
                #   2) average
'''
'''
def forward(self, x, y, g):
    # x shape is N, W,H,C
    # y shape is N,1
    # g shape is N,1
    "output of model dim is 2G"
    y_hat = self.model(x)
    output_len = len(y_hat)
    #first G is cls
    g_hat = y_hat[: output_len/2]
    # add MSE
    
        #to be added into the model ini part
    
    self.loss_mse = torch.nn.MSELoss()
    self.loss_la = LAloss(self.cls_num_list, tau=1.0)
    if self.mode == 'train':
        #loss_la = LAloss(self.cls_num_list, tau=1.0)
        # compute the group cls loss
        loss_ce = self.loss_la(g_hat, g)
        y_hat_index = output_len/2 + g
        yhat = y_hat[y_hat_index]
        loss_mse = self.loss_mse(yhat, y)
        loss = loss_mse + self.sigma *loss_ce
        return yhat, g_hat, loss
    else:
        if self.output_strategy == 1:
            y_hat_index = output_len/2 + torch.argmax(g_hat, dim =1)
            yhat = y_hat[y_hat_index]
        elif self.output_strategy == 2:
            # wegihted
            yhat = torch.sum(self.weighted * y_hat[output_len/2:])
            pass
        else:
            assert "no output strategy defined"
        return yhat, g_hat
'''
