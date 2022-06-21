import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class LAloss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LAloss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        iota_list = tau * np.log(cls_probs)

        self.iota_list = torch.cuda.FloatTensor(iota_list)

    def forward(self, x, target):
        output = x + self.iota_list

        return F.cross_entropy(output, target)



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
