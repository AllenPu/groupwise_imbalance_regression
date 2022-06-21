import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from loss import LAloss


class ResNet_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_regression, self).__init__()
        self.groups = args.groups
        self.model = torchvision.models.resnet18(pretrained=False)
        output_dim = args.groups*2
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            #nn.Linear(fc_inputs, 1024),
            #nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(fc_inputs, output_dim)
        )
        ###### LOSS
        #self.loss_mse = torch.nn.MSELoss()
        #self.loss_la = LAloss(self.cls_num_list, tau=1.0)
        ######
        #self.mode = args.mode
        self.sigma = args.sigma
        #self.output_strategy = args.outpt_strategy
        #self.weighted = args.weighted
        
    # g is the same shape of y
    def forward(self, x, g, mode):
        #"output of model dim is 2G"
        y_hat = self.model(x)
        output_len = y_hat.shape[1]
        #first G is cls
        g_hat = y_hat[: , : int(output_len/2)]
        if mode == 'train':
        #loss_la = LAloss(self.cls_num_list, tau=1.0)
        # compute the group cls loss
            #loss_ce = self.loss_la(g_hat, g)
            g_len = int(output_len/2)
            g_index = g.unsqueeze(-1) + g_len
            yhat = torch.gather(y_hat, dim = 1, index = g_index).squeeze(-1)
            #loss_mse = self.loss_mse(yhat, y)
            #loss = loss_mse + self.sigma *loss_ce
            return yhat, g_hat
        else:
            #if self.output_strategy == 1:
            y_hat_index = output_len/2 + torch.argmax(g_hat, dim =1).unsqueeze(-1)
            yhat_1 = torch.gather(y_hat, dim = 1, index = y_hat_index).squeeze(-1)
            yhat_2 = torch.mean(y_hat[:, int(output_len/2):], dim =1)
            #elif self.output_strategy == 2:
                # wegihted
                #yhat = torch.sum(self.weighted * y_hat[output_len/2:])
            #    pass
            #else:
                #assert "no output strategy defined"
            return yhat_1, yhat_2, g_hat
