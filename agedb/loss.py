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