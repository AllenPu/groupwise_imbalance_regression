
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import json
import os
import torch
import sys
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time
import math
import pandas as pd
from loss import LAloss
from network import ResNet_regression, ResNet_ordinal_regression, ResNet_regression_sep
from datasets.IMDBWIKI import IMDBWIKI
from utils import AverageMeter, accuracy, adjust_learning_rate
from datasets.datasets_utils import group_df
from tqdm import tqdm
from train import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=123)
parser.add_argument('--mode', default='train', type= str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--groups', type=int, default=10, help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')
parser.add_argument('--tau', default=1, type=int, help = ' tau for logit adjustment ')
parser.add_argument('--group_mode', default='normal', type=str, help = ' group mode for group orgnize')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--output_dim', type=int, default=1, help='output dim of network')



def train_one_epoch(model, train_loader, mse_loss, opt, device, sigma):
    model.train()
    for idx, (x, y, g) in enumerate(train_loader):
        opt.zero_grad()
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        # should be (batch, 1)
        y_output = model(x)
        #
        g_pred, y_pred = torch.split(y_output, [1,10], dim=1)
        #
        g_pred_floor = torch.floor(g_pred)
        deter = g_pred - g_pred_floor
        deter[deter>=0.5] = 1
        deter[deter<0.5] = 1
        g_hat = g_pred_floor + deter
        #         
        y_hat = torch.gather(y_pred, dim = 1, index = g.to(torch.int64))    
        #
        mse_g = mse_loss(g_hat, g)
        mse_y = mse_loss(y_hat, y)
        #
        loss = mse_y + mse_g
        loss.backward()
        opt.step()
        #
    return model

def test_step(model, test_loader, device):
    model.eval()
    acc_g_floor = AverageMeter()
    acc_g_ceil = AverageMeter()
    mae_y_pred = AverageMeter()

    for idx, (inputs, targets, group) in enumerate(test_loader):
        #
        bsz = targets.shape[0]
        #
        inputs = inputs.to(device)
        targets = targets.to(device)
        group = group.to(device)

        with torch.no_grad():
            y_output = model(inputs.to(torch.float32))
            #
            g_hat, y_hat = torch.split(y_output, [1,10], dim=1)
            #
            g_hat_floor = torch.floor(g_hat)
            g_hat_ceil = torch.ceil(g_hat)
            #
            g_acc_floor = torch.sum(g_hat_floor==targets) / bsz
            g_acc_ceil = torch.sum(g_hat_ceil==targets) / bsz
            #
            mae_y = torch.mean(torch.abs(y_hat-targets))


        acc_g_floor.update(g_acc_floor.item(), bsz)
        acc_g_ceil.update(g_acc_ceil.item(), bsz)
        mae_y_pred.update(mae_y.item(), bsz)

    return acc_g_floor.avg, acc_g_ceil.avg,  mae_y_pred.avg

        


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    #loss_ce = LAloss(cls_num_list, tau=args.tau).to(device)
    #oss_or = nn.MSELoss()
    model = ResNet_regression_sep(args).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    num_groups = args.groups
    #
    sigma = args.sigma
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #print(" Training on the epoch {} with group {}".format(e, gs))
        adjust_learning_rate(opt, e, args)
        model = train_one_epoch(model, train_loader, loss_mse, opt, device, sigma)
        #torch.save(model.state_dict(), './model.pth')
    acc_floor, acc_ceil, mae_y = test_step(model, test_loader,device)
    print('acc floor is {} acc ceil is {}, mae y is {},'.format(acc_floor, acc_ceil, mae_y))
    # cls for groups only

     
            
