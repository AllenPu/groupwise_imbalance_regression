
# this is  the first version : MSE/ bce of the ordinary

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
import time
import math
import pandas as pd
from loss import LAloss, Weight_CE
from network import ResNet_regression, ResNet_ordinal_regression
from datasets.IMDBWIKI import IMDBWIKI
from utils import AverageMeter, accuracy, adjust_learning_rate
from datasets.datasets_utils import group_df
from tqdm import tqdm

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
parser.add_argument('--ord_binary', type=bool, nargs='*', default=True, help='train  with the mode of ordinary regression')
parser.add_argument('--ord_single', type=bool, nargs='*', default=False, help='train  with the mode of ordinary regression with single output')
parser.add_argument('--cls', type=bool, nargs='*', default=False, help='train  with the mode of ordinary regression only for cls')
parser.add_argument('--output_dim', type=int, default=2, help='number of out put dim')
parser.add_argument('--model_depth', type=int, default=50, help='resnet 18 or resnnet 50')

def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    if args.group_mode != 'normal':
        nb_groups = int(args.groups)
        df = group_df(df, nb_groups)
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    ##### how to orgnize the datastes
    #if args.group_mode != 'normal':
    #    nb_groups = int(args.groups)
    #    df_train = group_df(df_train, nb_groups)
    #    df_test = group_df(df_test, nb_groups)
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train', \
                                                                group_num = args.groups,  ord_binary = args.ord_binary,ord_single=args.ord_single)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val', group_num = args.groups)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test', group_num = args.groups)
    #
    train_group_cls_num = train_dataset.get_group()
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, test_loader, val_loader, train_group_cls_num


def train_one_epoch(model, train_loader, mse_loss, or_loss, opt, args):
    model.train()
    mse_y= 0
    # from mse : [1,0] is bigger [0,1] is smaller
    mse_o = 0
    # from rank, e.g. rank 9 - rank 1 = 8 <- abs() required
    mse_o_2 = 0
    # from ce (paper)
    bce_o = 0
    # 
    bce = nn.BCELoss()
    mse_rank = nn.MSELoss()
    #
    for idx, (x, y, g, o) in enumerate(train_loader):
        bsz = x.shape[0]
        gsz = o.shape[1]
        opt.zero_grad()
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        x, y, g, o = x.to(device), y.to(device), g.to(device), o.to(device)
        #
        y_hat, out = model(x)
        #
        y_predicted = torch.gather(y_hat, dim = 1, index = g.to(torch.int64))
        #
        mse_y = mse_loss(y_predicted, y)
        # rank distance from the y based on previous prediction
        pred_ord = torch.sum(out, dim=1)[:, 0]
        pred_ord = pred_ord.unsqueeze(-1)
        mse_o_2 = mse_rank(pred_ord, y)
        # ordinary loss 1
        #mse_o = or_loss(out, o)     
        # ordinary loss 2
        #clone_out  = out.clone()
        out [out  >= 0.5 ] = 1
        out [out  < 0.5 ] = 0
        #
        bce_o = loss_ord(out, o)
        
        #
        loss = mse_y + sigma*mse_o + mse_o_2 + bce_o
        loss.backward(retain_graph=True)
        opt.step()
        #
    return model

def test_step(model, test_loader, device):
    model.eval()
    mse_pred = AverageMeter()
    mae_pred = AverageMeter()
    mae_group = AverageMeter()
    #acc_mae = AverageMeter()
    mse = nn.MSELoss()
    for idx, (inputs, targets, group) in enumerate(test_loader):
        #
        bsz = targets.shape[0]
        #
        inputs = inputs.to(device)
        targets = targets.to(device)
        group = group.to(device)

        with torch.no_grad():
            y_output, ord_out = model(inputs.to(torch.float32))
            #
            ord_out[ord_out >= 0.5] = 1
            ord_out[ord_out < 0.5] = 0
            #
            #print(" shape of is ", ord_out.shape)
            # should not add 1
            pred_ord = torch.sum(ord_out, dim = 1)[:, 0] + 1
            pred_ord = pred_ord.unsqueeze(-1) 
            # write down the acc
            acc_bs = torch.sum(pred_ord == group)/bsz
            #
            #
            y_predicted = torch.gather(y_output, dim = 1, index = group.to(torch.int64))
            # MSE
            mse_1 = mse(y_predicted, targets)
            # MAE
            reduct = torch.abs(y_predicted - targets)
            mae_loss = torch.mean(reduct)
  

        mae_group.update(acc_bs.item(), bsz)
        mse_pred.update(mse_1.item(), bsz)
        mae_pred.update(mae_loss.item(), bsz)

    return mae_group.avg, mse_pred.avg, mae_pred.avg

        


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    loss_ord = Weight_CE()
    loss_ce = LAloss(cls_num_list, tau=args.tau).to(device)
    #oss_or = nn.MSELoss()
    print(" tau is {} group is {} lr is {} ord binary is {} ord sinagle is ".format(\
                                args.tau, args.groups, args.lr, args.ord_binary, args.ord_single))
    #
    #model = ResNet_regression(args).to(device)
    model = ResNet_ordinal_regression(args).to(device)
    #print(model)
    # for cls for group only
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    sigma = args.sigma
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #print(" Training on the epoch ", e)
        adjust_learning_rate(opt, e, args)
        model = train_one_epoch(model, train_loader, loss_mse, loss_ord, opt, args)
    #torch.save(model.state_dict(), './model.pth')
        if e%20 == 0:
            acc_ord, mse_y, mae_y = test_step(model, test_loader, device)
            print('mse of the ordinary group is {}, mse is {}, mae is {}'.format(acc_ord, mse_y, mae_y))
    acc_ord, mse_y, mae_y = test_step(model, test_loader, device)
    print('mse of the ordinary group is {}, mse is {}, mae is {}'.format(acc_ord, mse_y, mae_y))
    # cls for groups only

     
            
