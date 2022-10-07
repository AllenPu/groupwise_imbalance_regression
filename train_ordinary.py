
# this is the seconf version of ordinary : -p log q

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
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='imdb_wiki',
                    choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='./data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--seeds', default=123, type=int, help=' random seed ')
parser.add_argument('--tau', default=1, type=int,
                    help=' tau for logit adjustment ')
parser.add_argument('--group_mode', default='normal',
                    type=str, help=' group mode for group orgnize')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--ord', type=bool, nargs='*', default=False,
                    help='train  with the mode of ordinary regression')
parser.add_argument('--cls', type=bool, nargs='*', default=False,
                    help='train  with the mode of ordinary regression only for cls')


def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    if args.group_mode != 'normal':
        nb_groups = int(args.groups)
        df = group_df(df, nb_groups)
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    ##### how to orgnize the datastes
    #if args.group_mode != 'normal':
    #    nb_groups = int(args.groups)
    #    df_train = group_df(df_train, nb_groups)
    #    df_test = group_df(df_test, nb_groups)
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train,
                             img_size=args.img_size, split='train', group_num=args.groups, ord=args.ord)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val,
                           img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test,
                            img_size=args.img_size, split='test', group_num=args.groups)
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



#
#
# To rewrite the two step update
#
#
def train_one_epoch(model, train_loader, ce, opt, args):
    model.train()
    mse_y = 0
    #
    mse = nn.MSELoss()
    #
    for idx, (x, y, g, o) in enumerate(train_loader):
        #
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        x, y, g, o = x.to(device), y.to(device), g.to(device), o.to(device)
        #
        y_hat,  out = model(x)
        # ground truth
        y_predicted = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
        #
        mse_y = mse(y_predicted, y)
        #
        ce_y = ce(out, o)
        #
        loss = mse_y + ce_y
        #
        loss.backward()
        #
        opt.step()
        
    return model


def test_step(model, test_loader, device):
    model.eval()
    mse_pred = AverageMeter()
    mae_pred = AverageMeter()
    acc_group = AverageMeter()
    acc_1_group = AverageMeter()
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
            pred_ord = torch.sum(ord_out, dim=1)[:, 0]
            pred_ord = pred_ord.unsqueeze(-1)
            #
            assert pred_ord.shape == group.shape
            # write down the acc
            acc_bs = torch.sum(pred_ord == group)/bsz
            #
            if idx == 0:
                torch.save(pred_ord,'./tensor/pred_out.pt')
                torch.save(group, './tensor/group.pt')
                torch.save(ord_out, './tensor/ord_out.pt')
            #
            # the prediceted y = y + 1
            acc_bs_plus_1 = torch.sum((pred_ord+1) == group)/bsz
            #
            y_predicted = torch.gather(
                y_output, dim=1, index=group.to(torch.int64))
            # MSE
            mse_1 = mse(y_predicted, targets)
            # MAE
            reduct = torch.abs(y_predicted - targets)
            mae_loss = torch.mean(reduct)

        acc_group.update(acc_bs.item(), bsz)
        acc_1_group.update(acc_bs_plus_1.item(), bsz)
        mse_pred.update(mse_1.item(), bsz)
        mae_pred.update(mae_loss.item(), bsz)

    return acc_group.avg, acc_1_group.avg, mse_pred.avg, mae_pred.avg


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    loss_ord = nn.MSELoss()
    loss_ce = LAloss(cls_num_list, tau=args.tau).to(device)
    ce = Weight_CE().cuda()
    #oss_or = nn.MSELoss()
    #
    #model = ResNet_regression(args).to(device)
    model = ResNet_ordinal_regression(args).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    sigma = args.sigma
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #print(" Training on the epoch ", e)
        adjust_learning_rate(opt, e, args)
        model = train_one_epoch(model, train_loader, ce, opt, args)
    #torch.save(model.state_dict(), './model.pth')
    acc_ord, acc_1_ord, mse_y, mae_y = test_step(model, test_loader, device)
    print('acc of the ordinary group is {}, acc plus 1 is {} mse is {}, mae is {}'.format(
        acc_ord, acc_1_ord, mse_y, mae_y))
    # cls for groups only
