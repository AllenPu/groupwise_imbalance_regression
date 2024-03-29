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

def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    if args.group_mode != 'normal':
        nb_groups = int(args.groups)
        df = group_df(df, nb_groups)
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']

    group = args.groups
    train_list = []
    test_list = []
    val_list = []
    train_cls_num = []
    group_range = int(100/group)
    for i in range(group):
        start = i * group_range
        end = (i+1) * group_range
        if i == group - 1:
            df_tr = df_train[ start<= df_train['age']]
            df_te = df_test[ start<= df_test['age']]
            df_va = df_val[ start<= df_val['age']]
        else:
            df_tr = df_train[ (start<= df_train['age'] ) & (df_train['age'] < end)]
            df_te = df_test[ (start<= df_test['age'] ) & (df_test['age'] < end)]
            df_va = df_val[ (start<= df_val['age'] ) & (df_val['age'] < end)]
        train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_tr, img_size=args.img_size, split='train', group_num = args.groups)
        val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_va, img_size=args.img_size, split='val', group_num = args.groups)
        test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_te, img_size=args.img_size, split='test', group_num = args.groups)
        #
        #print(f"Training data size: {len(train_dataset)}")
        #print(f"Validation data size: {len(val_dataset)}")
        #print(f"Test data size: {len(test_dataset)}")
        #
        train_group_cls_num = train_dataset.get_group()
        #
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
        print(f"Group dataset is : {i}")
        print(f"Done Training data size: {len(train_dataset)}")
        print(f"Done Validation data size: {len(val_dataset)}")
        print(f"Done Test data size: {len(test_dataset)}")
        #
        train_list.append(train_loader)
        test_list.append(test_loader)
        val_list.append(val_loader)
        train_cls_num.append(train_group_cls_num)
        #
    return train_list, test_list, val_list, train_cls_num


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
        # 10
        if y_output.shape[-1] == 10:
            y_pred = torch.gather(y_output, dim = 1, index = g.to(torch.int64))
        # 20
        if y_output.shape[-1] == 20:
            y_hat = torch.chunk(y_output, 2, dim=1)
            y_pred = torch.gather(y_hat[1], dim = 1, index = g.to(torch.int64))
        # 1
        else:
            y_pred = y_output
        #
        mse_y = mse_loss(y_pred, y)
        #
        loss = mse_y
        loss.backward()
        opt.step()
        #
    return model

def test_step(model, test_loader, device):
    model.eval()
    mse_pred = AverageMeter()
    mae_pred = AverageMeter()
    mse = nn.MSELoss()
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
            if y_output.shape[-1] == 10:
                y_output = torch.gather(y_output, dim = 1, index = group.to(torch.int64))
            if y_output.shape[-1] == 20:
                y_hat = torch.chunk(y_output, 2, dim=1)
                y_output = torch.gather(y_hat[1], dim = 1, index = group.to(torch.int64))
            
            mse_y = mse(y_output, targets)
            mae_y = torch.mean(torch.abs(y_output-targets))



        mse_pred.update(mse_y.item(), bsz)
        mae_pred.update(mae_y.item(), bsz)

    return mse_pred.avg,  mae_pred.avg

        


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
    #
    if args.output_dim != 1:
        model = ResNet_regression_sep(args).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    num_groups = args.groups
    #
    sigma = args.sigma
    #print(" raw model for group classification trained at epoch {}".format(e))
    for gs in range(num_groups):
        #
        if args.output_dim == 1:
            model = ResNet_regression_sep(args).to(device)
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        #
        for e in tqdm(range(args.epoch)):
            #print(" Training on the epoch {} with group {}".format(e, gs))
            adjust_learning_rate(opt, e, args)
            model = train_one_epoch(model, train_loader[gs], loss_mse, opt, device, sigma)
        #torch.save(model.state_dict(), './model.pth')
        mse_y, mae_y = test_step(model, test_loader[gs],device)
        print('group is {} mse is {}, mae is {},'.format(gs, mse_y, mae_y))
    # cls for groups only

     
            
