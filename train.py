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
from network import ResNet_regression
from datasets.IMDBWIKI import IMDBWIKI
from utils import AverageMeter, accuracy

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
parser.add_argument('--seed', default=123)


def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train', group_num = args.groups)
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


def train_one_epoch(model, train_loader, mse_loss, ce_loss, opt, device, sigma, mode):
    model.train()
    for idx, (x, y, g) in enumerate(train_loader):
        #
        x, y, g = x.to(device), y.to(device), g.to(device)
        y_hat, g_hat = model(x, g, mode)
        #
        opt.zero_grad()
        #
        loss_mse = mse_loss(y_hat, y)
        loss_ce = ce_loss(g_hat, g)
        loss = loss_mse + sigma*loss_ce
        loss.backward()
        opt.step()
    return model

def test_step(net, loader, device, mode):
    net.eval()
    acc = AverageMeter()
    acc_2 = AverageMeter()
    acc_g = AverageMeter()
    for idx, (inputs, targets, group) in enumerate(loader):

        bsz = targets.shape[0]

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            y_hat_1, y_hat_2, g_hat = net(inputs.to(torch.float32), mode)
            acc1 = accuracy(y_hat_1, targets, topk=(1,))
            acc2 = accuracy(g_hat, group, topk=(1,))
            acc3 = accuracy(y_hat_2, targets, topk=(1,))


        acc.update(acc1[0].item(), bsz)
        acc_g.update(acc2[0].item(), bsz)
        acc_2.update(acc3[0].item(), bsz)

    return acc.avg,  acc_2.avg, acc_g.avg


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    loss_ce = LAloss(cls_num_list, tau=1.0)
    #
    model = ResNet_regression(args).to(device)
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    sigma = args.sigma
    #
    for e in range(args.epoch):
        model = train_one_epoch(model, train_loader, loss_mse, loss_ce, opt, device)
    acc_y, acc_y2, acc_g = test_step(model, test_loader,device)
    print(' acc of the max is {}, acc of the mean is {}, acc of the group assinment is {}'.format(acc_y, acc_y2, acc_g))