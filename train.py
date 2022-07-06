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
parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')


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


def train_one_epoch(model, train_loader, mse_loss, ce_loss, opt, device, sigma):
    model.train()
    for idx, (x, y, g) in enumerate(train_loader):
        opt.zero_grad()
        #
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        x, y, g = x.to(device), y.to(device), g.to(device)
        y_output= model(x)
        #split into two parts : 
        #       first is the group, second is the prediction
        y_chunk = torch.chunk(y_output, 2, dim = 1)
        g_hat, y_hat = y_chunk[0], y_chunk[1]
        #
        #extract y out
        y_predicted = torch.gather(y_hat, dim = 1, index = g.to(torch.int64))
        #
        #
        mse_y = mse_loss(y_predicted, y)
        ce_g = ce_loss(g_hat, g.squeeze().long())
        # moreover, we want the predicted g also have the truth guide to the predicted y
        #
        #_, pred = g_hat.topk(1,1,True, True)
        #mse_g = mse_loss(pred, g)
        #
        loss = mse_y + sigma*ce_g
        loss.backward()
        opt.step()
    return model

def test_step(model, test_loader, device):
    model.eval()
    mse_pred = AverageMeter()
    mse_mean = AverageMeter()
    acc_g = AverageMeter()
    acc_mae = AverageMeter()
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
            y_chunk = torch.chunk(y_output, 2, dim = 1)
            g_hat, y_hat = y_chunk[0], y_chunk[1]
            #
            y_predicted = torch.gather(y_hat, dim = 1, index = group.to(torch.int64))
            y_predicted_mean = torch.mean(y_hat, dim = 1).unsqueeze(-1)

            mse_1 = mse(y_predicted, targets)
            mse_mean = mse(y_predicted_mean, targets)
            #
            reduct = torch.abs(y_predicted - targets)
            mae_loss = torch.mean(reduct)
  

            #acc1 = accuracy(y_predicted, targets, topk=(1,))
            #acc2 = accuracy(y_predicted_mean, targets, topk=(1,))
            acc3 = accuracy(g_hat, group, topk=(1,))


        mse_pred.update(mse_1.item(), bsz)
        mse_mean.update(mse_mean.item(), bsz)
        acc_g.update(acc3[0].item(), bsz)
        acc_mae.update(mae_loss.item(), bsz)

    return mse_pred.avg,  mse_mean.avg, acc_g.avg, acc_mae.avg


def train_raw_cls_model(train_loader, model, opt, device, epoch=90):
    model.train()
    for idx, (x, y ,g) in enmuerate(train_loader):
        x, g = x.to(device), g.to(device)
        output = model(x)
        loss = F.cross_entropy(output, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def test_raw_cls_model(test_loaderm model, device):
    model.eval()
    acc = AverageMeter()
    for idx, (x, y ,g) in emuerate(test_loader):
        x, g = x.to(device), g.to(device)
        output = model(x)
        acc_y = accuracy(output, group, topk=(1,))
        acc.update(qcc_y, x.shape[0])
    reurn acc.avg
        
        


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    loss_ce = LAloss(cls_num_list, tau=1.0)
    #
    model = ResNet_regression(args).to(device)
    # for cls for group only
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    sigma = args.sigma
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in range(args.epoch):
        print(" Training on the epoch ", e)
        model = train_one_epoch(model, train_loader, loss_mse, loss_ce, opt, device, sigma)
    acc_y, acc_y2, acc_g, acc_mae = test_step(model, test_loader,device)
    print(' acc of the max is {}, acc of the mean is {}, acc of the group assinment is {}, mae is {}'.format(acc_y, acc_y2, acc_g, acc_mae))
    # cls for groups only
    model_cls = torchvision.models.resnet18(pretrained=False)
    fc_inputs = model_cls.fc.in_features
    model_cls.fc = nn.Linear(fc_inputs, args.groups)
    opt_cls = optim.Adam(model_cls.parameters(), lr=args.lr, weight_decay=5e-4)
    for e in range(args.epoch):
        print(" Training raw on the epoch ", e)
        model_cls = train_raw_cls_model(train_loader, model_cls, opt_cls, device)
    acc_mean = test_raw_cls_model(test_loader, model, device)
    print(" The output acc from raw is : {}".format())
     
            
