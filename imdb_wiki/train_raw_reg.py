import argparse
from mimetypes import init
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
from utils import AverageMeter, accuracy, adjust_learning_rate
from train import get_dataset
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=123)
parser.add_argument('--mode', default='train', type= str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--groups', type=int, default=10, help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')
parser.add_argument('--output_dim', type=int, default=10, help='output dim of network')
parser.add_argument('--tau', type=float, default=1, help='output dim of network')
parser.add_argument('--group_mode', default='normal', type=str, help = ' group mode for group orgnize')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--reg_mode', default='g', type=str, help = ' y denotes for regress for label, g for group')

def get_model(model_name = 'resnet50', output_dim = 10):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    else:
        print(" no model specified! ")
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, output_dim)
    return model

def train_raw_reg_model(train_loader, model, mse_loss, opt, device, reg_mode = 'g'):
    model.train()
    for idx, (x, y ,g) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        output = model(x)
        if reg_mode == 'g':
            loss = mse_loss(output, g)
        else:
            loss = mse_loss(output, y)
        loss = mse_loss(output, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_raw_reg_model(test_loader, model, device, reg_mode = 'g'):
    model.eval()
    mae = AverageMeter()
    #
    ceil_meter = AverageMeter()
    floor_meter = AverageMeter()
    #
    if reg_mode == 'g':
        for idx, (x, y ,g) in enumerate(test_loader):
            bsz = x.shape[0]
            with torch.no_grad():
                x, y, g = x.to(device), y.to(device), g.to(device)
                output = model(x)
                #reduct = torch.abs(output - y)
                reduct = torch.abs(output - g)
                mae_loss = torch.mean(reduct)
                #
                ceil = torch.ceil(output)
                floor = torch.floor(output)
                ceil_acc = torch.sum(torch.eq(ceil, g))/bsz
                floor_acc = torch.sum(torch.eq(floor, g))/bsz
            #
            mae.update(mae_loss.item(), bsz)
            #
            ceil_meter.update(ceil_acc.item(), bsz)
            floor_meter.update(floor_acc.item(), bsz)

        return mae.avg, ceil_meter.avg, floor_meter.avg
    else:
        for idx, (x, y ,g) in enumerate(test_loader):
            bsz = x.shape[0]
            with torch.no_grad():
                x, y, g = x.to(device), y.to(device), g.to(device)
                output = model(x)
                #reduct = torch.abs(output - y)
                reduct = torch.abs(output - g)
                mae_loss = torch.mean(reduct)
            mae.update(mae_loss.item(), bsz)
        return mae.avg






if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    model = get_model(output_dim = 1)
    model = model.to(device)
    #
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    mse_loss = nn.MSELoss()
    #
    for e in tqdm(range(args.epoch)):
        adjust_learning_rate(opt, e, args)
        model = train_raw_reg_model(train_loader, model, mse_loss, opt, device, args.reg_mode)
    mae, ceil, floor = test_raw_reg_model(test_loader, model, device)
    #print(" the mae of the reg for original is {}".format(mae))
    if args.reg_mode == 'g':
        mae, ceil, floor = test_raw_reg_model(test_loader, model, device, args.reg_mode)
        print(" the mae of the reg for original is {}, ceil {}, floor {}".format(mae, ceil, floor))
    else:
        mae = test_raw_reg_model(test_loader, model, device, args.reg_mode)
        print(" the mae of the reg for original is {}".format(mae))
    

        
