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
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')
parser.add_argument('--output_dim', type=int, default=10, help='output dim of network')

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

def train_raw_cls_model(train_loader, model, loss_la, opt, device):
    model.train()
    for idx, (x, y ,g) in enumerate(train_loader):
        x, g = x.to(device), g.to(device)
        output = model(x)
        if output.shape[-1] == 20:
            outputs = torch.chunk(output,2 dim=1)
            loss = loss_la(outputs[0], g.squeeze().long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_raw_cls_model(test_loader, model, device):
    model.eval()
    acc = AverageMeter()
    for idx, (x, y ,g) in enumerate(test_loader):
        bsz = x.shape[0]
        with torch.no_grad():
            x, g = x.to(device), g.to(device)
            output = model(x)
            if output.shape[-1] == 20:
                outputs = torch.chunk(output,2 dim=1)
            acc_y = accuracy(outputs[0], g, topk=(1,))
        acc.update(acc_y[0].item(), bsz)
    return acc.avg






if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    ####
    train_loader, test_loader, val_loader,  cls_num_list = get_dataset(args)
    #
    model = get_model(output_dim = args.output_dim)
    model = model.to(device)
    #
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    loss_la = LAloss(cls_num_list, tau=1.0).to(device)
    #
    for e in range(args.epoch):
        adjust_learning_rate(opt, e, args)
        model = train_raw_cls_model(train_loader, model, loss_la, opt, device)
    acc = test_raw_cls_model(test_loader, model, device)
    print(" the acc of the cls for groups is {}".format(acc))
    

        
