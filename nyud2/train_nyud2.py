import argparse
import time
import os
import shutil
import logging
import torch
import torch.backends.cudnn as cudnn
from datasets import load_nyud2_data as loaddata
from tqdm import tqdm
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')


def load_data(args):
    train_loader = loaddata.getTrainingData(args, args.batch_size)
    test_loader = loaddata.getTestingData(args, 1)
    return train_loader, test_loader


def train_one_epoch(model, train_loader,loss_ce, loss_mse, opt, args):
    model.train()
    for idex, sample in enumerate(train_loader):
        x, y = sample['image'], sample['depth']
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        pred = model(x)
        chunks = torch.chunk(pred, 2, dim = 1)
        g_pred, y_pred = chunks[0], chunks[1]
        y_hat = torch.gather(y_pred, dim = 1, index = g.to(torch.int64))
        #
        loss = 0
        loss_list = []
        loss_list.append(loss_ce(g_pred, g.squeeze().long()))
        loss_list.append(loss_mse(y_hat, y))
        #
        for i in loss_list:
            loss += i
        loss.backwards()
        opt.step()
    return model

def test(model, test_loader, train_labels, args):
    model.eval()#
    acc_g = AverageMeter()
    mae_y_pred = AverageMeter()
    mae_y_gt = AverageMeter()
    mse_y_pred = AverageMeter()
    mse_y_gt = AverageMeter()
    pred_gt, pred, labels = [], [], []
    for idex, sample in enumerate(test_loader):
        x, y = sample['image'], sample['depth']
        labels.extend(y.data.numpy())
        bsz = x.size(0)
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        with torch.no_grad():
            pred = model(x)
            chunks = torch.chunk(pred, 2, dim = 1)
            g_pred, y_pred = chunks[0], chunks[1]
            y_hat_gt = torch.gather(y_pred, dim = 1, index = g.to(torch.int64))
            g_hat = torch.argmax(g_pred, dim=1).unsqueeze(-1)
            y_hat_pred = torch.gather(y_pred, dim=1, index = g_hat)
            #
            pred_gt.extend(y_hat_gt.data.cpu().numpy())
            pred.extend(y_hat_pred.data.cpu().numpy())
            #
            acc = accuracy(g_hat, g, topk=(1,))
            mae_pred = torch.mean(torch.abs(y_hat_pred - y))
            mae_gt = torch.mean(torch.abs(y_hat_gt - y))
            mse_pred = torch.mean((y_hat_pred - y) ** 2)
            mse_gt = torch.mean((y_hat_gt - y) ** 2)
        #
        acc_g.update(acc[0].item(),bsz)
        mae_y_pred.update(mae_pred.item(),bsz)
        mae_y_gt.update(mae_gt.item(),bsz)
        mse_y_pred.update(mse_pred.item(), bsz)
        mse_y_gt.update(mse_gt.item(), bsz)
    #


