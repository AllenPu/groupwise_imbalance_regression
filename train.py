import argparse
from symbol import parameters
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
from network import ResNet_regression, ResNet_ordinal_regression
from datasets.IMDBWIKI import IMDBWIKI
from utils import AverageMeter, accuracy, adjust_learning_rate, shot_metric, setup_seed
from datasets.datasets_utils import group_df
from tqdm import tqdm
# additional for focal
from focal_loss.focal_loss import FocalLoss

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
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')
parser.add_argument('--tau', default=1, type=float, help = ' tau for logit adjustment ')
parser.add_argument('--group_mode', default='i_g', type=str, help = ' b_g is balanced group mode while i_g is imbalanced group mode')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--regulize', type=bool, default=False, help='if to regulaize the previous classification results')
parser.add_argument('--la', type=bool, default=False, help='if use logit adj to train the imbalance')
parser.add_argument('--fl', type=bool, default=False, help='if use focal loss to train the imbalance')
parser.add_argument('--model_depth', type=int, default=50, help='resnet 18 or resnnet 50')




def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    if args.group_mode == 'b_g':
        nb_groups = int(args.groups)
        df = group_df(df, nb_groups)
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    ##### how to orgnize the datastes
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train', group_num = args.groups, group_mode=args.group_mode)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val', group_num = args.groups, group_mode=args.group_mode)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test', group_num = args.groups, group_mode=args.group_mode)
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
    train_labels = df_train['age']
    return train_loader, test_loader, val_loader, train_group_cls_num, train_labels


def train_one_epoch(model, train_loader, ce_loss, mse_loss, opt, args):
    sigma, regu, la, fl = args.sigma, args.regulize, args.la, args.fl
    model.train()
    mse_y = 0
    ce_g = 0
    #
    if fl:
        m = torch.nn.Softmax(-1)
    #
    for idx, (x, y, g) in enumerate(train_loader):
        opt.zero_grad()
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        y_output, z = model(x)

        #split into two parts : first is the group, second is the prediction
        y_chunk = torch.chunk(y_output, 2, dim = 1)
        g_hat, y_hat = y_chunk[0], y_chunk[1]      
        #
        #extract y out
        y_predicted = torch.gather(y_hat, dim = 1, index = g.to(torch.int64))
        #
        mse_y = mse_loss(y_predicted, y)
        if regu :
            if la:
                ce_g = ce_loss(g_hat, g.squeeze().long())
            if fl:
                ce_g = ce_loss(m(g_hat), g.squeeze().long())
        else :
            ce_g = F.cross_entropy(g_hat, g.squeeze().long())
        #
        loss = mse_y + sigma*ce_g
        loss.backward()
        opt.step()
        #
    return model

def test_step(model, test_loader, train_labels):
    model.eval()
    mse_gt = AverageMeter()
    #mse_mean = AverageMeter()
    acc_g = AverageMeter()
    acc_mae_gt = AverageMeter()
    mse_pred = AverageMeter()
    acc_mae_pred = AverageMeter()
    mse = nn.MSELoss()
    pred, labels = [], []
    for idx, (inputs, targets, group) in enumerate(test_loader):
        #
        bsz = targets.shape[0]
        #
        inputs = inputs.to(device)
        targets = targets.to(device)
        group = group.to(device)
        #
        labels.extend(targets.data.cpu().numpy())

        with torch.no_grad():
            y_output, _ = model(inputs.to(torch.float32))
            y_chunk = torch.chunk(y_output, 2, dim = 1)
            g_hat, y_hat = y_chunk[0], y_chunk[1]
            #
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            #
            group = group.to(torch.int64)
            #
            y_gt = torch.gather(y_hat, dim = 1, index = group.to(torch.int64))
            #y_predicted_mean = torch.mean(y_hat, dim = 1).unsqueeze(-1)
            y_pred = torch.gather(y_hat, dim=1, index = g_index)
            # 
            pred.extend(y_pred.data.cpu().numpy())
            #
            mse_1 = mse(y_gt, targets)
            mse_2 = mse(y_pred, targets)
            #mse_mean_1 = mse(y_predicted_mean, targets)
            #
            reduct = torch.abs(y_gt - targets)
            mae_loss = torch.mean(reduct)
            #
            mae_loss_2 = torch.mean(torch.abs(y_pred - targets))
            
            #acc1 = accuracy(y_predicted, targets, topk=(1,))
            #acc2 = accuracy(y_predicted_mean, targets, topk=(1,))
            acc3 = accuracy(g_hat, group, topk=(1,))


        mse_gt.update(mse_1.item(), bsz)
        #mse_mean.update(mse_mean_1.item(), bsz)
        mse_pred.update(mse_2.item(), bsz)
        acc_g.update(acc3[0].item(), bsz)
        acc_mae_gt.update(mae_loss.item(), bsz)
        acc_mae_pred.update(mae_loss_2.item() ,bsz)
        #
    shot_dict = shot_metric(pred, labels, train_labels)


    return mse_gt.avg,  mse_pred.avg, acc_g.avg, acc_mae_gt.avg, acc_mae_pred.avg, shot_dict

        


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    #
    total_result = 'total_result_model_'+str(args.model_depth)+'.txt'
    #
    store_name = 'la_' + str(args.la) + '_regu_' + str(args.regulize) + '_tau_'+ str(args.tau) + \
                        '_lr_' + str(args.lr) + '_g_'+ str(args.groups) + '_model_' + str(args.model_depth) + '_epoch_' + str(args.epoch)
    ####
    print(" store name is ", store_name)
    #
    store_name = store_name + '.txt'
    #
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_dataset(args)
    #
    loss_mse = nn.MSELoss()
    loss_ce = LAloss(cls_num_list, tau=args.tau).to(device)
    #oss_or = nn.MSELoss()
    #
    model = ResNet_regression(args).to(device)
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    # focal loss
    if args.fl:
        loss_ce = FocalLoss(gamma=0.75)
    #
    print(" tau is {} group is {} lr is {} model depth {}".format(args.tau, args.groups, args.lr, args.model_depth))
    #
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #print(" Training on the epoch ", e)
        adjust_learning_rate(opt, e, args)
        model = train_one_epoch(model, train_loader, loss_ce, loss_mse, opt, args)
    #torch.save(model.state_dict(), './model.pth')
    acc_gt, acc_pred, g_pred, mae_gt, mae_pred, shot_dict = test_step(model, test_loader, train_labels)
    #
    print(' mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred))
    with open(store_name, 'w') as f:
        f.write(' tau is {} group is {} lr is {} model depth {} epoch {}'.format(args.tau, args.groups, args.lr, args.model_depth, args.epoch) +"\n" )
        f.write(' mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred)+"\n")
        #
        #
        f.write(' Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict['many']['l1'], shot_dict['median']['l1'], shot_dict['low']['l1'])+ "\n" )
        #
        f.close()
    # cls for groups only
    with open(total_result, 'a') as f:
        f.write(' new '+"\n")
        f.write(' tau is {} group is {} lr is {} model depth {} epoch {}'.format(args.tau, args.groups, args.lr, args.model_depth, args.epoch)+"\n")
        f.write(' mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred)+"\n")
        f.close()
