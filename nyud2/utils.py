import torch
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import os
import random
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Evaluator:
    def __init__(self):
        self.shot_idx = {
            'many': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49],
            'medium': [7, 8, 46, 48, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 63],
            'few': [57, 59, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        }
        self.output = torch.tensor([], dtype=torch.float32)
        self.depth = torch.tensor([], dtype=torch.float32)

    def __call__(self, output, depth):
        output = output.squeeze().view(-1).cpu()
        depth = depth.squeeze().view(-1).cpu()
        self.output = torch.cat([self.output, output])
        self.depth = torch.cat([self.depth, depth])

    def evaluate_shot_balanced(self):
        metric_dict = {'overall': {}, 'many': {}, 'medium': {}, 'few': {}}
        self.depth_bucket = np.array(list(map(lambda v: self.get_bin_idx(v), self.depth.cpu().numpy())))

        bin_cnt = []
        for i in range(100):
            cnt = np.count_nonzero(self.depth_bucket == i)
            cnt = 1 if cnt >= 1 else 0
            bin_cnt.append(cnt)

        bin_metric = []
        for i in range(100):
            mask = np.zeros(self.depth.size(0), dtype=np.bool)
            mask[np.where(self.depth_bucket == i)[0]] = True
            mask = torch.tensor(mask, dtype=torch.bool)
            bin_metric.append(self.evaluate(self.output[mask], self.depth[mask]))

        for shot in metric_dict.keys():
            if shot == 'overall':
                for k in bin_metric[0].keys():
                    metric_dict[shot][k] = 0.
                    for i in range(7, 100):
                        metric_dict[shot][k] += bin_metric[i][k]
                    if k!= 'NUM':
                        metric_dict[shot][k] /= sum(bin_cnt)
            else:
                for k in bin_metric[0].keys():
                    metric_dict[shot][k] = 0.
                    for i in self.shot_idx[shot]:
                        metric_dict[shot][k] += bin_metric[i][k]
                    if k != 'NUM':
                        metric_dict[shot][k] /= sum([bin_cnt[i] for i in self.shot_idx[shot]])

        logging.info('\n***** TEST RESULTS *****')
        for shot in ['Overall', 'Many', 'Medium', 'Few']:
            logging.info(f" * {shot}: RMSE {metric_dict[shot.lower()]['MSE'] ** 0.5:.3f}\t"
                        f"MSE {metric_dict[shot.lower()]['MSE']:.3f}\t"
                        f"ABS_REL {metric_dict[shot.lower()]['ABS_REL']:.3f}\t"
                        f"LG10 {metric_dict[shot.lower()]['LG10']:.3f}\t"
                        f"MAE {metric_dict[shot.lower()]['MAE']:.3f}\t"
                        f"DELTA1 {metric_dict[shot.lower()]['DELTA1']:.3f}\t"
                        f"DELTA2 {metric_dict[shot.lower()]['DELTA2']:.3f}\t"
                        f"DELTA3 {metric_dict[shot.lower()]['DELTA3']:.3f}\t"
                        f"NUM {metric_dict[shot.lower()]['NUM']}")

        return metric_dict
