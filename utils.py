import torch
import numpy as np

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


def collect_results(output, target, groups, preds, labels):
    # direct output of model
    # direct target return
    # preds is a list record all the true predicts
    # labels is a list record all the true labels
    labels.extend(target.data.cpu().numpy())
    # gather  the predicted y
    y_predicted = torch.gather(output, dim=1, index=groups.to(torch.int64))
    # 
    preds.extend(y_predicted.data.cpu().numpy())
    #  
    return preds, labels


def short_metric(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(\
            train_labels[train_labels == l]))
        test_class_count.append(\
            len(labels[labels == l]))
        l1_per_class.append(\
            np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(\
            np.abs(preds[labels == l] - labels[labels == l]))


    return 0
