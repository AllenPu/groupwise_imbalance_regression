import numpy as np
import torch


def group_df(df, nb_groups):
    """
    :param df: data frame for training & test 
    :param nb_groups: int for the number of groups
    :return: new data frame with group features
    """
    length = len(df)
    target_v, target_c = np.unique(df["age"].values, return_counts=True)
    target_q = np.zeros_like(target_c)
    nb_perg = length//nb_groups
    cur_group = 0
    sum_item = 0
    for k, i in enumerate(target_c):
        sum_item += i
        target_q[k] = cur_group
        if sum_item >= nb_perg and cur_group <= nb_groups - 1:
            cur_group += 1
            sum_item = 0

    key = dict(zip(target_v,target_q))

    df["group"] = df["age"].map(key)

    return df


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
