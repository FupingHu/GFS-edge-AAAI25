import math
import numpy as np
import scipy.io as sio
from input_data import load_data
import scipy.sparse as ssp
import random


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg