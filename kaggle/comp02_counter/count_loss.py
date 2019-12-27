#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import torch
from torch import nn


class BCEWLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pr, y_gt, mskw=None):
        ret = self.bce(y_pr, y_gt)
        if mskw is not None:
            ret *= mskw
        ret = ret.mean()
        return ret


def main_test():
    loss_fun = BCEWLoss()
    img = torch.zeros([10, 10], dtype=torch.float32)
    msk = torch.zeros([10, 10], dtype=torch.float32)
    img[2:-2] = -10
    msk[1:-3] = 1
    loss = loss_fun(img, msk)
    print('-')


if __name__ == '__main__':
    main_test()