#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import logging
import glob

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import segmentation_models_pytorch as smp

from count_train import CountSystem
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger



def main_evaluate():
    logging.basicConfig(level=logging.INFO)

    path_idx_trn = '/home/ar/data/yshad-2019/corn_seed_counting_train_v2_clean/idx-val.txt'
    path_idx_val = '/home/ar/data/yshad-2019/corn_seed_counting_train_v2_clean/idx-val.txt'
    path_ckpt = path_idx_trn + '_ckpt'
    model_counter = CountSystem(path_idx_trn, path_idx_val).build()
    logger = TestTubeLogger(save_dir=path_ckpt, version=1)
    weights_path = glob.glob(os.path.join(logger.experiment.get_logdir(), '../../../results/*.ckpt'))[0]
    model_counter.load_state_dict(torch.load(weights_path)['state_dict'])
    dataloader_ = model_counter.val_dataloader()[0]
    with torch.no_grad():
        for xi, x in enumerate(dataloader_):
            img = x['img']
            y_gt = x['msk']
            img = img.permute([0, 3, 1, 2]).type(torch.float32) / 255.
            y_gt = y_gt.type(torch.float32) / 255.
            y_pr = torch.sigmoid(model_counter.forward()).cpu().numpy()

            print('-')



if __name__ == '__main__':
    main_evaluate()