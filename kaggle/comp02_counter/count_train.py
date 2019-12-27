#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import logging

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import segmentation_models_pytorch as smp

from count_data import CountDataset, CountDataset_Validation, worker_init_fn_random
from count_loss import BCEWLoss

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger



class CountSystem(pl.LightningModule):

    def __init__(self, path_idx_trn: str, path_idx_val: str, batch_size=4):
        super(CountSystem, self).__init__()
        self.path_idx_trn = path_idx_trn
        self.path_idx_val = path_idx_val
        self.batch_size = batch_size
        self.dir_batches = self.path_idx_val + '_batches'
        self.model = smp.Unet('resnet34', encoder_weights=None, activation=None)
        # self.model = nn.Conv2d(3, 1, 3, padding=1)
        self.loss_fun = BCEWLoss()

    def build(self):
        self.dataset_trn = CountDataset(path_idx=self.path_idx_trn).build()
        self.dataset_val = CountDataset_Validation(dir_batches=self.dir_batches).build()
        return self

    def forward(self, x):
        # return torch.relu(self.l1(x.view(x.size(0), -1)))
        return torch.squeeze(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        img, y_gt = batch['img'], batch['msk']
        img = img.permute([0, 3, 1, 2]).type(torch.float32)/255.
        y_gt = y_gt.type(torch.float32) / 255.
        #
        y_pr = self.forward(img)
        loss_ = self.loss_fun(y_pr, y_gt)
        tensorboard_logs = {'train_loss': loss_}
        return {'loss': loss_, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        img, y_gt = batch['img'], batch['msk']
        img = img.permute([0, 3, 1, 2]).type(torch.float32) / 255.
        y_gt = y_gt.type(torch.float32) / 255.
        y_pr = self.forward(img)
        loss_ = self.loss_fun(y_pr, y_gt)
        return {'val_loss': loss_}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.dataset_trn, batch_size=self.batch_size, num_workers=1, worker_init_fn=worker_init_fn_random)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=1)


def main_train():
    logging.basicConfig(level=logging.INFO)
    path_idx_trn = '/home/ar/data/yshad-2019/corn_seed_counting_train_v2_clean/idx-val.txt'
    path_idx_val = '/home/ar/data/yshad-2019/corn_seed_counting_train_v2_clean/idx-val.txt'
    path_ckpt = path_idx_trn + '_ckpt'
    model_counter = CountSystem(path_idx_trn, path_idx_val).build()
    logger = TestTubeLogger(save_dir=path_ckpt, version=1)
    #
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(path_ckpt, 'results'),
                                          verbose=True, monitor='val_loss', mode='min', save_best_only=True)
    trainer = Trainer(default_save_path=path_ckpt,
                      gpus=0,
                      logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      max_nb_epochs=100,
                      early_stop_callback=False)
    trainer.fit(model_counter)



if __name__ == '__main__':
    main_train()