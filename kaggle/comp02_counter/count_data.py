#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import cv2
import time
import skimage.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union as U, Optional as O


def affine_matrix_rot_scale_around_center(center_rc: U[tuple, list],
                                          angle_deg: float,
                                          scale: float,
                                          crop_size: U[tuple, list, np.ndarray, int]) -> np.ndarray:
    center_xy = center_rc[::-1]
    if not isinstance(crop_size, (tuple, list, np.ndarray)):
        crop_size = int(crop_size)
        crop_size = (crop_size, crop_size)
    # (1) precalc parameters
    ang_rad = (np.pi / 180.) * angle_deg
    cosa = np.cos(ang_rad)
    sina = np.sin(ang_rad)
    # (2) prepare separate affine transformation matrices
    matShiftB = np.array([[1., 0., -center_xy[0]], [0., 1., -center_xy[1]], [0., 0., 1.]])
    matRot = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
    # matShiftF = np.array([[1., 0., +center_xy[0]], [0., 1., +center_xy[1]], [0., 0., 1.]])
    matScale = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., 1.]])
    matShiftCrop = np.array([[1., 0., crop_size[0] / 2.], [0., 1., crop_size[1] / 2.], [0., 0., 1.]])
    # (3) build total-matrix
    mat_total = matShiftCrop.dot(matRot.dot(matScale.dot(matShiftB)))
    return mat_total


def get_random_crop_img(sample: dict, rnd_angles: tuple, rnd_scales: tuple, crop_size: int) -> dict:
    img, msk = sample['img'], sample['msk']
    size_rc = img.shape[:2]
    rnd_r = np.random.randint(crop_size, size_rc[0] - crop_size)
    rnd_c = np.random.randint(crop_size, size_rc[1] - crop_size)
    rc = (rnd_r, rnd_c)
    angle = np.random.uniform(*rnd_angles)
    scale = np.random.uniform(*rnd_scales)
    rnd_mat = affine_matrix_rot_scale_around_center(rc, angle, scale, crop_size=crop_size)
    rnd_mat = rnd_mat[:-1, :]
    img_crop = cv2.warpAffine(img, rnd_mat,
                              dsize=(crop_size, crop_size),
                              borderMode=cv2.BORDER_REFLECT,
                              flags=cv2.INTER_LINEAR)
    msk_crop = cv2.warpAffine(img, rnd_mat,
                              dsize=(crop_size, crop_size),
                              borderMode=cv2.BORDER_REFLECT,
                              flags=cv2.INTER_LINEAR)
    ret = {
        'img': img_crop,
        'msk': msk_crop
    }
    return ret


class CountDataset(Dataset):

    def __init__(self, path_idx: str, in_memory=True, dataset_size=400, crop_size: int = 256):
        self.path_idx = path_idx
        self.wdir = os.path.dirname(self.path_idx)
        self.data_idx = pd.read_csv(self.path_idx)
        self.data_idx['path_img_abs'] = [os.path.join(self.wdir, x) for x in self.data_idx['path_img']]
        self.data_idx['path_msk_abs'] = [os.path.join(self.wdir, x) for x in self.data_idx['path_msk']]
        self.data = None
        self.in_memory = in_memory
        self.dataset_size = dataset_size
        self.crop_size = crop_size
        self.rnd_angles = (-30, +30)
        self.rnd_scales = (0.5, 2)

    def _read_sample(self, idx: int) -> dict:
        row = self.data_idx.iloc[idx]
        path_img = row['path_img_abs']
        path_msk = row['path_msk_abs']
        img = np.array(io.imread(path_img))[:, :, :3]
        msk = np.array(io.imread(path_msk))
        ret = {
            'img': img,
            'msk': msk
        }
        return ret

    def get_sample(self, idx: int) -> dict:
        if self.in_memory:
            return self.data[idx]
        else:
            return self._read_sample(idx)

    def build(self):
        if self.in_memory:
            t1 = time.time()
            logging.info(':: loading data into memeory...')
            self.data = [self._read_sample(x) for x in range(len(self.data_idx))]
            dt = time.time() - t1
            logging.info(f'\t\t... done, dt ~ {dt:0.2f}')
        else:
            self.data = None
        return self

    def __len__(self):
        if self.dataset_size is not None:
            return len(self.data_idx)
        else:
            return self.dataset_size

    def _crop_random(self, sample: dict) -> dict:
        crop = get_random_crop_img(sample,
                                   rnd_angles=self.rnd_angles,
                                   rnd_scales=self.rnd_scales,
                                   crop_size=self.crop_size)
        return crop

    def __getitem__(self, item):
        idx = np.random.randint(0, len(self))
        sample = self.get_sample(idx)
        ret = self._crop_random(sample)
        return ret


def worker_init_fn_random(idx):
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    seed_ = seed + idx
    torch.manual_seed(seed_)
    np.random.seed(seed_)


def main_debug_dataset():
    logging.basicConfig(level=logging.INFO)
    path_idx = '/home/ar/data/yshad-2019/corn_seed_counting_train_v2_clean/idx-val.txt'
    dataset = CountDataset(path_idx=path_idx,
                           in_memory=False,
                           dataset_size=10000000000).build()
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, worker_init_fn=worker_init_fn_random)
    #
    # for xi, x in enumerate(dataset):
    for xi, x in enumerate(dataloader):
        # logging.info('[{}] -> {}'.format(xi, x['idx']))
        # plt.subplot(1, 2, 1)
        # plt.imshow(x['img'])
        # plt.subplot(1, 2, 2)
        # plt.imshow(x['msk'])
        # plt.show()

        print('-')


if __name__ == '__main__':
    main_debug_dataset()