#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as skio
import skimage.color as skcol
import phcorr as ph


if __name__ == '__main__':
    # pathIdx = '../img/sem02/data_test_frames_ShiftX_0/idx.txt'
    pathIdx = '../img/sem02/data_test_frames_ShiftY_0/idx.txt'
    wdir = os.path.dirname(os.path.abspath(pathIdx))
    pathImgs = np.array([os.path.join(wdir, xx) for xx in pd.read_csv(pathIdx)['path']])
    numImgs = len(pathImgs)

    img1 = skcol.rgb2gray(skio.imread(pathImgs[0]))
    img2 = skcol.rgb2gray(skio.imread(pathImgs[1]))

    CC, dxy, maxVal, pq = ph.phaseCorr(img1, img2)

    img2_shift = np.roll(img2,       int(np.floor(+dxy[0])), 1)
    img2_shift = np.roll(img2_shift, int(np.floor(+dxy[1])), 0)
    # frm2_nrm_shift=np.roll(frm2_nrm_shift, int(math.floor(-dxy[1])), 0)

    print ('-')

    plt.figure(figsize=(16,6))
    plt.subplot(1, 5, 1)
    plt.imshow(img1)
    plt.title('img#1')
    plt.subplot(1, 5, 2)
    plt.imshow(img2)
    plt.title('img#2')
    plt.subplot(1, 5, 3)
    plt.imshow(np.dstack((img1, img2, img1)))
    plt.title('img#1 vs img#2')
    plt.subplot(1, 5, 4)
    plt.imshow(img2_shift)
    plt.title('img#2_shift')
    plt.subplot(1, 5, 5)
    plt.imshow(np.dstack((img1, img2_shift, img1)))
    plt.title('img#1 vs img#2_shift')
    plt.show()


    print ('-')