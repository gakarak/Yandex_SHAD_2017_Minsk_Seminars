#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as skio
import skimage.color as skcol
import phcorr as ph

###############################
if __name__ == '__main__':
    pathIdx = '../img/sem02/data_test_frames_rot_scl_0/idx.txt'
    wdir = os.path.dirname(os.path.abspath(pathIdx))
    pathImgs = np.array([os.path.join(wdir, xx) for xx in pd.read_csv(pathIdx)['path']])
    numImgs = len(pathImgs)

    img1 = skcol.rgb2gray(skio.imread(pathImgs[0]))
    img2 = skcol.rgb2gray(skio.imread(pathImgs[1]))

    plt.figure(figsize=(12,6))
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title('img#1')
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title('img#2')
    plt.subplot(1, 3, 3)
    plt.imshow(np.dstack((img1, img2, img1)))
    plt.title('img#1 vs img#2')
    # plt.show()

    # CC, dxy, maxVal, pq = ph.phaseCorr(img1, img2)
    dxyS, dAng, dScl, (max_V, pqRS) = ph.imregcorr(img1, img2, isDebug=True)

    print ('-')
