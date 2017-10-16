#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob

import cv2
import skimage.io as skio
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sift = cv2.xfeatures2d.SIFT_create()
    pathImgs = glob.glob('data/highways/*.jpg')
    numImgs  = len(pathImgs)
    for ii, ipath in enumerate(pathImgs):
        timgC = cv2.imread(ipath, cv2.IMREAD_COLOR)
        timgG = cv2.cvtColor(timgC, cv2.COLOR_BGR2GRAY)
        kp, kd = sift.detectAndCompute(timgG, None)
        img = cv2.drawKeypoints(timgG, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(kd)
        plt.subplot(1, 3, 3)
        plt.plot(kd[:10,:].transpose())
        plt.grid(True)
        print ('[{}/{}] --> {}'.format(ii, numImgs, ipath))
        plt.show()
