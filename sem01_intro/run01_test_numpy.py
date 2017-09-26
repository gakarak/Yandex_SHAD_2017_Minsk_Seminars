#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

###############################
if __name__ == '__main__':
    fimg = '../img/doge2.jpg'
    img  = skio.imread(fimg)
    #
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(img[60:200,60:200,:])
    plt.subplot(2, 2, 3)
    plt.imshow(img[:, :, 0] + img[:, :, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(1.*img[:, :, 0] + img[:, :, 1])
    plt.show()