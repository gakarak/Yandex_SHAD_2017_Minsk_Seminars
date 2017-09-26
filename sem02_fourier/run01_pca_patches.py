#!/usr/bin/python
# -*- coding: utf-8 -*-
from cv2 import imshow

__author__ = 'ar'

import os
import numpy as np
import skimage.io as skio
import skimage.color as skcol
import sklearn.decomposition as sld
import matplotlib.pyplot as plt

################################
def gkern(psiz=32, psigm=3.):
    ax = np.arange(-psiz // 2 + 1., psiz // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * psigm ** 2))
    return kernel / np.sum(kernel)

################################
if __name__ == '__main__':
    # fimg = '../img/lena.png'
    fimg = '../img/doge2.jpg'
    isRandom = False
    if not isRandom:
        # (1) Real image ...
        img  = skcol.rgb2gray(skio.imread(fimg))
    else:
        # (2) ... or random image
        img  = np.random.uniform(-1.0, +1.0, (512, 512))
    imgShp = np.array(img.shape[:2])
    cropSize = 32
    cropShp  = (cropSize, cropSize)
    numSamples = 2000
    img_gauss = gkern(psiz=cropSize, psigm=5.)
    #
    rndRR = np.random.randint(0, imgShp[0] - cropSize - 1, numSamples)
    rndCC = np.random.randint(0, imgShp[0] - cropSize - 1, numSamples)
    data = np.zeros((numSamples, cropSize * cropSize))
    for ii, (rr, cc)  in enumerate(zip(rndRR, rndCC)):
        if isRandom:
            tcrop = img[rr:rr+cropSize, cc:cc+cropSize]
        else:
            tcrop = img[rr:rr+cropSize, cc:cc+cropSize] * img_gauss
        data[ii] = tcrop.reshape(-1)
    pca = sld.PCA()
    pca.fit(data)

    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_))
    plt.title('variances')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.imshow(img_gauss)
    # plt.show()

    plt.figure(figsize=(12,8))
    nx, ny = 8, 8
    nxy = nx * ny
    cnt = 0
    for xx in range(nx):
        for yy in range(ny):
            plt.subplot(nx, ny, cnt + 1)
            tcomp = pca.components_[cnt].reshape(cropShp)
            plt.imshow(tcomp)
            plt.xticks([])
            plt.yticks([])
            plt.title('#{:d} : {:0.2f}'.format(cnt, pca.explained_variance_[cnt]))
            cnt += 1
    plt.show()
    print ('-')

