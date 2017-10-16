#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import shutil
import os
import math
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.color as skcol
import skimage.transform as sktf
import skimage.exposure as skexp
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, merge
from keras.models import Model
import keras.optimizers as kopt
import keras.losses
import keras.callbacks as kall
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model as kplot
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

import tensorflow as tf

from run01_FCN_UNet_train import buildModelUNet, read_img, read_msk, data_generator_simple

################################################
if __name__ == '__main__':
    fidxTrn = '../img/sem04/idx.txt-train.txt'
    fidxVal = '../img/sem04/idx.txt-val.txt'
    numCls  = 5
    imgSiz  = 256
    imgShp  = (imgSiz, imgSiz, 3)
    batchSize = 4
    numSamples = 16
    #
    configGPU = tf.ConfigProto()
    configGPU.gpu_options.per_process_gpu_memory_fraction = 0.6
    keras.backend.tensorflow_backend.set_session(tf.Session(config=configGPU))
    wdirTrn = os.path.dirname(fidxTrn)
    wdirVal = os.path.dirname(fidxVal)
    numSamplesTrn = len(pd.read_csv(fidxTrn))
    numSamplesVal = len(pd.read_csv(fidxVal))
    pathModelPrefix = '{}_CNN'.format(fidxTrn)
    pathLog = '%s-log.csv' % pathModelPrefix
    pathLogDir = '%s-logdir' % pathModelPrefix
    pathModelValLoss = '{0}_valLoss.h5'.format(pathModelPrefix)
    pathModelRestart = pathModelValLoss
    #
    if not os.path.isfile(pathModelRestart):
        raise Exception('Cant find pretrained model [{}]'.format(pathModelRestart))
    model = keras.models.load_model(pathModelRestart)
    model.summary()
    #
    numIterPerEpochTrn = int(np.ceil(float(numSamplesTrn) / batchSize))
    numIterPerEpochVal = int(np.ceil(float(numSamplesVal) / batchSize))
    #
    generatorVal = data_generator_simple(pathCSV=fidxVal, numCls=numCls,
                                         pimgSize=imgSiz, batchSize=numSamples)
    dataX, dataY = next(generatorVal)
    #
    for ii, (datax, datay) in enumerate(zip(dataX, dataY)):
        tsiz = datax.shape[:2]
        tret = model.predict_on_batch(np.expand_dims(datax, axis=0))
        tretProb = tret[0].reshape(list(tsiz) + [tret.shape[-1]])
        tretMsk  = np.argmax(tretProb, axis=-1)
        tmskGT = np.argmax(datay, axis=-1).reshape(tsiz)
        #
        plt.subplot(1, 3, 1)
        plt.imshow( ((datax + 1.0)*127.5).astype(np.uint8) )
        plt.subplot(1, 3, 2)
        plt.imshow( tretMsk )
        plt.subplot(1, 3, 3)
        plt.imshow(tmskGT)
        plt.show()
        print ('[{}/{}]'.format(ii, numSamples))

