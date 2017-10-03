#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.utils import np_utils

from run00_common import buildModelCNN_Classification_V1, getCNNFeaturesFun, calcFeatures, draw_features

################################
if __name__ == '__main__':
    numCls      = 10
    imgSiz      = 28
    imgShp      = (imgSiz, imgSiz, 1)
    batchSize   = 256
    numEpochs   = 10
    isDebug     = False
    numSamples  = 1000
    #
    (_, _), (val_X, val_Y) = mnist.load_data()
    val_X = val_X.reshape([val_X.shape[0]] + list(val_X.shape[1:]) + [1])/127.5 - 1.0
    # val_Y = np_utils.to_categorical(val_Y, numCls)
    #
    modelPrefix = 'model_MNIST'
    pathLog     = '%s-log.csv' % modelPrefix
    pathLogDir  = '%s-logdir' % modelPrefix
    pathModelValLoss = '{0}_valLoss.h5'.format(modelPrefix)
    pathModelRestart = pathModelValLoss
    #
    if not os.path.isfile(pathModelRestart):
        raise Exception('Cant find pretrained model [{}]'.format(pathModelRestart))
    # model = buildModelCNN_Classification_V1(inpShape=imgShp, numCls=numCls,
    #                                         numConv=2, numFlt=8, numSubsampling=3, numHidden=128)
    # model.load_weights(pathModelRestart, by_name=True)
    model = keras.models.load_model(pathModelRestart)
    model.summary()
    #
    rndIdx = np.random.permutation(range(val_X.shape[0]))[:numSamples]
    dataX  = val_X[rndIdx]
    dataY  = val_Y[rndIdx]


    lst_names = ['input_1', 'max_pooling2d_1', 'max_pooling2d_2', 'max_pooling2d_3', 'dense_1']
    dict_dsc = {xx:calcFeatures(getCNNFeaturesFun(model, [xx]), dataX).reshape([numSamples,-1]) for xx in lst_names}

    for f_label, f_data in dict_dsc.items():
        print (':: processing label [{}]'.format(f_label))
        draw_features(f_data, dataY, f_label, isShow=False)
    plt.show()
    print ('-')



