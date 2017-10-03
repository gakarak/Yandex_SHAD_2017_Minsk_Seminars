#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

import tensorflow as tf

from run00_common import getCNNFeaturesFun, calcFeatures, draw_features, data_generator

################################
if __name__ == '__main__':
    numCls      = 2
    imgSiz      = 64
    imgShp      = (imgSiz, imgSiz, 3)
    batchSize   = 256
    numEpochs   = 10
    isDebug     = False
    numSamples  = 1000
    #
    configGPU = tf.ConfigProto()
    configGPU.gpu_options.per_process_gpu_memory_fraction = 0.6
    keras.backend.tensorflow_backend.set_session(tf.Session(config=configGPU))
    #
    fidxTrn = '../img/sem03/cats-vs-dogs/size_64x64/idx.txt-train.txt'
    fidxVal = '../img/sem03/cats-vs-dogs/size_64x64/idx.txt-val.txt'
    wdirTrn = os.path.dirname(fidxTrn)
    wdirVal = os.path.dirname(fidxVal)
    numSamplesTrn = len(pd.read_csv(fidxTrn))
    numSamplesVal = len(pd.read_csv(fidxVal))
    #
    pathModelPrefix = '{}_CNN'.format(fidxTrn)
    pathLog = '%s-log.csv' % pathModelPrefix
    pathLogDir = '%s-logdir' % pathModelPrefix
    pathModelValLoss = '{0}_valLoss.h5'.format(pathModelPrefix)
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
    generatorVal = data_generator(pathCSV=fidxVal, pimgSize=imgSiz, batchSize=numSamples, isLoadIntoMemory=False)
    dataX, dataY = next(generatorVal)
    #
    lst_names = ['input_1', 'max_pooling2d_1', 'max_pooling2d_2', 'flatten_1']
    dict_dsc = {xx: calcFeatures(getCNNFeaturesFun(model, [xx]), dataX).reshape([numSamples, -1]) for xx in lst_names}

    for f_label, f_data in dict_dsc.items():
        print (':: processing label [{}]'.format(f_label))
        draw_features(f_data, dataY, f_label, isShow=False)
    plt.show()

    print ('-')



