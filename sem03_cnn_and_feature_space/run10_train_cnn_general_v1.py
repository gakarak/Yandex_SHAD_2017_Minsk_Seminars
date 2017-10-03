#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import shutil
import numpy as np
import pandas as pd
import keras.callbacks as kall
import keras.optimizers as kopt
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.utils import np_utils

from run00_common import buildModelCNN_Classification_V1, data_generator
import tensorflow as tf

################################
if __name__ == '__main__':
    numCls      = 2
    imgSiz      = 64
    imgShp      = (imgSiz, imgSiz, 3)
    batchSize   = 256
    numEpochs   = 100
    isDebug     = False
    isLoadIntoMemory = False
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
        print (':: Trained model not found: build new model...')
        model = buildModelCNN_Classification_V1(inpShape=imgShp, numCls=numCls,
                                                numConv=2, numFlt=16, numSubsampling=4, numHidden=128,
                                                isUseDropout=True)
        with tf.name_scope('Keras_Optimizer'):
            modelOptimizer = kopt.Adam(lr=0.0004)
        if numCls == 2:
            model.compile(optimizer=modelOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=modelOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print ('!!! WARNING !!! Found trained model, loading... [{0}]'.format(pathModelRestart))
        pref = time.strftime('%Y.%m.%d-%H.%M.%S')
        pathModelBk = '%s-%s.bk' % (pathModelRestart, pref)
        shutil.copy(pathModelRestart, pathModelBk)
        model = keras.models.load_model(pathModelRestart)
    model.summary()
    if isDebug:
        fimgModel = 'model_general.png'
        keras.utils.plot_model(model, fimgModel, show_shapes=True)
        plt.imshow(plt.imread(fimgModel))
        plt.show()
    #
    numIterPerEpochTrn = int(np.ceil(float(numSamplesTrn) / batchSize))
    numIterPerEpochVal = int(np.ceil(float(numSamplesVal) / batchSize))
    #
    generatorTrn = data_generator(pathCSV=fidxTrn, pimgSize=imgSiz, batchSize=batchSize,
                                  isLoadIntoMemory=isLoadIntoMemory)
    generatorVal = data_generator(pathCSV=fidxVal, pimgSize=imgSiz, batchSize=batchSize,
                                  isLoadIntoMemory=isLoadIntoMemory)
    #
    model.fit_generator(
        generator=generatorTrn,
        steps_per_epoch=numIterPerEpochTrn,
        epochs=numEpochs,
        validation_data=generatorVal,
        validation_steps=numIterPerEpochVal,
              callbacks=[
                  kall.ModelCheckpoint(pathModelValLoss, verbose=True, save_best_only=True, monitor='val_loss'),
                  kall.TensorBoard(log_dir=pathLogDir),
                  kall.CSVLogger(pathLog, append=True)
              ])
    print ('-')

