#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import shutil
import numpy as np
import keras.callbacks as kall
import keras.optimizers as kopt
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.utils import np_utils

from run00_common import buildModelCNN_Classification_V1
import tensorflow as tf

################################
if __name__ == '__main__':
    numCls      = 10
    imgSiz      = 28
    imgShp      = (imgSiz, imgSiz, 1)
    batchSize   = 256
    numEpochs   = 10
    isDebug     = False
    #
    (trn_X, trn_Y), (val_X, val_Y) = mnist.load_data()
    trn_X = trn_X.reshape([trn_X.shape[0]] + list(trn_X.shape[1:]) + [1])/127.5 - 1.0
    val_X = val_X.reshape([val_X.shape[0]] + list(val_X.shape[1:]) + [1])/127.5 - 1.0
    trn_Y = np_utils.to_categorical(trn_Y, numCls)
    val_Y = np_utils.to_categorical(val_Y, numCls)
    #
    modelPrefix = 'model_MNIST'
    pathLog     = '%s-log.csv' % modelPrefix
    pathLogDir  = '%s-logdir' % modelPrefix
    pathModelValLoss = '{0}_valLoss.h5'.format(modelPrefix)
    pathModelRestart = pathModelValLoss
    #
    if not os.path.isfile(pathModelRestart):
        print (':: Trained model not found: build new model...')
        model = buildModelCNN_Classification_V1(inpShape=imgShp, numCls=numCls,
                                                numConv=2, numFlt=8, numSubsampling=3, numHidden=128)
        with tf.name_scope('Keras_Optimizer'):
            modelOptimizer = kopt.Adam(lr=0.001)
        if  numCls == 2:
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
        fimgModel = 'model_mnist.png'
        keras.utils.plot_model(model, fimgModel, show_shapes=True)
        plt.imshow(plt.imread(fimgModel))
        plt.show()
    #
    model.fit(x=trn_X, y=trn_Y,
              batch_size=batchSize,
              epochs=numEpochs,
              validation_data=(val_X, val_Y),
              callbacks=[
                  kall.ModelCheckpoint(pathModelValLoss, verbose=True, save_best_only=True, monitor='val_loss'),
                  kall.TensorBoard(log_dir=pathLogDir),
                  kall.CSVLogger(pathLog, append=True)
              ])
    print ('-')

