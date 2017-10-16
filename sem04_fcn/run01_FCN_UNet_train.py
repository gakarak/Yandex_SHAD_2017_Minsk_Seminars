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

################################################
def buildModelUNet(inpShape=(256, 256, 3), numCls=3, numConv=2, kernelSize=3, numFlt=8, ppad='same', numSubsampling=5, isDebug=False):
    dataInput = Input(shape=inpShape)
    fsiz = (kernelSize, kernelSize)
    psiz = (2, 2)
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2**cc), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        lstMaxPools.append(x)
        x = MaxPooling2D(pool_size=psiz)(x)
    # -------- Decoder --------
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2 ** (numSubsampling - 1 -cc)), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        x = UpSampling2D(size=psiz)(x)
        if cc< (numSubsampling-1):
            x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
        # x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    #
    # 1x1 Convolution: emulation of Dense layer
    if numCls == 2:
        x = Conv2D(filters=1, kernel_size=(1,1), padding='valid', activation='sigmoid')(x)
        x = Flatten()(x)
    else:
        x = Conv2D(filters=numCls, kernel_size=(1, 1), padding='valid')(x)
        x = Reshape([-1, numCls])(x)
        x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    if isDebug:
        retModel.summary()
        fimg_model = 'model_FCN_UNet.png'
        kplot(retModel, fimg_model, show_shapes=True)
        plt.imshow(skio.imread(fimg_model))
        plt.show()
    return retModel

################################################
def read_img(pimg, pimsiz = None):
    if isinstance(pimg, str) or isinstance(pimg, unicode):
        pimg = skio.imread(pimg)
    if pimsiz is not None:
        pimg = sktf.resize(pimg, (pimsiz, pimsiz), order=1, preserve_range=True)
    else:
        pimg = pimg.astype(np.float32)
    if pimg.ndim < 3:
        pimg = skcol.gray2rgb(pimg)
    pimg = pimg/127.5 - 1.0
    return pimg

def read_msk(pmsk, pimsiz = None):
    if isinstance(pmsk, str) or isinstance(pmsk, unicode):
        pmsk = skio.imread(pmsk)
    pmsk[pmsk > 0] -= 128
    if pimsiz is not None:
        pmsk = sktf.resize(pmsk, (pimsiz, pimsiz), order=0, preserve_range=True)
    return pmsk

def data_generator_simple(pathCSV, numCls, batchSize=4, pimgSize=256):
    dataCSV = pd.read_csv(pathCSV)
    numSamples = len(dataCSV)
    wdir = os.path.dirname(pathCSV)
    pathImgs = np.array([os.path.join(wdir, xx) for xx in dataCSV['pathimg']])
    pathMsks = np.array([os.path.join(wdir, xx) for xx in dataCSV['pathmsk_s']])
    idxRange = np.array(range(numSamples))
    while True:
        rndIdx = np.random.permutation(idxRange)[:batchSize]
        dataX = None
        dataY = None
        for ii, iidx in enumerate(rndIdx):
            pimg = pathImgs[iidx]
            pmsk = pathMsks[iidx]
            timg = read_img(pimg, pimgSize)
            tmsk = read_msk(pmsk, pimgSize)
            tmskC = np_utils.to_categorical(tmsk.reshape(-1), num_classes=numCls)
            if dataX is None:
                dataX = np.zeros([batchSize] + list(timg.shape))
                dataY = np.zeros([batchSize] + list(tmskC.shape))
            dataX[ii] = timg
            dataY[ii] = tmskC
        yield dataX, dataY


################################################
if __name__ == '__main__':
    fidxTrn = '../img/sem04/idx.txt-train.txt'
    fidxVal = '../img/sem04/idx.txt-val.txt'
    numCls  = 5
    imgSiz  = 256
    imgShp  = (imgSiz, imgSiz, 3)
    batchSize = 4
    numEpochs = 100
    #
    configGPU = tf.ConfigProto()
    configGPU.gpu_options.per_process_gpu_memory_fraction = 0.8
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
        print (':: Trained model not found: build new model...')
        model = buildModelUNet(inpShape=imgShp, numCls=numCls, isDebug=True)
        with tf.name_scope('Keras_Optimizer'):
            modelOptimizer = kopt.Adam(lr=0.0001)
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
    #
    numIterPerEpochTrn = int(np.ceil(float(numSamplesTrn) / batchSize))
    numIterPerEpochVal = int(np.ceil(float(numSamplesVal) / batchSize))
    #
    generatorTrn = data_generator_simple(pathCSV=fidxTrn, numCls=numCls,
                                         pimgSize=imgSiz, batchSize=batchSize)
    generatorVal = data_generator_simple(pathCSV=fidxVal, numCls=numCls,
                                         pimgSize=imgSiz, batchSize=batchSize)
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

