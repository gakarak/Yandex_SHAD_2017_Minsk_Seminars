#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.transform as sktf

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tensorflow as tf

################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

################################
def buildModelCNN_Classification_V1(inpShape=(28, 28, 3),
                                    numCls=3, kernelSize=3, numFlt = 8,
                                    numConv=2, numSubsampling=2, ppadding='same', numHidden=None, isUseDropout=False):
    fsiz = (kernelSize, kernelSize)
    psiz = (2, 2)
    with tf.name_scope('SimpleCNN'):
        dataInput = Input(shape=inpShape)
        #
        x = dataInput
        # (1) Conv-layers
        for cc in range(numSubsampling):
            with tf.name_scope('conv_block_{}'.format(cc)):
                if cc==0:
                    tfsiz = (3, 3)
                else:
                    tfsiz = fsiz
                for ii in range(numConv):
                    x = Conv2D(filters=numFlt * (2 **cc), kernel_size=tfsiz,
                               activation='relu', padding=ppadding)(x)
                x = MaxPooling2D(pool_size=psiz, padding=ppadding)(x)
        # (2) flatening
        x = Flatten()(x)
        # (3) hidden dense-layers
        if numHidden is not None:
            if isinstance(numHidden, list):
                for numUnits in numHidden:
                    x = Dense(units=numUnits, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

            else:
                x = Dense(units=numHidden, activation='relu')(x)
            if isUseDropout:
                x = Dropout(rate=0.5)(x)
        # (4) multiclass-output
        if numCls == 2:
            x = Dense(units=1, activation='sigmoid')(x)
        else:
            x = Dense(units=numCls, activation='softmax')(x)
        retModel = Model(inputs=dataInput, outputs=x, name='cnn_model')
    return retModel

def getCNNFeaturesFun(p_model, p_lst_layers):
    featuresOut = []
    all_layers_names = [xx.name for xx in p_model.layers]
    for lli, ll_name in enumerate(p_lst_layers):
        if ll_name in all_layers_names:
            featuresOut.append(p_model.get_layer(ll_name).output)
        else:
            raise Exception('Cant find layer [{}] in model!!'.format(ll_name))
    tfeaturesNN = K.function([p_model.layers[0].input], featuresOut)
    return tfeaturesNN

def calcFeatures(p_dscfun, p_data, p_batchSize=32):
    idx_range = list(range(p_data.shape[0]))
    idx_split = split_list_by_blocks(idx_range, p_batchSize)
    ret = []
    for ssi, ss in enumerate(idx_split):
        ret.append(p_dscfun([p_data[ss]])[0])
        print ('[{}/{}]'.format(ssi*p_batchSize, len(idx_range)))
    print ('-')
    return np.vstack(ret)

################################
def draw_features(p_dataX, p_data_Y, p_label, isShow=True):
    plt.figure()
    # (1) PCA
    pca = PCA(n_components=2)
    p_dataX_PCA = pca.fit_transform(p_dataX)
    plt.subplot(1, 2, 1)
    tmp = []
    for ii in np.unique(p_data_Y):
        tmp.append(plt.scatter(p_dataX_PCA[p_data_Y == ii, 0], p_dataX_PCA[p_data_Y == ii, 1],
                               label=p_data_Y[p_data_Y == ii]))
    plt.legend(tmp, ['csl_%d' % xx for xx in np.unique(p_data_Y)])
    plt.grid(True)
    plt.title('PCA: {}'.format(p_label))
    #
    # (2) t-SNE
    tsne = TSNE(n_components=2, init='pca')
    p_dataX_TSNE = tsne.fit_transform(p_dataX)
    plt.subplot(1, 2, 2)
    for ii in np.unique(p_data_Y):
        tmp.append(plt.scatter(p_dataX_TSNE[p_data_Y == ii, 0], p_dataX_TSNE[p_data_Y == ii, 1],
                               label=p_data_Y[p_data_Y == ii]))
    plt.legend(tmp, ['cls_%d' % xx for xx in np.unique(p_data_Y)])
    plt.grid(True)
    plt.title('t-SNE: {}'.format(p_label))
    if isShow:
        plt.show()

################################
def getImage(pimg, pimsiz=256):
    if isinstance(pimg, str) or isinstance(pimg, unicode):
        pimg = skio.imread(pimg)
    pimg = pimg.astype(np.float32)/127.5 - 1.
    dstSize = (pimsiz, pimsiz)
    if pimg.shape[:2] == dstSize:
        pass
    else:
        pimg = sktf.resize(pimg, dstSize, order=2)
    if pimg.ndim<3:
        return np.expand_dims(pimg, axis=-1)
    else:
        return pimg

def data_generator(pathCSV, pimgSize=256, batchSize=8, isLoadIntoMemory=False):
    wdir = os.path.dirname(pathCSV)
    dataCSV = pd.read_csv(pathCSV)
    dataAllY = dataCSV['cls'].as_matrix().astype(np.float32)
    pathImgs = np.array([os.path.join(wdir, xx) for xx in dataCSV['path']])
    numImgs = len(pathImgs)
    dataAllX = None
    if isLoadIntoMemory:
        print (':: loading data into memory... [{}]'.format(os.path.basename(pathCSV)))
        for ii, pimg in enumerate(pathImgs):
            timg = getImage(pimg, pimsiz=pimgSize)
            if dataAllX is None:
                dataAllX = np.zeros([numImgs] + list(timg.shape), dtype=np.float32)
            dataAllX[ii] = timg
            if (ii % 100) == 0:
                print ('\t[{}/{}] ... {}'.format(ii, numImgs, os.path.basename(pimg)))
    #
    arrIdx = np.array(range(numImgs))
    while True:
        rndIdx = np.random.permutation(arrIdx)[:batchSize]
        dataY = dataAllY[rndIdx].copy()
        if isLoadIntoMemory:
            dataX = dataAllX[rndIdx].copy()
        else:
            dataX = None
            for ii, iidx in enumerate(rndIdx):
                pimg = pathImgs[iidx]
                timg = getImage(pimg, pimsiz=pimgSize)
                if dataX is None:
                    dataX = np.zeros([batchSize] + list(timg.shape))
                dataX[ii] = timg
        yield dataX, dataY

################################
if __name__ == '__main__':
    pass