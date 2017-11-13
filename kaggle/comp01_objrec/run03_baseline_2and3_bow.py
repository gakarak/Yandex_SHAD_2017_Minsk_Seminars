#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import pandas as pd
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import sklearn.cluster as skcl
import sklearn.metrics.pairwise as pwd
from sklearn import neighbors, datasets
import cv2

import pickle as pkl

##########################################
def precalc_dsc(pathIdxCSV):
    dataCSV    = pd.read_csv(pathIdxCSV)
    numSamples = len(dataCSV)
    wdir       = os.path.dirname(pathIdxCSV)
    # pathImages = [os.path.join(wdir, xx) for xx in dataCSV['path'][:200]]
    # labels     = dataCSV['class'][:200].as_matrix()
    pathImages = [os.path.join(wdir, xx) for xx in dataCSV['path']]
    labels     = dataCSV['class'].as_matrix()
    #
    tsift = cv2.xfeatures2d.SIFT_create()
    tarrDsc = []
    tarrIdx = []
    tarrLbl = []
    for ppi, (pp, lli) in enumerate(zip(pathImages, labels)):
        timg = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
        _, tdsc = tsift.detectAndCompute(timg, None)
        tdsc = np.sqrt(tdsc)
        tdscMean = np.tile(np.sum(tdsc, axis=-1).reshape((-1, 1)), (1, tdsc.shape[-1]))
        tnumDsc = tdsc.shape[0]
        # tdsc = tdsc.astype(np.float32) / tdsc.sum()
        tarrDsc.append(tdsc/tdscMean)
        tarrIdx.append([ppi] * tnumDsc)
        tarrLbl.append([lli] * tnumDsc)
        if (ppi % 100) == 0:
            print ('\t{}/{}'.format(ppi, numSamples))
    tarrIdx = np.concatenate(tarrIdx).astype(np.int32)
    tarrLbl = np.concatenate(tarrLbl).astype(np.int32)
    tarrDsc = np.concatenate(tarrDsc)
    # print ('-')
    tarrLBL = np.hstack( (tarrIdx.reshape((-1,1)), tarrLbl.reshape((-1,1))))
    return tarrDsc, tarrLBL

def loadDscData(ppathIdx):
    # csvTrn = pd.read_csv(pathIdxTrn)
    # numTrn = len(csvTrn)
    # wdir = os.path.dirname(pathIdxTrn)
    # pathTrn = [os.path.join(wdir, xx) for xx in csvTrn['path']]
    # lblTrn = csvTrn['class'].as_matrix()
    pathIdxDsc = '{}-sift.npy'.format(ppathIdx)
    pathIdxLbl = '{}-lbl.npy'.format(ppathIdx)
    if os.path.isfile(pathIdxDsc):
        # with open(pathIdxDsc, 'rb') as fd, open(pathIdxLbl, 'rb') as fl:
        print ('[*] Loading precalculated desciptors [{}]'.format(pathIdxDsc))
        arrDsc = np.load(pathIdxDsc)
        arrLbl = np.load(pathIdxLbl)
    else:
        print ('[*] Precalculate descriptors:')
        arrDsc, arrLbl = precalc_dsc(ppathIdx)
        # with open(pathIdxDsc, 'wb') as fd, open(pathIdxLbl, 'wb') as fl:
        print ('[*] Save --> {}'.format(pathIdxDsc))
        np.save(pathIdxDsc, arrDsc)
        np.save(pathIdxLbl, arrLbl)
    numImages = len(np.unique(arrLbl[:, 0]))
    return arrDsc, arrLbl, numImages

def calcQDsc(parrDsc, parrLbl, pkmeans):
    tarrDscC = pkmeans.predict(parrDsc)
    numC     = len(np.unique(pkmeans.labels_))
    tmpBins  = list(range(numC))
    lstIdx   = np.sort(np.unique(parrLbl[:, 0]))
    dscq = []
    lblq = []
    for ii, iidx in enumerate(lstIdx):
        tmp = tarrDscC[parrLbl[:, 0]==ii]
        tdsc, _ = np.histogram(tmp, tmpBins)
        # tdsc = np.sqrt(tdsc.astype(np.float32))
        tdsc = tdsc.astype(np.float32)
        tdsc /= np.sum(tdsc)
        dscq.append(tdsc)
        lblq.append(parrLbl[parrLbl[:, 0]==ii, 1][0])
    return dscq, np.array(lblq)
    # print ('-')

##########################################
if __name__ == '__main__':
    pathIdxTrn  = '../data/01_brodatz_dataset/brodatz_dataset_train.csv'
    pathIdxVal  = '../data/01_brodatz_dataset/brodatz_dataset_test.csv'
    #
    arrDscTrn, arrLblTrn, numTrn = loadDscData(pathIdxTrn)
    arrDscVal, arrLblVal, numVal = loadDscData(pathIdxVal)
    #
    paramK        = 4096
    paramN        = 80000
    pathKCls      = '{}-kcls-n{}-k{}.pkl'.format(pathIdxTrn, paramN, paramK)
    pathIdxPred   = '{}-predk-n{}-k{}.csv'.format(pathIdxVal, paramN, paramK)
    #
    arrIdxRnd = np.random.permutation(range(arrDscTrn.shape[0]))[:paramN]
    arrDscRnd = arrDscTrn[arrIdxRnd, :]

    if os.path.isfile(pathKCls):
        print ('[**] Loading precalculated KMeans model [{}]'.format(pathKCls))
        with open(pathKCls, 'rb') as f:
            kmeans = pkl.load(f)
    else:
        print ('[**] Build KMeans model'.format(pathKCls))
        kmeans = skcl.KMeans(n_clusters=paramK, n_jobs=4).fit(arrDscRnd)
        print ('[**] Save KMeans model into [{}]'.format(pathKCls))
        with open(pathKCls, 'wb') as f:
            pkl.dump(kmeans, f)
    sizDsc = arrDscTrn.shape[-1]
    #
    arrQDscTrn, arrQLblTrn = calcQDsc(arrDscTrn, arrLblTrn, kmeans)
    arrQDscVal, arrQLblVal = calcQDsc(arrDscVal, arrLblVal, kmeans)
    # dst = pwd.pairwise_distances(arrDscTrnQ, arrDscTrnQ, 'l1')
    #
    clf = neighbors.KNeighborsClassifier(1, metric='l1')
    clf.fit(arrQDscTrn, arrQLblTrn)
    #
    retTrnQ = clf.predict(arrQDscTrn)
    retValQ = clf.predict(arrQDscVal)
    #
    errTrn = float(np.sum(retTrnQ == arrQLblTrn)) / numTrn
    errVal = float(np.sum(retValQ == arrQLblVal)) / numVal

    print ('Accuracy train:      {:0.2f} %'.format(100. * errTrn))
    print ('Accuracy validation: {:0.2f} %\t * ({})'.format(100. * errVal, os.path.basename(pathIdxVal)))

    csvVal = pd.read_csv(pathIdxVal)
    dataOut = pd.DataFrame(data={
        'path':  csvVal['path'],
        'class': retValQ
    })
    # dataOut.to_csv(pathIdxPred, index=False, columns=['path', 'class'])
    print ('-')

