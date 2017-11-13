#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt
import sklearn.cluster as skcl
import sklearn.metrics.pairwise as pwd
from sklearn import neighbors, datasets

import pickle as pkl

###########################################
def getIndexPQ(psizDsc, psizQ = 16, psizD = 32, pdir=None):
    if psizDsc < psizD:
        raise Exception('[!] Total descriptor length must been more than size of one Q-Part: {} < {}'
                        .format(psizDsc, psizD))
    if pdir is None:
        pdir = os.path.abspath('.')
    fidxBase = 'pqidx_s{}_q{}_d{}.txt'.format(psizDsc, psizQ, psizD)
    fidx = os.path.join(pdir, fidxBase)
    if os.path.isfile(fidx):
        print ('[*] Loading PQ-Index from file [{}]'.format(fidx))
        retIdx = pd.read_csv(fidx, header=None).as_matrix()
        return retIdx
    else:
        lstIdx = list(range(psizDsc))
        retIdx = []
        for ddi in range(psizQ):
            tidx = np.random.permutation(lstIdx)[:psizD]
            retIdx.append(tidx)
        retIdx = np.array(retIdx)
        print (' [*] crate PQ-Index and save to [{}]'.format(fidx))
        pdfIdx = pd.DataFrame(retIdx, index=None, columns=None)
        pdfIdx.to_csv(fidx, header=False, index=False)
        return retIdx

###########################################
def precalculateDscFP(pathIdxCSV):
    dataCSV = pd.read_csv(pathIdxCSV)
    numSamples = len(dataCSV)
    wdir = os.path.dirname(pathIdxCSV)
    pathImages = [os.path.join(wdir, xx) for xx in dataCSV['path']]
    labels = dataCSV['class'].as_matrix()
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
        tarrDsc.append(tdsc / tdscMean)
        tarrIdx.append([ppi] * tnumDsc)
        tarrLbl.append([lli] * tnumDsc)
        if (ppi % 100) == 0:
            print ('\t{}/{}'.format(ppi, numSamples))
    tarrIdx = np.concatenate(tarrIdx).astype(np.int32)
    tarrLbl = np.concatenate(tarrLbl).astype(np.int32)
    tarrDsc = np.concatenate(tarrDsc)
    tarrLBL = np.hstack((tarrIdx.reshape((-1, 1)), tarrLbl.reshape((-1, 1))))
    return tarrDsc, tarrLBL

def loadDscData(ppathIdx):
    pathIdxDsc = '{}-sift.npy'.format(ppathIdx)
    pathIdxLbl = '{}-lbl.npy'.format(ppathIdx)
    if os.path.isfile(pathIdxDsc):
        # with open(pathIdxDsc, 'rb') as fd, open(pathIdxLbl, 'rb') as fl:
        print ('[*] Loading precalculated desciptors [{}]'.format(pathIdxDsc))
        arrDsc = np.load(pathIdxDsc)
        arrLbl = np.load(pathIdxLbl)
    else:
        print ('[*] Precalculating descriptors:')
        arrDsc, arrLbl = precalculateDscFP(ppathIdx)
        print ('[*] Save --> {}'.format(pathIdxDsc))
        np.save(pathIdxDsc, arrDsc)
        np.save(pathIdxLbl, arrLbl)
    numImages = len(np.unique(arrLbl[:, 0]))
    return arrDsc, arrLbl, numImages

###########################################
def loadKmeansClsPQ(pfidx, psizeQ, psizeD, pnumQK, pnumSmpl = 10000):
    wdir = os.path.dirname(pfidx)
    pathKCls = '{}-kclsPQ-n{}-q{}-d{}-k{}.pkl'.format(pfidx, pnumSmpl, psizeQ, psizeD, pnumQK)
    if os.path.isfile(pathKCls):
        print ('[**] Found precalcaulated PQ-KMeans-Cls, loading... [{}]'.format(pathKCls))
        with open(pathKCls, 'rb') as f:
            tdata = pkl.load(f)
        return tdata
    else:
        print ('[**] Create PQ-KMeans-Cls')
        arrDscFP, arrLblFP, numFP = loadDscData(pathIdxTrn)
        numSmplMax = arrDscFP.shape[0]
        if pnumSmpl > numSmplMax:
            pnumSmpl = numSmplMax
        sizDSC  = arrDscFP.shape[-1]
        idxAll  = list(range(arrDscFP.shape[0]))
        idxPQ   = getIndexPQ(sizDSC, psizQ=psizeQ, psizD=psizeD, pdir=wdir)
        arrKCls = []
        for qqi in range(psizeQ):
            tidxRnd  = np.random.permutation(idxAll)[:pnumSmpl]
            tarrQDsc = arrDscFP[tidxRnd, :]
            tarrQDsc = tarrQDsc [:, idxPQ[qqi]]
            tkmeans  = skcl.KMeans(n_clusters=pnumQK, n_jobs=4).fit(tarrQDsc)
            arrKCls.append(tkmeans)
            print ('\t[{}/{}] build Q-Kmeans cld'.format(qqi, psizeQ))
        #
        with open(pathKCls, 'wb') as f:
            tdata = {
                'pqkcls': arrKCls,
                'pqidx':  idxPQ,
            }
            pkl.dump(tdata, f)
        return tdata

def getQDscFromArrFP(arrFP, idxPQ, lstKcls):
    numQ, sizD = idxPQ.shape
    assert (numQ == len(lstKcls))
    numK = lstKcls[0].n_clusters
    tmpKbin = list(range(numK+1))
    dscIndexed = []
    for ii, ikcls in enumerate(lstKcls):
        tdsc = arrFP[:, idxPQ[ii]]
        tdscIdx = ikcls.predict(tdsc)
        tdscSum = np.histogram(tdscIdx, bins=tmpKbin)[0]
        dscIndexed.append(tdscSum)
    dscIndexed = np.array(dscIndexed)
    dscIndexed = dscIndexed.reshape(-1).astype(np.float32)
    dscIndexed /= dscIndexed.sum()
    return dscIndexed

def precalculateDscPQK(pfidx, psizeQ, psizeD, psizeK, pnumSmpl = 10000, pfidxKClsPQ = None):
    wdir = os.path.dirname(pfidx)
    pathDscPQK = '{}-dscpqk-n{}-q{}-d{}-k{}.npy'.format(pfidx, pnumSmpl, psizeQ, psizeD, psizeK)
    # (1) load data PQ-Data
    #FIXME : check this point
    # pathKCls = '{}-kclsPQ-n{}-q{}-d{}-k{}.pkl'.format(pfidx, pnumSmpl, psizeQ, psizeD, pnumQK)
    # if os.path.isfile(pathKCls):
        # raise Exception('Cant find PQ-KMeans-Cls file [{}]'.format(pathKCls))
    arrDscFP, arrLblFP, numFP = loadDscData(pfidx)
    arrImgIdx = np.unique(arrLblFP[:, 0])
    arrLbl = np.array([arrLblFP[arrLblFP[:,0]==xx, 1][0] for xx in arrImgIdx])
    if os.path.isfile(pathDscPQK):
        print (':: loadind Dsc-PQK from file [{}]'.format(pathDscPQK))
        arrDscPQ = np.load(pathDscPQK)
    else:
        print (':: precalculation Dsc-PQK...')
        if pfidxKClsPQ is None:
            dataPQKCls = loadKmeansClsPQ(pfidx, psizeQ=psizeQ, psizeD=psizeD, pnumQK=psizeK, pnumSmpl=pnumSmpl)
        else:
            dataPQKCls = loadKmeansClsPQ(pfidxKClsPQ, psizeQ=psizeQ, psizeD=psizeD, pnumQK=psizeK, pnumSmpl=pnumSmpl)
        lstPQKcls = dataPQKCls['pqkcls']
        idxPQ     = dataPQKCls['pqidx']
        numDscFP  = arrDscFP.shape[0]
        # (2) build quantized FQ-descriptors
        arrDscPQ = []
        for ii, imgIdx in enumerate(arrImgIdx):
            tarrImgDsc = arrDscFP[arrLblFP[:, 0] == imgIdx]
            tdscQImg = getQDscFromArrFP(tarrImgDsc, idxPQ, lstPQKcls)
            arrDscPQ.append(tdscQImg)
            if (ii%100)==0:
                print ('\t\t[{}/{}]'.format(ii, len(arrImgIdx)))
        print ('\t:: save precalculated Dsc-PQK to file: [{}]'.format(pathDscPQK))
        arrDscPQ = np.array(arrDscPQ)
        np.save(pathDscPQK, arrDscPQ)
    return arrDscPQ, arrLbl

###########################################
if __name__ == '__main__':
    pathIdxTrn = '/home/ar/data/brodatz/brodatz_dataset_train.csv'
    pathIdxVal = '/home/ar/data/brodatz/brodatz_dataset_test.csv'
    #
    # paramSizeQ  = 32
    # paramSizeD  = 16
    # paramNumQK  = 64
    #
    # paramSizeQ = 256
    # paramSizeD = 8
    # paramNumQK = 32
    #
    paramSizeQ = 64
    paramSizeD = 12
    paramNumQK = 128
    #
    # paramSizeQ = 32
    # paramSizeD = 12
    # paramNumQK = 256

    paramN      = 40000
    #
    dscTrn, lblTrn = precalculateDscPQK(pathIdxTrn,
                                psizeQ=paramSizeQ, psizeD=paramSizeD, psizeK=paramNumQK, pnumSmpl=paramN)
    dscVal, lblVal = precalculateDscPQK(pathIdxVal,
                                psizeQ=paramSizeQ, psizeD=paramSizeD, psizeK=paramNumQK, pnumSmpl=paramN,
                                pfidxKClsPQ=pathIdxTrn)
    # lblTrnD = np.load('{}-lbl.npy'.format(pathIdxTrn))
    # lblValD = np.load('{}-lbl.npy'.format(pathIdxVal))
    # lblTrn  = np.unique(lblTrnD[:,0])
    # lblVal  = np.unique(lblValD[:,0])
    #
    clf = neighbors.KNeighborsClassifier(1, metric='l1')
    clf.fit(dscTrn, lblTrn)
    #
    numTrn, numVal = len(lblTrn), len(lblVal)
    lblTrnPred = clf.predict(dscTrn)
    lblValPred = clf.predict(dscVal)
    errTrn = float(np.sum(lblTrnPred == lblTrn)) / numTrn
    errVal = float(np.sum(lblValPred == lblVal)) / numVal
    print ('Accuracy train:      {:0.2f} %'.format(100. * errTrn))
    print ('Accuracy validation: {:0.2f} %\t * ({})'.format(100. * errVal, os.path.basename(pathIdxVal)))

    print ('-')


