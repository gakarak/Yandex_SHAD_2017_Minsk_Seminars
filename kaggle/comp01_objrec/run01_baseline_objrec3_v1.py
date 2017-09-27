#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import pandas as pd
import numpy as np
import skimage.io as skio

from sklearn import neighbors, datasets

##############################
def calc_dsc(ppathImg):
    tmpBins = list(range(256))
    numSamples = len(ppathImg)
    arrDsc = []
    for ppi, pp in enumerate(ppathImg):
        timg = skio.imread(pp)
        tdsc, _ = np.histogram(timg, tmpBins)
        tdsc = tdsc.astype(np.float32) / tdsc.sum()
        arrDsc.append(tdsc)
        if (ppi % 100) == 0:
            print ('\t{}/{}'.format(ppi, numSamples))
    arrDsc = np.array(arrDsc)
    return arrDsc

#############################
if __name__ == '__main__':

    pathIdxTrn = '../data/01_brodatz_dataset/brodatz_dataset_train.csv'
    pathIdxVal = '../data/01_brodatz_dataset/brodatz_dataset_test_submit.csv'
    pathIdxPred = '{}-pred.csv'.format(pathIdxVal)
    csvTrn = pd.read_csv(pathIdxTrn)
    csvVal = pd.read_csv(pathIdxVal)
    numTrn = len(csvTrn)
    numVal = len(csvVal)
    #
    wdir = os.path.dirname(pathIdxTrn)
    pathTrn = [os.path.join(wdir, xx) for xx in csvTrn['path']]
    pathVal = [os.path.join(wdir, xx) for xx in csvVal['path']]
    #
    lblTrn = csvTrn['class'].as_matrix()
    lblVal = csvVal['class'].as_matrix()
    #
    arrDscTrn = calc_dsc(pathTrn)
    arrDscVal = calc_dsc(pathVal)

    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(arrDscTrn, lblTrn)

    retTrn = clf.predict(arrDscTrn)
    retVal = clf.predict(arrDscVal)

    errTrn = float(np.sum(retTrn == lblTrn)) / numTrn
    # errVal = float(np.sum(retVal == lblVal)) / numVal

    print ('Accuracy train:      {:0.2f} %'.format(100. * errTrn))
    # print ('Accuracy validation: {:0.2f} %\t * ({})'.format(100. * errVal, os.path.basename(pathIdxVal)))

    dataOut = pd.DataFrame(data={
        'path': csvVal['path'],
        'class': retVal
    })
    dataOut.to_csv(pathIdxPred, index=False, columns=['path', 'class'])

    print (':: Test-prediction --> [{}]'.format(pathIdxPred))
