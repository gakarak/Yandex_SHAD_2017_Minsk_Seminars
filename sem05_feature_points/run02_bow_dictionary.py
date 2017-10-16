#!/usr/bin/python
__author__ = 'ar'

import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage as sk
import skimage.io
import skimage.color
from sklearn import cluster
from sklearn.metrics import pairwise as pwd

import os
import sys
import glob

######################################
def cropImagePatch(img, pt, angle, radiusPatch, isDebug=False, baseSize=(24,24)):
    px = pt[0]
    py = pt[1]
    #
    imgShape=img.shape
    imgCenter=(int(imgShape[1]/2.0), int(imgShape[0]/2.0))
    #
    matRot=cv2.getRotationMatrix2D(imgCenter, angle, 1.0)
    matShift=np.zeros((2,3))
    matShift[0,0]=1.0
    matShift[0,1]=0.0
    matShift[1,0]=0.0
    matShift[1,1]=1.0
    matShift[0,2]=+imgCenter[0] - px
    matShift[1,2]=+imgCenter[1] - py
    imgShift=cv2.warpAffine(img, matShift, (imgShape[1], imgShape[0]), None, cv2.INTER_CUBIC)
    cropSize=int(np.round(1.0*radiusPatch))
    imgShiftRot=cv2.warpAffine(imgShift, matRot, (imgShape[1], imgShape[0]), None, cv2.INTER_CUBIC)
    imgShiftRotCrop=imgShiftRot[imgCenter[1]-cropSize:imgCenter[1]+cropSize, imgCenter[0]-cropSize:imgCenter[0]+cropSize]
    if isDebug:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(imgShift), plt.title('Shift')
        plt.subplot(1,3,2)
        plt.imshow(imgShiftRot), plt.title('Shift-Rot')
        plt.subplot(1,3,3)
        plt.imshow(imgShiftRotCrop), plt.title('Shift-Rot-Crop')
    brd=1
    baseSizeBrd=(baseSize[0]-brd*2, baseSize[1]-brd*2)
    imgShiftRotCropResize=cv2.resize(imgShiftRotCrop, baseSizeBrd, None,0,0,cv2.INTER_CUBIC)
    imgShiftRotCropResize=cv2.copyMakeBorder(imgShiftRotCropResize,brd,brd,brd,brd,cv2.BORDER_CONSTANT, None, (0,0,0))
    return imgShiftRotCropResize

######################################
def_NumFeaturesPerImage=100
def_NumClustersSqrt=10
def_NumClusters=def_NumClustersSqrt**2
def_numBestPatches=int(def_NumClusters*3/4)
def_sizePatch=32

######################################
if __name__=='__main__':
    lstF=glob.glob('data/highways/*.jpg')
    # sift=cv2.SIFT(nfeatures=def_NumFeaturesPerImage)
    sift=cv2.xfeatures2d.SIFT_create(nfeatures=def_NumFeaturesPerImage)
    lstK=[]
    lstD=[]
    lstImg=[]
    arrkp=None
    arrkd=None
    cnt=0
    print '(1) Prepare data:'
    for ii in xrange(len(lstF)):
        fimg=lstF[ii]
        img=cv2.imread(fimg, cv2.IMREAD_GRAYSCALE)
        lstImg.append(img)
        kp, kd=sift.detectAndCompute(img,None)
        kpa=np.zeros((len(kp), 5))
        for kki in xrange(len(kp)):
            tkp=kp[kki]
            kpa[kki,0]=tkp.pt[0]
            kpa[kki,1]=tkp.pt[1]
            kpa[kki,2]=tkp.size
            kpa[kki,3]=tkp.angle
        kpa[:,4]=ii
        if arrkd is None:
            arrkp=kpa.copy()
            arrkd=kd.copy()
        else:
            # kd /= np.tile(np.sum(kd, axis=-1).reshape((-1, 1)), (1, kd.shape[-1]))
            arrkd=np.concatenate([arrkd,kd])
            arrkp=np.concatenate([arrkp,kpa])
        lstK.append(kp)
        lstD.append(kd)
        if not (ii%20):
            print '%d/%d' % (ii,len(lstF))
    #
    print '(2) Run K-Means:'
    mdlKMeans=cluster.KMeans(n_clusters=def_NumClusters, max_iter=10, n_jobs=-1)
    arrCls=mdlKMeans.fit_predict(arrkd)
    print '(3) Calculate scores:'
    arrScores_DST = pwd.pairwise_distances(arrkd, mdlKMeans.cluster_centers_, 'l2')
    arrScores_DST = np.sort(arrScores_DST)[:,0]
    # arrScores=np.zeros(len(arrCls))
    # q1 = mdlKMeans.score(arrkd)
    # for ii in xrange(len(arrCls)):
    #     arrScores[ii]=-mdlKMeans.score(arrkd[ii, :].reshape((1, -1)))
    arrScores = arrScores_DST
    #
    print '(4) Visualize patches:'
    lstMeanCls=[]
    imgBig=np.zeros((def_sizePatch*def_numBestPatches, def_sizePatch*def_NumClusters))
    arrClsUniq=np.unique(arrCls)
    lstMeanPatch=[]
    for cci in arrClsUniq:
        idxc=(arrCls==cci)
        tarrkp=arrkp[idxc,:]
        tarrkd=arrkd[idxc,:]
        tscors=arrScores[idxc]
        idxGood=np.argsort(tscors)
        tnumBestPatches=np.min((len(idxGood),def_numBestPatches))
        tMeanPatch=np.zeros((def_sizePatch,def_sizePatch))
        for bbi in xrange(tnumBestPatches):
            tidx=idxGood[bbi]
            tpts=(tarrkp[tidx,0], tarrkp[tidx,1])
            tsiz=tarrkp[tidx,2]*5.0/2.0
            tang=tarrkp[tidx,3]
            tidxImg=int(tarrkp[tidx,4])
            timgPatch=cropImagePatch(lstImg[tidxImg], tpts,tang,tsiz,baseSize=(def_sizePatch,def_sizePatch))
            imgBig[bbi*def_sizePatch:bbi*def_sizePatch+timgPatch.shape[0], cci*def_sizePatch:cci*def_sizePatch+timgPatch.shape[1]]=timgPatch
            tMeanPatch+=timgPatch.astype(np.float)
        if tnumBestPatches>0:
            tMeanPatch/=tnumBestPatches
        lstMeanPatch.append(tMeanPatch)
        print '%d/%d' % (cci,len(arrClsUniq))
    plt.imshow(imgBig, cmap=plt.gray())
    imgBigNorm=cv2.normalize(imgBig, None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite("VisualDictionary.png", imgBigNorm)
    plt.show()
