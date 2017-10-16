#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import cv2
import skimage.io as skio
import skimage.transform as sktf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################################################
def buildImageWithRotScaleAroundCenter(pimg, pcnt, pangDec, pscale, pcropSize, isDebug=False, pborderMode = cv2.BORDER_REPLICATE):
    # (1) precalc parameters
    angRad = (np.pi / 180.) * pangDec
    cosa = np.cos(angRad)
    sina = np.sin(angRad)
    # (2) prepare separate affine transformation matrices
    matShiftB = np.array([[1., 0., -pcnt[0]], [0., 1., -pcnt[1]], [0., 0., 1.]])
    matRot = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
    matShiftF = np.array([[1., 0., +pcnt[0]], [0., 1., +pcnt[1]], [0., 0., 1.]])
    matScale = np.array([[pscale, 0., 0.], [0., pscale, 0.], [0., 0., 1.]])
    matShiftCrop = np.array([[1., 0., pcropSize[0] / 2.], [0., 1., pcropSize[1] / 2.], [0., 0., 1.]])
    # matTotal_OCV = matShiftF.dot(matRot.dot(matScale.dot(matShiftB)))
    # (3) build total-matrix
    matTotal = matShiftCrop.dot(matRot.dot(matScale.dot(matShiftB)))
    if isDebug:
        print ('(1) mat-shift-backward = \n{0}'.format(matShiftB))
        print ('(2) mat-scale = \n{0}'.format(matScale))
        print ('(3) mat-rot = \n{0}'.format(matRot))
        print ('(4) mat-shift-forward = \n{0}'.format(matShiftF))
        print ('(5) mat-shift-crop = \n{0}'.format(matShiftCrop))
        print ('---\n(*) mat-total = \n{0}'.format(matTotal))
    # (4) warp image with total affine-transform
    if (pimg.ndim>2) and (pimg.shape[-1]>3):
        pimg0 = pimg[:, :, :3]
        pimg1 = pimg[:, :, -1]
        imgRet0 = cv2.warpAffine(pimg0, matTotal[:2, :], pcropSize, flags=cv2.INTER_CUBIC, borderMode=pborderMode)
        imgRet1 = cv2.warpAffine(pimg1, matTotal[:2, :], pcropSize, flags=cv2.INTER_NEAREST, borderMode=pborderMode)
        imgRet = np.dstack((imgRet0, imgRet1))
    else:
        imgRet = cv2.warpAffine(pimg, matTotal[:2, :], pcropSize, borderMode=pborderMode)
    return imgRet

#################################################
def getRandomInRange(vrange, pnum=None):
    vmin,vmax = vrange
    if pnum is None:
        trnd = np.random.rand()
    else:
        trnd = np.random.rand(pnum)
    ret = vmin + (vmax-vmin)*trnd
    return ret

#################################################
def generateRandomizedScene(pathBG, pathIdxFG, newSize=None, p_rangeSizes=(64, 256), p_rangeAngle=(0, 36), p_rangeNumSamples=(3, 16), isDebug=False):
    imgBG = skio.imread(pathBG)
    if newSize is not None:
        if np.min(imgBG.shape[:2]) < (np.max(newSize) + 5):
            imgBG = sktf.resize(imgBG, (512, 512))
        else:
            rndRR = np.random.randint(0, imgBG.shape[0] - newSize[0] - 2)
            rndCC = np.random.randint(0, imgBG.shape[1] - newSize[1] - 2)
            if imgBG.ndim>2:
                imgBG = imgBG[rndRR:rndRR + newSize[0], rndCC:rndCC + newSize[1], :].copy()
            else:
                imgBG = imgBG[rndRR:rndRR + newSize[0], rndCC:rndCC + newSize[1]].copy()
    newImgBG = imgBG.copy()
    sizBG = imgBG.shape[:2]
    retMskBG_S = np.zeros(imgBG.shape[:2])
    retMskBG_I = np.zeros(imgBG.shape[:2])
    #
    wdirFG = os.path.dirname(pathIdxFG)
    dataFG = pd.read_csv(fidxFG, sep=',')
    pathImgsFG = dataFG['path'].as_matrix()
    pathImgsFG = [os.path.join(wdirFG, xx) for xx in pathImgsFG]
    arrClsFG = dataFG['clsid'].as_matrix()
    numImgsFG = len(dataFG)
    #
    numSamples = np.random.randint(p_rangeNumSamples[0], p_rangeNumSamples[1])
    cntInstance = 1
    rndIdx = np.random.randint(0, numImgsFG, numSamples)
    retBBoxCls = []
    for iidx, idx in enumerate(rndIdx):
        pathFG = pathImgsFG[idx]
        imgFG = skio.imread(pathFG)
        (mskPC, mskR) = cv2.minEnclosingCircle(np.array(np.where(imgFG[:,:,-1] > 128)).transpose())
        rndSiz = int(getRandomInRange(def_range_sizes))
        rndAng = getRandomInRange(def_range_angle)
        pScale = float(rndSiz) / (2. * mskR)
        newImg = buildImageWithRotScaleAroundCenter(imgFG, mskPC[::-1], rndAng, pScale, (rndSiz, rndSiz))
        newCls = arrClsFG[idx]
        newMsk = (newImg[:, :, -1]>128).astype(np.int)
        newMsk[newMsk>0] = newCls
        # (1) generate temporary mask
        rndRR = np.random.randint(0, sizBG[0] - newMsk.shape[0] - 1)
        rndCC = np.random.randint(0, sizBG[1] - newMsk.shape[1] - 1)
        tmpMsk = np.zeros(imgBG.shape[:2])
        tmpMsk[rndRR:rndRR+rndSiz, rndCC:rndCC+rndSiz] = newMsk
        # (2) get coords of masked bounding-box
        ptsRC = np.where(tmpMsk)
        minR = np.min(ptsRC[0]) - 2
        maxR = np.max(ptsRC[0]) + 2
        minC = np.min(ptsRC[1]) - 2
        maxC = np.max(ptsRC[1]) + 2
        # (2.1) BBox format: ((x1,y1), (x2, y2))
        newBBox = ((minC, minR),(maxC, maxR))
        # newBBoxW = ((minC, minR), (maxC-minC, maxR-minR))
        for chi in range(3):
            tmpImgBG = newImgBG[rndRR:rndRR+rndSiz, rndCC:rndCC+rndSiz, chi]
            tmpImgFG = newImg[:,:,chi]
            tmpImgBG[newMsk>0] = tmpImgFG[newMsk>0]
            newImgBG[rndRR:rndRR+rndSiz, rndCC:rndCC+rndSiz, chi] = tmpImgBG
        retMskBG_S[tmpMsk > 0] = newCls + 128
        retMskBG_I[tmpMsk > 0] = cntInstance + 128
        retBBoxCls.append((newBBox, newCls, cntInstance))
        cntInstance += 1
        #
    if isDebug:
        plt.subplot(1,3,1)
        plt.imshow(newImgBG)
        for ibbox in retBBoxCls:
            tbbox = ibbox[0]
            tbboxW = tbbox[1][0] - tbbox[0][0]
            tbboxH = tbbox[1][1] - tbbox[0][1]
            plt.gcf().gca().add_artist(plt.Rectangle((tbbox[0][0], tbbox[0][1]), tbboxW, tbboxH, edgecolor='r', fill=False))
        plt.title('Image')
        plt.subplot(1, 3, 2)
        plt.imshow(retMskBG_S)
        for ibbox in retBBoxCls:
            tbbox = ibbox[0]
            tbboxW = tbbox[1][0] - tbbox[0][0]
            tbboxH = tbbox[1][1] - tbbox[0][1]
            plt.gcf().gca().add_artist(plt.Rectangle((tbbox[0][0], tbbox[0][1]), tbboxW, tbboxH, edgecolor='r', fill=False))
        plt.title('Semantic segments')
        plt.subplot(1, 3, 3)
        plt.imshow(retMskBG_I)
        for ibbox in retBBoxCls:
            tbbox = ibbox[0]
            tbboxW = tbbox[1][0] - tbbox[0][0]
            tbboxH = tbbox[1][1] - tbbox[0][1]
            plt.gcf().gca().add_artist(plt.Rectangle((tbbox[0][0], tbbox[0][1]), tbboxW, tbboxH, edgecolor='r', fill=False))
        plt.title('Instance segments')
        plt.show()
    return {
        'img': newImgBG.astype(np.uint8),
        'msk_s': retMskBG_S.astype(np.uint8),
        'msk_i': retMskBG_I.astype(np.uint8),
        'bbox': retBBoxCls
    }

#################################################
if __name__ == '__main__':
    fidxFG = '/home/ar/data/test_dataset_0/02_data_resized_512x512_mask/idx.txt'
    fidxBG = '/home/ar/data/test_dataset_0/03_data_backgrounds/idx.txt'
    wdirOut = '/home/ar/data/test_dataset_0/data_generated'
    #
    isDebug = False
    def_newSize    = (512, 512)
    def_range_sizes=(128, 256 + 64)
    def_range_angle=(-32, +32)
    def_range_num_samples = (1, 3)
    def_cls_names = {
        1: 'bear',
        2: 'cat',
        3: 'dog',
        4: 'duc',
    }
    if def_newSize is not None:
        wdirOut = '{}-{}x{}'.format(wdirOut, def_newSize[0], def_newSize[1])
    if not os.path.isdir(wdirOut):
        os.makedirs(wdirOut)
    #
    def_num_gen_samples = 1000
    #
    # dataFG = pd.read_csv(fidxFG, sep=',')
    dataBG = pd.read_csv(fidxBG, sep=',')
    # numImgsFG = len(dataFG)
    numImgsBG = len(dataBG)
    # wdirFG = os.path.dirname(fidxFG)
    wdirBG = os.path.dirname(fidxBG)
    # arrLbl = dataFG['clsid'].as_matrix()
    # (1) FG
    # pathImgsFG = dataFG['path'].as_matrix()
    # pathImgsFG = [os.path.join(wdirFG, xx) for xx in pathImgsFG]
    # (2) BG
    pathImgsBG = dataBG['path'].as_matrix()
    pathImgsBG = [os.path.join(wdirBG, xx) for xx in pathImgsBG]
    #
    imgCnt = 0
    fotInfoAll = os.path.join(wdirOut, 'info-idx.txt')
    fall = open(fotInfoAll, 'w')
    for ii in range(def_num_gen_samples):
        ridx = np.random.randint(0, numImgsBG)
        pathBG = pathImgsBG[ridx]
    # for ipathBG, pathBG in enumerate(pathImgsBG):
        ret = generateRandomizedScene(pathBG=pathBG,
                                newSize=def_newSize,
                                pathIdxFG=fidxFG,
                                p_rangeSizes=def_range_sizes,
                                p_rangeAngle=def_range_angle,
                                p_rangeNumSamples=def_range_num_samples, isDebug=isDebug)
        imgOut  = ret['img']
        mskOutS = ret['msk_s']
        mskOutI = ret['msk_i']
        bboxOut = ret['bbox']
        numObj = len(bboxOut)
        foutPref = os.path.join(wdirOut, 'generated_{0:05d}_n{1:02d}'.format(imgCnt, numObj))
        foutImg = '{0}-img.jpg'.format(foutPref)
        # foutMskS = '{0}-msks.png'.format(foutPref)
        # foutMskI = '{0}-mski.png'.format(foutPref)
        # foutInfo = '{0}.csv'.format(foutPref)
        foutMskS = '{0}-msks.png'.format(foutImg)
        foutMskI = '{0}-mski.png'.format(foutImg)
        foutInfo = '{0}.csv'.format(foutImg)
        skio.imsave(foutImg,  imgOut)
        skio.imsave(foutMskS, mskOutS)
        skio.imsave(foutMskI, mskOutI)
        with open(foutInfo, 'w') as f:
            f.write('x1,y1,x2,y2,cls,instance\n')
            for ii,info in enumerate(bboxOut):
                tbbox = info[0]
                tcls = info[1]
                tidx = info[2]
                f.write('{0},{1},{2},{3}, {4},{5}\n'.format(tbbox[0][0], tbbox[0][1], tbbox[1][0], tbbox[1][1], tcls, tidx))
                fall.write('{5},{0},{1},{2},{3},{4}\n'.format(
                    tbbox[0][0], tbbox[0][1], tbbox[1][0], tbbox[1][1], def_cls_names[tcls], os.path.basename(foutImg)))
        imgCnt += 1
        print ('\t[{0}/{1}] : {2}'.format(ii, def_num_gen_samples, pathBG))