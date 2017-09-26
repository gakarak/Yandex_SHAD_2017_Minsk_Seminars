#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

#######################
"""
Calculate Shift, Rotation and Scale for two images
:return (Shift[dx,dy], Angle, Scale, Correlation_Peak_Quality[quality_CorrShift, Quality_Corr_RotScale])
"""
def imregcorr(img1,img2, isUseGradMag=False, isDebug=False, isSelectBestShiftPQ=False, pqThreshBestShift=0.1):
    img1=img1.astype(np.float)
    img2=img2.astype(np.float)
    img1=getSquaredImg(img1)
    img2=getSquaredImg(img2)
    if isUseGradMag:
        img1 = calcGradMag(img1)
        img2 = calcGradMag(img2)
    siz1=img1.shape
    siz2=img2.shape
    assert (siz1==siz2)
    #
    isRSIgnored=False
    dxyS0=(0,0)
    pqS0=0.0
    if isSelectBestShiftPQ:
        CCS0,dxyS0,ccvalS0,pqS0=phaseCorr(img1, img2, RadPQ=5, isUseHann=True, isFloatAcc=True)
    #
    img1F=np.abs(np.fft.fftshift(np.fft.fft2(img1)))
    img2F=np.abs(np.fft.fftshift(np.fft.fft2(img2)))
    img1FLP,MM=getLogPolar(img1F.copy())
    img2FLP,MM=getLogPolar(img2F.copy())
    hwLP=getCylHanning(img1FLP.shape, axis=0) #TODO: Check this point !!!
    # hwLP=cv2.createHanningWindow(img1FLP.shape[::-1], cv2.CV_64F)
    img1FLP=img1FLP*hwLP
    img2FLP=img2FLP*hwLP
    CCRS,dxyRS,ccvalRS,pqRS=phaseCorr(img1FLP,img2FLP,RadPQ=5, isUseHann=True, isFloatAcc=True)
    # _,_,pqRS=calcPeakQ(CCRS,dxyRS,Rad=3)
    dAng=(360./CCRS.shape[0])*dxyRS[1]
    dScl=np.exp(float(dxyRS[0])/MM)
    rotm2=cv2.getRotationMatrix2D((img2.shape[1]/2.0, img2.shape[0]/2.0), -dAng, 1./dScl)
    img2RS=cv2.warpAffine(img2.copy(), rotm2, img2.shape)
    CCS,dxyS,ccvalS,pqS=phaseCorr(img1, img2RS, RadPQ=5, isUseHann=True, isFloatAcc=True)
    if isSelectBestShiftPQ:
        if (pqS<pqThreshBestShift) and (pqS0>pqS):
            dAng=0.0
            dScl=1.0
            dxyS=dxyS0
            isRSIgnored=True
            img2RS=img1
    # _,_,pqS=calcPeakQ(CCS,dxyS,Rad=5)
    if isDebug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,9))
        plt.subplot(3, 2, 1)
        plt.imshow(nrmMat(np.concatenate( (img1, img2 ), axis=1 )))
        plt.title("win0-original")
        plt.subplot(3, 2, 2)
        plt.imshow(nrmMat(np.concatenate( (np.log(img1F+1),np.log(img2F+1)), axis=1 )))
        plt.title("win1-FFT-ABS")
        plt.subplot(3, 2, 3)
        plt.imshow(nrmMat(np.concatenate( (img1FLP,img2FLP), axis=1 )))
        plt.title("win2-FFT-LPT")
        #
        plt.subplot(3, 2, 4)
        plt.imshow(nrmMat(CCRS))
        plt.title("win3-CC-LogPolar")
        if isSelectBestShiftPQ:
            plt.subplot(3, 2, 5)
            plt.imshow(nrmMat(np.concatenate( (CCS0,CCS), axis=1 )))
            plt.title("win4-CC-Shift-XY (CCS0,CCS)")
        else:
            plt.subplot(3, 2, 5)
            plt.imshow(nrmMat(CCS))
            plt.title("win4-CC-Shift-XY")
        img2RS_Shifted=np.roll(np.roll(img2RS.copy(),int(np.round(dxyS[0])),1), int(np.round(dxyS[1])), 0)
        plt.subplot(3, 2, 6)
        plt.imshow(np.dstack( (nrmMat(img1),nrmMat(img2RS_Shifted),nrmMat(img1)) ))
        plt.title("win5-Registered")
        plt.show()
        print("dXY=%s, dAngle=%s, dScale=%s, peakQuality(XY,RS,RS0)=[%0.3f, %0.3f; %0.3f], isRSIgnored=%s" % (dxyS, dAng, dScl, pqS, pqRS, pqS0, isRSIgnored))
    return (dxyS,dAng,dScl,(max(pqS,pqS0),pqRS))

"""
:return Min-Max normed image in uchar type
"""
def nrmMat(mat):
    return cv2.normalize(mat, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)

"""
Calculate weighted center of mass <r>=Sum(mat_ij*pos_ij)/Sum(mat_ij)
"""
def calcWeightedCenter(mat, pos, siz=3):
    w=mat.shape[1]
    h=mat.shape[0]
    isBorder=False
    if (pos[0]<siz) or ((pos[0]+siz)>=w):
        isBorder=True
    if (pos[1]<siz) or ((pos[1]+siz)>=h):
        isBorder=True
    if isBorder:
        return pos
    else:
        x1=pos[0]-siz
        x2=pos[0]+siz+1
        y1=pos[1]-siz
        y2=pos[1]+siz+1
        rx=range(x1,x2)
        ry=range(y1,y2)
        xx,yy=np.meshgrid(rx,ry)
        cmat=mat[y1:y2, x1:x2]
        tmpx=cmat*xx
        tmpy=cmat*yy
        mm=np.sum(cmat)
        if mm==0:
            mm=1.0
        return (np.sum(tmpx)/mm,np.sum(tmpy)/mm)

"""
Calculate Correlation peak quality
:return CorrMax in rect RadxRad around pos,
CorrMax exclude region around pos, and normed quality [0,1]
"""
def calcPeakQ(mat, pos, Rad=3):
    siz=mat.shape
    r1=pos[1]-Rad
    r2=pos[1]+Rad+1
    c1=pos[0]-Rad
    c2=pos[0]+Rad+1
    if r1<0:
        r1=0
    if r2>=siz[0]:
        r2=siz[0]
    if c1<0:
        c1=0
    if c2>=siz[1]:
        c2=siz[1]
    cmat=mat[r1:r2, c1:c2].copy()
    p1=np.max(cmat)
    mat[r1:r2, c1:c2]-=p1
    p2=np.max(mat)
    mat[r1:r2, c1:c2]=cmat
    return (p1,p2,(p1-p2)/p1)

"""
calc phase correlation,
:return Correlation map, estimated shift (dx,dy), peak-value, and peak-quality (in progress)
"""
def phaseCorr(img1, img2, isUseHann=True, isDebug=False, isFloatAcc=True, RadPQ=3):
    assert(img1.shape==img2.shape)
    if isUseHann:
        hw=cv2.createHanningWindow(img1.shape[::-1], cv2.CV_64F)
        img1=img1*hw
        img2=img2*hw
    if isDebug:
        cv2.imshow("debug-img12", np.concatenate((nrmMat(img1), nrmMat(img2))))
    fft1=np.fft.fft2(img1)
    fft2=np.fft.fft2(img2)
    CC=np.fft.fftshift(np.fft.ifft2( (fft1*fft2.conj())/(fft2*fft2.conj()) ).real)
    minVal, maxVal, minLoc, maxLoc=cv2.minMaxLoc(CC)
    _,_,pq=calcPeakQ(CC,maxLoc,RadPQ)
    if isFloatAcc:
       maxLoc=calcWeightedCenter(CC,maxLoc,RadPQ)
    dxy=(maxLoc - np.array(img1.shape[::-1])/2) #TODO: Check this point !!!
    return (CC, dxy, maxVal,pq)

"""
:return cylindrical Hanning window in X or Y direction
axis '0' -> y
axis '1' -> x
"""
def getCylHanning(size, axis=0, ptype=cv2.CV_64F):
    tmp=cv2.createHanningWindow(size[::-1], ptype)
    if axis==0:
        tmp=tmp[size[1]/2,:]
        ret=np.tile(tmp,(size[0],1))
    else:
        tmp=tmp[:, size[0]/2]
        ret=np.transpose(np.tile(tmp,(size[1],1)))
        # print ret.shape
    return ret

"""
:return gradient-magnitude image
"""
def calcGradMag(img):
    gx=cv2.Sobel(img, cv2.CV_64F, 1,0)
    gy=cv2.Sobel(img, cv2.CV_64F, 0,1)
    gm=np.sqrt(gx*gx+gy*gy)
    # cv2.imshow("img-gm", nrmMat(gm))
    return gm

"""
:return LogPolar image cropped by 1/coeffCrop and (coeffCrop-1)/coeffCrop in scale-direction
"""
def getLogPolar(img, coeffCrop=6.):
    imgLP=cv2.cv.fromarray(img.copy())
    siz=np.array(img.shape, dtype=np.int32)
    p0=siz/2
    sizMin=np.min(p0[0:2])
    MM=2.0*sizMin/np.log(sizMin)
    cv2.cv.LogPolar(cv2.cv.fromarray(img),imgLP,(img.shape[1]/2, img.shape[0]/2), MM, cv2.cv.CV_WARP_FILL_OUTLIERS)
    cv2.line(img, (0, p0[0]), (siz[1],p0[0]), (255,255,255), 2)
    cv2.line(img, (p0[1], 0), (p0[1],siz[0]), (255,255,255), 2)
    imgLPM=np.asarray(imgLP)
    # sizLPM=np.array(imgLPM.shape, dtype=np.int32)
    # p0LPM=siz/2
    cropPTS1=int(MM*np.log(1.*sizMin/coeffCrop)) #TODO: check this point: default value of coeffCrop is optimal?
    cropPTS2=int(MM*np.log((coeffCrop-1)*sizMin/coeffCrop))
    imgLPCrop=imgLPM[:,cropPTS1:cropPTS2].copy()
    return (imgLPCrop,MM)
"""
:return Squared image cropped around center by min(width,height)
"""
def getSquaredImg(img):
    if img.shape[0]==img.shape[1]:
        return img.copy()
    else:
        siz=img.shape[0:2]
        if siz[0]>siz[1]:
            dy=(siz[0]-siz[1])/2
            return img[dy:dy+siz[1],:].copy()
        else:
            dx=(siz[1]-siz[0])/2
            return img[:,dx:dx+siz[0]].copy()

"""
:return resized image with size [maxSize, maxSize*img.height/img.width] or [maxSize*img.width/img.height, maxSize]
"""
def resizeToMaxSize(img, maxSize):
    siz=(img.shape[1], img.shape[0])
    newSize=(maxSize, int(round(maxSize*float(siz[1])/float(siz[0]))))
    if siz[0]<siz[1]:
        newSize=(int(round(maxSize*float(siz[0])/float(siz[1]))), maxSize)
    return cv2.resize(img, newSize, None, 0,0, cv2.INTER_CUBIC)

"""
siz - size of rectangule: [height, width]
rad - 0..1
"""
def getCirlceMask(rad, siz):
    sizMin=min(siz)
    xx,yy=np.meshgrid( np.linspace(0,float(siz[1])/sizMin,siz[1]), np.linspace(0,float(siz[0])/sizMin,siz[0]))
    xy0=(0.5*siz[1]/sizMin, 0.5*siz[0]/sizMin)
    msk=(((xx-xy0[0])**2+(yy-xy0[1])**2)<((rad/2.0)**2))
    return msk.astype(np.float)

####################################
"""
Helper class for read images listed in CSV-file,
work similar to cv2.VideoCapture()
if in CSV file contains full path, then parameter 'parWDir'  is not need to specify.
"""
class VideoCSVReader:
    def __init__(self, parFidx, parWDir=None):
        self.isDebug=False
        self.pos=0
        self.wdir=None
        self.fcsvIdx=None
        self.listFImg=[]
        self.readFrameList(parFidx, parWDir)
    def readFrameList(self, parFidx, parWDir):
        self.pos=0
        self.fcsvIdx=None
        self.wdir=None
        self.listFImg=[]
        if os.path.isfile(parFidx):
            tmpList=np.recfromtxt(parFidx)
            isAllOk=True
            if len(tmpList)>0:
                for ii in tmpList:
                    tmpFimg=ii
                    if parWDir!=None:
                        tmpFimg='%s/%s' % (parWDir, ii)
                    if os.path.isfile(tmpFimg):
                        self.listFImg.append(tmpFimg)
                    else:
                        isAllOk=False
                        if self.isDebug:
                            print "Can't find file [%s]" % tmpFimg
                        break
            else:
                isAllOk=False
            if isAllOk:
                self.fcsvIdx=parFidx
                self.wdir=parWDir
            else:
                self.listFImg=[]
    def resetPos(self):
        self.pos=0
    def getNumFrames(self):
        return len(self.listFImg)
    def isOpened(self):
        return (self.getNumFrames()>0)
    def read(self, parPos=-1):
        numfrm=self.getNumFrames()
        if parPos<0:
            if self.pos<numfrm:
                ret=(True, cv2.imread(self.listFImg[self.pos]))
                self.pos+=1
                return ret
            else:
                return (False, None)
        else:
            if parPos<numfrm:
                ret=(True, cv2.imread(self.listFImg[parPos]))
                return ret
            else:
                return (False, None)
    def printInfo(self):
        print "fcsv=%s, wdir=[%s]" % (self.fcsvIdx, self.wdir)
        print "#frames=%d" % self.getNumFrames()
        print self.listFImg

####################################
"""
Helper very simple class for Video-Navigation, calculate RotSclShift and accumulate track-info
"""
class VideoNavigator:
    def __init__(self, maxFrameSize=-1):
        self.a2r=(np.pi/180.)
        self.listNavData=[]
        self.currFrame=None
        self.maxFrameSize=maxFrameSize
    def setStartState(self, frm, posXY=(0, 0), angle=0.0, scl=1.0):
        self.listNavData=[]
        self.currFrame=None
        if frm is not None:
            tmpState=(posXY[0], posXY[1], angle, scl, 1.0)
            if self.maxFrameSize>0:
                self.currFrame=resizeToMaxSize(frm, self.maxFrameSize)
            else:
                self.currFrame=frm.copy()
            self.listNavData.append(tmpState)
    def processNewFrame(self, frm, isDebug=False, isUseGradMag=False, isSelectBestShiftPQ=True):
        if self.isInitialised and (frm!=None):
            if self.maxFrameSize>0:
                frm=resizeToMaxSize(frm, self.maxFrameSize)
            dxy,dAng,dScl,pQ=imregcorr(self.currFrame, frm, isDebug=isDebug, isUseGradMag=isUseGradMag, isSelectBestShiftPQ=isSelectBestShiftPQ)
            resOld=self.listNavData[-1]
            dxAbs=(+dxy[0])*np.cos(+self.a2r*resOld[2])-(+dxy[1])*np.sin(+self.a2r*resOld[2])
            dyAbs=(+dxy[0])*np.sin(+self.a2r*resOld[2])+(+dxy[1])*np.cos(+self.a2r*resOld[2])
            dxAbs*=resOld[3]
            dyAbs*=resOld[3]
            newState=(resOld[0]+dxAbs,resOld[1]+dyAbs,resOld[2]+dAng,resOld[3]/dScl,pQ[0])
            self.listNavData.append(newState)
            self.currFrame=frm.copy()
            return pQ[0]
        return 0.0
    def getNavDataAsArray(self):
        if self.isInitialised():
            return np.array(self.listNavData, dtype=float)
        else:
            return None
    def isInitialised(self):
        return (self.currFrame!=None)
    def printInfo(self):
        print "isInitialised=%s" % self.isInitialised()
        print self.listNavData
    def getLastState(self):
        if self.isInitialised() and (len(self.listNavData)>0):
            return self.listNavData[-1]
        return None
    def getCurrentFrame(self):
        if self.currFrame!=None:
            return self.currFrame.copy()
        else:
            return None

####################################
"""
Test-function for VideoNavigator, fvideo - path to video-file or camera
"""
def test_Video_Corr_With_VideoNavigator(fvideo):
    isFirstRunCamera=True
    if not np.isreal(fvideo):
        if not os.path.isfile(fvideo):
            print "Can't find file [%s]" % fvideo
            return
        else:
            isFirstRunCamera=False
    videoNavigator=VideoNavigator(maxFrameSize=512)
    isFirstRun=True
    cap=cv2.VideoCapture(fvideo)
    if cap.isOpened():
        while True:
            ret,currFrame=cap.read() # skip some #Frames
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            if not ret:
                break
            currFrame=cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
            if isFirstRunCamera:
                print "Please setup camera and press 'SPACEBAR' to start correlation-processing"
                cv2.imshow("win-camera", currFrame)
                key=cv2.waitKey(10)
                if key==32:
                    cv2.destroyAllWindows()
                    isFirstRunCamera=False
                # print key
            else:
                if videoNavigator.isInitialised():
                    pQ=videoNavigator.processNewFrame(currFrame, isDebug=True)
                    if pQ<0.1:
                        cv2.waitKey(0)
                    else:
                        key=cv2.waitKey(5)
                        if key==27:
                            break
                    if isFirstRun:
                        cv2.waitKey(0)
                        isFirstRun=False
                else:
                    videoNavigator.setStartState(currFrame)
    print "-----[ Results ]-----"
    arrResults=videoNavigator.getNavDataAsArray()
    print arrResults
    import matplotlib.pyplot as plt
    plt.plot(arrResults[:,0],arrResults[:,1])
    plt.show()

"""
Test-function, fvideo - path to video-file
"""
def test_Video_Corr(fvideo):
    if not os.path.isfile(fvideo):
        print "Can't find file [%s]" % fvideo
        return
    isFirstRun=True
    frm1=None
    cap=cv2.VideoCapture(fvideo)
    listResults=[(0,0,0,1.0, 1.0)]
    if cap.isOpened():
        while True:
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            ret,currFrame=cap.read()
            if not ret:
                break
            currFrame=cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
            currFrame=resizeToMaxSize(currFrame,512)
            if frm1!=None:
                dxy,dAng,dScl,pQ=imregcorr(frm1, currFrame, isDebug=True, isUseGradMag=False, isSelectBestShiftPQ=True)
                resOld=listResults[-1]
                dxAbs=(-dxy[0])*np.cos(resOld[2])-(-dxy[1])*np.sin(resOld[2])
                dyAbs=(-dxy[0])*np.sin(resOld[2])+(-dxy[1])*np.cos(resOld[2])
                resNew=(resOld[0]+dxAbs,resOld[1]+dyAbs,resOld[2]-dAng,resOld[3]/dScl,pQ[0])
                listResults.append(resNew)
                if min(pQ)<0.1:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(5)
                if isFirstRun:
                    cv2.waitKey(0)
                    isFirstRun=False
            frm1=currFrame.copy()
    print "-----[ Results ]-----"
    arrResults=np.array(listResults, dtype=float)
    print arrResults
    import matplotlib.pyplot as plt
    plt.plot(arrResults[:,0],arrResults[:,1])
    plt.show()

"""
Test-function, fvideo - path to CSV-file with path of images
"""
def test_CSVFrames_Corr(fcsv):
    if not os.path.isfile(fcsv):
        print "Can't find file [%s]" % fcsv
        return
    isFirstRun=True
    frm1=None
    cap=VideoCSVReader(fcsv)
    cap.printInfo()
    listResults=[(0,0,0,1.0, 1.0)]
    if cap.isOpened():
        while True:
            ret,currFrame=cap.read()
            if not ret:
                break
            currFrame=cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
            currFrame=resizeToMaxSize(currFrame,512)
            if frm1!=None:
                dxy,dAng,dScl,pQ=imregcorr(frm1, currFrame, isDebug=True, isUseGradMag=False, isSelectBestShiftPQ=True)
                resOld=listResults[-1]
                resNew=(resOld[0]-dxy[0],resOld[1]-dxy[1],resOld[2]-dAng,resOld[3]/dScl,pQ[0])
                listResults.append(resNew)
                if min(pQ)<0.1:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(5)
                if isFirstRun:
                    cv2.waitKey(0)
                    isFirstRun=False
            frm1=currFrame.copy()
    print "-----[ Results ]-----"
    arrResults=np.array(listResults, dtype=float)
    print arrResults
    import matplotlib.pyplot as plt
    plt.plot(arrResults[:,0],arrResults[:,1])
    plt.show()

####################################
if __name__=='__main__':
    print "this is python module [%s]" % os.path.basename(sys.argv[0])

    ## Example usage Video-File Correlation:
    # fvideo='/home/ar/video/cam_video/VID_20150402_104432.mp4'
    # fvideo='/home/ar/video/drone_project/video_2015.3.3_10.31.5_s640x480_24.0FPS_color.avi'
    fvideo='/home/ar/big.data/data.UAV/data_AlexKravchonok/video.avi'
    test_Video_Corr_With_VideoNavigator(fvideo)

    ## Example usage Camera-Video Correlation:
    # test_Video_Corr_With_VideoNavigator(0)

    ## Example usage CSV-file (with images path) Correlation:
    # fcsv='/home/ar/video/drone_project/frames.txt'
    # test_CSVFrames_Corr(fcsv)
    # cap=VideoCSVReader(fcsv)
    # cap.printInfo()

