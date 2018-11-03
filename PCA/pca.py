#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang


'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)   # 竖着求平均值
    meanRemoved = dataMat - meanVals  # 中心化
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 求取特征值和特征向量
    print(eigVals)
    print(eigVects)
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    print(eigValInd)
    print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 变换回原坐标系
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat
