#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


# def createDataSet():
#     dataSet = [[1, 1, 'yes'],
#                [1, 1, 'yes'],
#                [1, 0, 'no'],
#                [0, 1, 'no'],
#                [0, 1, 'no']]
#     labels = ['no surfacing', 'flippers']
#     # change to discrete values
#     return dataSet, labels


# 属性集：色泽-浅白（white）、青绿（dark green）、乌黑（black）0-2
#         根蒂-硬挺（stiff）、稍缩（shrink）、蜷缩（curl up）
#         敲声-沉闷、浊响、清脆
#         纹理-模糊、稍糊、清晰
#         脐部：凹陷、稍凹、平坦
#         触感：软粘、硬滑
def createDataSet():
    dataSet = [[1, 2, 1, 2, 0, 1, 'yes'],
               [2, 2, 0, 2, 0, 1, 'yes'],
               [2, 2, 1, 2, 0, 1, 'yes'],
               [1, 2, 0, 2, 0, 1, 'yes'],
               [0, 2, 1, 2, 0, 1, 'yes'],
               [1, 2, 1, 2, 1, 0, 'yes'],
               [2, 1, 1, 1, 1, 0, 'yes'],
               [2, 1, 1, 2, 1, 1, 'yes'],
               [2, 1, 0, 1, 1, 1, 'no'],
               [1, 1, 2, 2, 2, 0, 'no'],
               [0, 0, 2, 0, 2, 1, 'no'],
               [0, 0, 1, 0, 2, 0, 'no'],
               [1, 2, 1, 1, 0, 1, 'no'],
               [0, 1, 0, 1, 0, 1, 'no'],
               [2, 1, 1, 2, 1, 0, 'no'],
               [0, 2, 1, 0, 2, 1, 'no'],
               [1, 2, 0, 1, 1, 1, 'no']
               ]
    labels = ['color', 'root', 'sound', 'texture', 'navel', 'touch']
    # change to discrete values
    return dataSet, labels


# 计算信息熵,公式为：Ent = Pk * LOG2 Pk求和
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


# 按照给定特征划分数据集，其中axis-特征下标，value-特征取值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式，返回特征对应下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # 合并所有特征，让特征不重复
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer

# 提取叶节点中类别最多的类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   # 所有类为同一类，停止分类
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet,数据中只剩类标记，分类完成
        return majorityCnt(classList)    #
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

#读写时都有可能产生IOError，一旦出错，后面的f.close()就不会调用。
# 所以，为了保证无论是否出错都能正确地关闭文件，我们可以使用
# try ... finally来实现，但是每次都这么写实在太繁琐，所以，
# Python引入了with语句来自动帮我们调用close()方法：
# with open('/path/to/file', 'r') as f:
#     print(f.read())
# with open('/Users/michael/test.txt', 'w') as f:
#     f.write('Hello, world!')
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
