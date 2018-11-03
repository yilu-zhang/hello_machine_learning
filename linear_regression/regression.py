#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang


"""
Created on Jan 8, 2011
@author: Peter
"""
from numpy import *
import matplotlib.pyplot as plt

# 打开一个用tab键分隔的文本文件
# dataMat - 属性/输入x，labelMat - 标记/输出y
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 根据最小拟二乘法公式求出参数ws
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat

    # 计算矩阵行列式
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse" # 行列式为0时，矩阵逆矩阵不存在
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def min_liner_revisit_plot(file_name):
    x_arr, y_arr = loadDataSet(file_name)
    coe = standRegres(x_arr, y_arr)   # coe-coefficient
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    # y_hat = x_mat * coe
    fig = plt.figure()   # 画图

    # 参数第一位，第二位分布表示y轴、x轴子图个数，第三位表示子图在第几象限
    ax = fig.add_subplot(111)

    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.A)   # 根据样本画出散点图
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * coe
    ax.plot(x_copy[:, 1], y_hat)  # 画根据最小拟二乘法求出直线
    plt.show()


# 增量式梯度下降线性回归算法
def inc_gradient_descent(file_name, iter_num=0, step_len=0.5, is_convergence_plt=False, algorithm_type='newton'):
    import time
    x_arr, y_arr = loadDataSet(file_name)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    feature_num = shape(x_mat)[1]   # 这里特征1没用
    coe = mat([1.0]*feature_num)  # 可用coe = ones((feature_num, 1))
    m = shape(x_mat)[0]
    if (iter_num > m) or (iter_num == 0):
        iter_num = m
    all_coe = mat([1.0] * feature_num)   # 浮点数需要写成1.0，否则插入浮点数会被自动转换成整型
    start_time = time.time()
    if algorithm_type == 'newton':
        for i in range(iter_num):
            #if fabs((((y_mat.T - x_mat * coe.T)).T * x_mat)[0, 0]) < 0.1:
            if fabs((((y_mat.T - x_mat * coe.T)).T * x_mat)[0, 0]) < 0.5:
                print((((y_mat.T - x_mat * coe.T)).T * x_mat)[0, 1])
                iter_num = i
                break
            # hessian = []
            #             # for n in range(feature_num):
            #             #     hess = [x_mat[i, n]*x_mat[i, m] for m in range(feature_num)]
            #             #     hessian.append(hess)
            #             # hess_mat = mat(hessian)
            #             # #hh_t = hess_mat.T * hess_mat
            #             # if linalg.det(hess_mat) == 0.0:
            #             #     print "This matrix is singular, cannot do inverse"
            #             #     if i == 150:
            #             #         return
            #             #     else:
            #             #         continue
            #coe = coe - (hess_mat.I * ((x_mat[i, :] * coe.T - y_mat[0, i])[0, 0]*x_mat[i, :].T)).T
            error = (x_mat[i, :] * coe.T - y_mat[0, i])[0, 0]
            coe = coe - mat([error * pow(x_mat[i, 0], 2), error * pow(x_mat[i, 1], 2)])
            all_coe = insert(all_coe, i, coe, axis=0)  # 往矩阵插入矩阵

    else:
        for j in range(iter_num):
            # for i in range(feature_num):  # 未善用矩阵
            if algorithm_type == 'incremental_gradient_descent':
                coe = coe + float(step_len * (y_mat[0, j] - x_mat[j, :] * coe.T)) * x_mat[j, :]
            elif algorithm_type == 'batch_gradient_descent':
                coe = coe + (step_len * (y_mat.T - x_mat * coe.T)).T * x_mat
            all_coe = insert(all_coe, j, coe, axis=0)   # 往矩阵插入矩阵
    used_time = time.time() - start_time
    print('The ' + algorithm_type + 'algorithm use is:')
    print(used_time)
    print('The iterable number is:' )
    print(iter_num)
    print('The parameter is:')
    print(coe)
    fig = plt.figure(figsize=(10, 10))  # 画图

    # 参数第一位，第二位分布表示y轴、x轴子图个数，第三位表示子图在第几象限
    ax1 = fig.add_subplot(311)

    ax1.scatter(x_mat[:, 1].flatten().A[0], y_mat.A)  # 根据样本画出散点图
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy*coe.T
    ax1.plot(x_copy[:, 1], y_hat)  # 画根据最小拟二乘法求出直线
    if is_convergence_plt:
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        x = arange(0, iter_num, 1)  # 创建0-iter_num，间隔为1的数组
        all_coe_mat = mat(all_coe)
        y1 = all_coe_mat[:iter_num, 0]
        y2 = all_coe_mat[:iter_num, 1]
        ax2.plot(x, y1)
        ax3.plot(x, y2)
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]  # 查看xMat行数
    weights = mat(eye((m)))  # eye - 创建对角数组
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 画出局部加权二乘法图形
def lwlr_test_plot(testArr, filename, k=1.0):
    x_arr, y_arr = loadDataSet(filename)
    y_hat = lwlrTest(testArr, x_arr, y_arr, k)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    srt_ind = x_mat[:, 1].argsort(0)  # 按第二列排序，排序结果返回对应行索引
    x_sort = x_mat[srt_ind][:, 0, :]  # 注意这里x_mat[srt_ind]是一个三维矩阵，需将其变成二维
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srt_ind])
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.A, s=2, c='red')
    plt.show()


def lwlrTestPlot(xArr, yArr, k=1.0):  # same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))  # easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    # returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n, 1));
    wsTest = ws.copy();
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # returnMat[i,:]=ws.T
    # return returnMat


# def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()

from time import sleep
import json
import urllib2


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'problem with item %d' % i


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = [];
        trainY = []
        testX = [];
        testY = []
        random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX);
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr);
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print "the best model from Ridge Regression is:\n", unReg
    print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)
