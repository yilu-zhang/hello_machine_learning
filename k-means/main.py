#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang

import kMeans
import numpy as np


if __name__ == '__main__':
    data_set = np.mat(kMeans.loadDataSet('testSet.txt'))
    cent, clus = kMeans.kMeans(data_set, 4)
    print(cent)
    #print(clus)