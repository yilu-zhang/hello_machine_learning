#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang

import svmMLiA

if __name__ == '__main__':
    # 简化版smo测试
    data_arr, label_arr = svmMLiA.loadDataSet('testSet.txt')
    # print(data_arr[:5])
    # print(label_arr[:5])
    w_t, b, alpha = svmMLiA.smoSimple(data_arr, label_arr, 0.6, 0.001, 40)
    svmMLiA.smo_simple_plt(data_arr, w_t, b)
    print(b)