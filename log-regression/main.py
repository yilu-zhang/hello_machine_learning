#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang


import logRegres
if __name__ == "__main__":
    # 测试批梯度上升
    data_arr, label_arr = logRegres.loadDataSet()
    weights = logRegres.gradAscent(data_arr, label_arr)
    logRegres.plotBestFit(weights)