#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang

import regression


if __name__ == '__main__':
    # 增量式梯度下降线性回归算法
    regression.inc_gradient_descent('ex0.txt', step_len=0.1, iter_num=200, algorithm_type='incremental_gradient_descent', is_convergence_plt= True)
    # 画出局部加权二乘法的图形，使用钟型核
    # x_arr, y_arr = regression.loadDataSet('ex0.txt')
    # regression.lwlr_test_plot(x_arr, 'ex0.txt', 0.1)

    # 画出最小拟二乘法图形
   # regression.min_liner_revisit_plot('ex0.txt')
    # 最小拟二乘法
    # x_arr, y_arr = regression.loadDataSet('ex0.txt')
    # linear_para = regression.standRegres(x_arr, y_arr)
    # print(linear_para)


