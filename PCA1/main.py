#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang


import pca

if __name__ == "__main__":
    data_mat = pca.loadDataSet("testSet.txt")
    low_mat, recon_mat = pca.pca(data_mat, 1)

