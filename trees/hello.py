#!/usr/bin/env python
# -*- coding: utf-8 -*-

#author : zhangyilu


import trees
import treePlotter

def test():
  print "hello world";
if __name__=='__main__':
    # train_data, labels = trees.createDataSet()
    # my_trees = trees.createTree(train_data, labels)
    # print(my_trees)
    #trees.storeTree(my_trees, 'classifiermelon.txt')

    melon_tree = trees.grabTree('classifiermelon.txt')
    print(melon_tree)
    melon_labels = ['color', 'root', 'sound', 'texture', 'navel', 'touch']
    melon_feature = [1, 1, 1, 1, 1, 1]
    print("the predicted result is:", trees.classify(melon_tree, melon_labels, melon_feature))

    treePlotter.createPlot(melon_tree)
    # print(treePlotter.getNumLeafs(my_trees), treePlotter.getTreeDepth(my_trees))

    # ent = trees.calcShannonEnt(train_data)
    # feature1 = trees.splitDataSet(train_data, 0, 0)
    # feature2 = trees.splitDataSet(train_data, 0, 1)
    # best_feature = trees.chooseBestFeatureToSplit(train_data)
    # print(ent)
    # print(feature1, feature2)
    # print(best_feature)



