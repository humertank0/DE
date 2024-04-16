# coding:UTF-8

import random as rd
import time
from random import sample

import numpy as np
import pandas as pd
import xgboost as xgb
# 导入要用的包
# from __future__ import division
from numpy import zeros, shape
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from filter import top_select, reliefFScore
from ReliefF import top_select, reliefFScore
from mutationOperator import *

'''
读入数据
'''
# rawData = np.loadtxt("..//datasets//arcene(200,10000).csv", delimiter=',',
#                      encoding='utf-8-sig')
rawData = np.loadtxt("D:\\OneDrive\\文档\\datasets\\de\\nci9.csv", delimiter=',',
                     encoding='utf-8-sig')
feat = rawData[:, :-1]
label = rawData[:, -1]
# rawData=pd.read_csv("D:\OS\Desktop\VTE\\temp1.csv",encoding='utf-8-sig',header=0)
# rawData=rawData.values
# rawData=np.delete(rawData,0,axis=1)
# feat=rawData[:,:-1]
# label=rawData[:,-1]

'''
标准化
'''
minmax = preprocessing.MinMaxScaler()
feat = minmax.fit_transform(feat)

'''
使用filter进行初步降维
'''
if (feat.shape[1] >= 7000):
    Score = reliefFScore(feat, label)
    top_index = top_select(Score)
    feat = feat[:, top_index]

'''
初始化参数
'''
NP = 40  # 种群大小
size = feat.shape[1]  # 特征数量
xMin = 0  # 下界
xMax = 1  # 上界
F = 0.2  # 缩放因子
CR = 0.8  # 交叉因子
fitnessVal = zeros((NP, 1))  # 初始化适应度值
XTemp = zeros((NP, size))  # 初始种群
gen = 0  # 代数
maxGen = 100  # 最大迭代次数


# 种群初始化
def popInitialization():
    # for i in range(NP):
    #     for j in range(size):
    #         XTemp[i, j] = xMin + rd.random() * (xMax - xMin)
    XTemp = np.random.rand(NP, size) * (xMax - xMin) + xMin
    return XTemp


# 计算适应值函数
def calFitness(XCorssOverTmp, X):
    xIndex = []
    for j in range(len(X)):
        if (X[j] >= 0.6):
            xIndex.append(j)
    Xtest = feat[:, xIndex]
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='manhattan', n_jobs=-1)
    # clf=KNeighborsClassifier(n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True)
    # fitness = cross_val_score(clf, Xtest, label, cv=10,n_jobs=-1).mean()
    fitness = cross_val_score(clf, Xtest, label, cv=kf, n_jobs=-1).mean()
    return -fitness
    # '''
    # 临时改动
    # '''
    # x_train,x_test,y_train,y_test= train_test_split(Xtest,label,test_size=0.3,random_state = 20,shuffle = True)
    #
    # dtrain = xgb.DMatrix(x_train, label=y_train)   # XGBoost的专属数据格式，但是也可以用dataframe或者ndarray
    # dtest = xgb.DMatrix(x_test, label=y_test)  # # XGBoost的专属数据格式，但是也可以用dataframe或者ndarray
    #
    #
    # params = {'max_depth': 6, 'eta': 0.3}    # 设置XGB的参数，使用字典形式传入
    # bst = xgb.train(params, dtrain)   # 训练
    #
    # # make prediction
    # preds = bst.predict(dtest)   # 预测
    # preds = np.round(preds)
    #
    # acc = np.sum(y_test == preds) / len(y_test)
    # # print(acc)
    # return -acc


def crossover(XTemp, XMutationTmp, CR):
    m, n = shape(XTemp)
    XCorssOverTmp = zeros((m, n))
    for i in range(m):
        for j in range(n):
            r = rd.random()
            if r <= CR:
                XCorssOverTmp[i, j] = XMutationTmp[i, j]
            else:
                XCorssOverTmp[i, j] = XTemp[i, j]
    Xtest = XCorssOverTmp
    return XCorssOverTmp


def selection(XTemp, XCorssOverTmp, fitnessVal):
    fitnessCrossOverVal = zeros((NP, 1))
    for i in range(NP):
        fitnessCrossOverVal[i, 0] = calFitness(XCorssOverTmp, XCorssOverTmp[i])
        if (fitnessCrossOverVal[i, 0] < fitnessVal[i, 0]):
            # for j in range(n):
            #     XTemp[i, j] = XCorssOverTmp[i, j]
            XTemp[i] = XCorssOverTmp[i]
            fitnessVal[i, 0] = fitnessCrossOverVal[i, 0]
    return XTemp, fitnessVal


def saveBest(XTemp, fitnessVal):
    m = shape(fitnessVal)[0]
    tmp = 0
    features = 0
    for i in range(1, m):
        if (fitnessVal[tmp] > fitnessVal[i]):
            tmp = i

    print(-fitnessVal[tmp][0] * 100, '%')
    for i in range(len(XTemp[tmp])):
        if (XTemp[tmp][i] >= 0.5):
            features = features + 1
    print('selected features:', features)


if __name__ == '__main__':
    print('样本数量为：', feat.shape[0], '样本特征数量为：', feat.shape[1])
    start = time.perf_counter()
    XTemp = popInitialization()
    # for i in range(NP):
    #     fitnessVal[i] = calFitness(XTemp, XTemp[i])
    # saveBest(XTemp,fitnessVal)
    while gen < maxGen:
        XMutationTmp = mutation_rr1(XTemp, F, NP, size)
        # XMutationTmp = mutation_temp(XTemp, F, NP, size)
        # XMutationTmp = mutation_rr2(XTemp, F, NP, size)
        # XMutationTmp = mutation_NSDE(XTemp, F, NP, size)
        # XMutationTmp = mutation_ctb1(XTemp, F, NP, size)
        # XMutationTmp = mutation_ctr1(XTemp, F, NP, size)
        XCorssOverTmp = crossover(XTemp, XMutationTmp, CR)
        XTemp, fitnessVal = selection(XTemp, XCorssOverTmp, fitnessVal)
        print('第', gen + 1, '代的准确率为：')
        saveBest(XTemp, fitnessVal)
        gen += 1
    end = time.perf_counter()
    print('time:', end - start)
    print("Done.")
