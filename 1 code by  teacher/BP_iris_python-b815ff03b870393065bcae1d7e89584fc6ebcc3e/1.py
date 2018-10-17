
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
from BP_version3 import BPNeuralNetwork

def dataReader():
   #读取数据
   data = pd.read_csv("trainFeature.csv",header=0)
   #均值归一化nomalization
   for i in ['feature1','feature2','feature3']:
       data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
   return data

def dataReader_test():
   #读取数据
   data_test = pd.read_csv("testFeature.csv",header=0)
   #均值归一化nomalization
   for i in ['feature1','feature2','feature3']:
       data_test[i]=(data_test[i]-min(data_test[i]))/(max(data_test[i])-min(data_test[i]))
   # print data_test
   return data_test

nn = BPNeuralNetwork([3,5,2],'sigmoid')
data=dataReader()

data_test=dataReader_test()

print data
print data_test
# '''
# 当输入参数X为一个样本时增量学习
# 当输入参数X为多个样本时批量学习
# '''
# 打标签
y=[]
for j in range(6318):
    if data.loc[j][3]==0:
        y.append(np.array([1,0]))#列表转数组
               
    elif data.loc[j][3]==1:
        y.append(np.array([0,1]))
               
               
y_test=[]
for j in range(7580):
    if data_test.loc[j][3]==0:
        y_test.append(np.array([1,0]))#列表转数组
               
    elif data_test.loc[j][3]==1:
        y_test.append(np.array([0,1]))
               

        
y=np.array(y)
y_test=np.array(y_test)
nn.BPalgorithm(data.iloc[0:6318,0:3],y,0.3,1000)

# #测试集
nn.generalize(data_test.iloc[0:7580,0:3],y_test)
