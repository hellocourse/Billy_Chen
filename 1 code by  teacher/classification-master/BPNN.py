# -*- coding: utf-8 -*-
from __future__ import division
import math
import random
import string
import pickle
import numpy as np
import pandas as pd
import time

Lableslist = {0:'0',
                 1:'1'}
random.seed(0)
# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b-a)*random.random() + a

# 生成大小 I*J 的矩阵，默认零矩阵 (当然，亦可用 NumPy 提速)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# 函数 sigmoid，这里采用 tanh，因为看起来要比标准的 1/(1+e^-x) 漂亮些
def sigmoid(x):
    return math.tanh(x)

# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    ''' 三层反向传播神经网络 '''
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1 # 增加一个偏差节点
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # 最后建立动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        # 激活输入层
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        # print self.ao
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        ''' 反向传播 '''

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        # for k in range(len(targets)):
        #     error = error + 0.5*(targets[k]-self.ao[k])**2
        error += 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        count = 0
        for p in patterns:
            target = Lableslist[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            print(p[0], ': 标签是 ', target, '-> 预测结果是 ', Lableslist[index])
            count += (target == Lableslist[index])
        accuracy = float(count/len(patterns))
        print("################################################")
        print("计算正确率中....")
        time.sleep(2)
        print('正确率accuracy: %-.9f' % accuracy)


    def train(self, patterns, iterations=10, N=0.1, M=0.01):
        # N: 学习速率(learning rate)
        # M: 动量因子(momentum factor)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                #print(p)
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 1 == 0:
                print('误差 %-.9f' % error)
        print("################ 训练完成  ######################")
        print("测试即将开始。。。。")
        time.sleep(3)

def readData(path):
    data = []
    # read dataset
    raw = pd.read_csv(path)
    raw_data = raw.values
    raw_feature = raw_data[0:,0:3]
    print(raw_feature)
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][3] == 0.0:
           ele.append([1,0])
        else:
            ele.append([0,1])
        data.append(ele)
    random.shuffle(data)
    return data


# features 0-2
# labels 3
def iris():

    training = readData('trainFeature.csv')
    test = readData('testFeature.csv')

    nn = NN(3,7,2)
    nn.train(training,iterations=100)
    nn.test(test)

if __name__ == '__main__':
    iris()

