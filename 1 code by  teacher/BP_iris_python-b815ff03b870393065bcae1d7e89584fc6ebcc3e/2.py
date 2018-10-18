# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import e
#数据读取和处理
data = pd.read_csv("iris.csv",header=0)
#数据集的基本操作
print(data.head())
print(data.columns)
print(data['Species'])#输出指定索引的一列
print(data.loc[0][0:4])#数组第0行前4个数