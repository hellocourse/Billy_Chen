# coding: utf-8


#-*- encoding:utf-8 -*-
import sys   #reload()之前必须要引入模块
reload(sys)
sys.setdefaultencoding('utf-8')

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png')
plt.imshow(img)

plt.show()