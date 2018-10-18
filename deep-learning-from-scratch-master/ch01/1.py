#coding:utf-8 
# import matplotlib.pyplot as plt
# from  matplotlib.image  import  imread

# img =  imread('../dataset/lena.png')
# plt.imshow(img)

# plt.show()

#练习一下 类的格式  没什么了不起的 垃圾
# class man:
# 	def   __init__(self,name):
# 		self.name = name 
# 		print 'initlized'

# 	def  hello(self):
# 		print 'hello '+ self.name + '!'

# 	def  goodbye(self):
# 		print 'GOOdbye ' +  self.name + "!"


# m  =  man('chenyong')
# m.hello()
# m.goodbye()

#练习一下  矩阵函数，  结合 画图函数

# import  numpy  as np 
# import  matplotlib.pyplot  as  plt


# x=  np.arange(-16,16,0.1)

# y1 =  np.sin(x)
# y2 =  np.cos(x)



# plt.plot(x,y1,label = 'sin')
# plt.plot(x,y2,linestyle = '--',label = 'cos')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sin & cos ')
# plt.legend(loc = 'best way')



# plt.show()




#与门构建
import numpy as  np  

def   AND(x1,x2):
	x   =  np.array([x1,x2])
	w   =  np.array([0.5,0.5])
	b=  -0.7
	tmp = np.sum(w*x) +b
	if tmp<=0:
		return 0
	elif tmp>0 :
		return 1
# if __name__=='__main__':
# 	y=[]
# 	for xs in [(0,0),(1,0),(0,1),(1,1)]:
		
# 		y.append(AND(xs[0],xs[1]))
# 		print y

#非门 构建  
import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# if __name__ == '__main__':
#     y = []
#     for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
       
#         y.append(NAND(xs[0], xs[1]))
#         print(str(xs) + " -> " + str(y))


# import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# if __name__ == '__main__':
#     for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
#         y = OR(xs[0], xs[1])
#         print(str(xs) + " -> " + str(y))






# from and_gate import AND
# from or_gate import OR
# from nand_gate import NAND

# #异或门实现
# def XOR(x1, x2):
#     s1 = NAND(x1, x2)
#     s2 = OR(x1, x2)
#     y = AND(s1, s2)
#     return y

# if __name__ == '__main__':
#     for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
#         y = XOR(xs[0], xs[1])
#         print(str(xs) + " -> " + str(y))



 





