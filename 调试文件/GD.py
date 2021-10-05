# import numpy as np
# import random
# import pprint
# from matplotlib import pyplot as plt
#
# x = np.arange(1,10,1)
# X1=x*x*x
# X2=x*x
# X3=x
# b=2
# # y=x^3-x^2-2x+2
# Y=X1-X2-2*X3+b
#
#
# # x=np.array([[1,2,3],[1,1,1],[2,2,2]])
# # x=np.matrix(x)
# # pprint.pprint(x)
# # pprint.pprint(np.sum(x,axis=1))
# # '''
# # matrix([[1, 2, 3],
# #         [1, 1, 1],
# #         [2, 2, 2]])
# # matrix([[6],
# #         [3],
# #         [6]])'''
# # pprint.pprint(np.sum(x,axis=0))
# # '''
# # matrix([[1, 2, 3],
# #         [1, 1, 1],
# #         [2, 2, 2]])
# # matrix([[4, 5, 6]])'''
#
# # 原始输入数据
# X = np.array([X1,X2,X3])
# X = np.transpose(X)
#
#
# # 初始化参数矩阵
# print("参数矩阵")
# W1=np.array([[np.random.uniform(0,10) for i in range(3)] for j in range(3)])
# W2=np.array([[np.random.uniform(0,10) for l in range(3)] for m in range(3)])
#
#
# # 前向传播
# print('第一层的结果')
# layer1=np.dot(X,W1)
# layer1_value=np.sum(layer1,axis=1)
# print(layer1)
# print(layer1_value)
# print('输出层结果')
# outputlayer=np.dot(layer1,W2)
# outputlayer_value=np.sum(outputlayer,axis=1)
# print(outputlayer)
# print(outputlayer_value)
# y=[i[0]-i[1]-2*i[2]+2 for i in outputlayer]
# loss=sum(np.array(y-Y)*np.array(y-Y))
#
# # 计算偏导数


import numpy
import matplotlib.pyplot as plt

X1 = [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593,
     0.719]
X2 = [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042,
     0.103]
Y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 网络结构 L1:3  L2:3   L3:1

# 数据初始化、预处理
m = len(X1)
#按列连接
X, Y = numpy.c_[numpy.ones(m), X1, X2], numpy.c_[Y]

# 数据洗牌, 即打乱顺序
orders = numpy.random.permutation(m)  # 打乱顺序，返回[0 - m-1]
X = X[orders]
Y = Y[orders]

# 数据集的切分，把70%作为训练集，30%作为测试集
nums = int(m * 0.7)
tranX, testX = numpy.split(X, (nums,))  # X[0:d]和X[d:-1]
tranY, testY = numpy.split(Y, (nums,))


# sigmoid函数
# 反向传播梯度下降要用到导数, 前向传播计算h
def g(z, deriv=False):  # deriv是判断是否对函数求导
   if deriv == True:  # 如果是求导的话，返回g(z)*(1-g(z)),a=g(z),传入参数为a值
       return z * (1 - z)
   return 1.0 / (1 + numpy.exp(-z))


# 前向传播
def FB(a1, theta1, theta2):
   z2 = numpy.dot(a1, theta1)  # 第二层的输入
   a2 = g(z2)  # 第二层的激活单元
   z3 = numpy.dot(a2, theta2)  # 第三层的输入
   a3 = g(z3)  # 第三层的激活单元，即预测
   return a2, a3


# 代价函数
def get_cost(a3, Y):
    m = len(a3)
    return (-1 / m) * (numpy.dot(Y.T, numpy.log(a3)) + numpy.dot((1 - Y).T, numpy.log(1 - a3)))
   # return -1.0/m*(Y.T.dot(numpy.log(a3))+(1-Y).T.dot(numpy.log(1-a3)))


# 反向传播
def BP(a1, a2, a3, Y, theta1, theta2, alpha):
   m = len(a1)
   delta3 = a3 - Y  # 小delta  最后一层的误差
   delta2 = delta3.dot(theta2.T) * g(a2, True)  # 第二层的误差

   deltaTheta2 = a2.T.dot(delta3)  # 第二层的deltatheta2
   deltaTheta1 = a1.T.dot(delta2)  # 第一层的deltatheta1
   # 更新theta
   theta1 -= alpha * (1 / m) * deltaTheta1
   theta2 -= alpha * (1 / m) * deltaTheta2

   # deltatheta2 = 1.0 / m * a2.T.dot(delta3)
   # deltatheta1 = 1.0 / m * a1.T.dot(delta2)
   # theta2 -= alpha * deltatheta2
   # theta1 -= alpha * deltatheta1

   return theta1, theta2


# 梯度下降
def gradDesc(a1, Y, alpha=0.1, iters=10):
   m, n = a1.shape
   # 初始化theta   随机种子(每次运行生成的随机数一样)
   numpy.random.seed(0)
   theta1 = 2 * numpy.random.rand(n, 3) - 1  # 第一层的theta定义shape为(n,3)  乘2减1是为了让theta值有正有负
   theta2 = 2 * numpy.random.rand(3, 1) - 1  # 第二层的theta定义shape为(3,1)
   J = numpy.zeros(iters)
   for i in range(iters):
       a2, a3 = FB(a1, theta1, theta2)
       J[i] = get_cost(a3, Y)
       theta1, theta2 = BP(a1, a2, a3, Y, theta1, theta2, alpha)  # BP更新theta
   return J, theta1, theta2



# 计算精度
def get_accuracy(a3, Y):
   m = len(a3)
   temp = 0
   for i in range(m):
       if numpy.where(a3[i] >= 0.5, 1, 0) == Y[i]:
           temp += 1
   return temp / m





# J, theta1, theta2 = gradDesc(tranX, tranY)
# a2, a3 = FB(testX, theta1, theta2)
# print('精度：', get_accuracy(a3, testY))
# print(J)
# print(theta1)
# print(theta2)

