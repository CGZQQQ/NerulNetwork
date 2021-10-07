"""制作数据集"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection

# n=10000
# train_x = np.random.randint(0, 10, (n,3))
# train_y = np.where(train_x.sum(axis=1) > 15, 1, 0)
# X=[]
# Y=[]
# print(train_y)
# for i in range(len(train_y)):
#     print(train_x[i],train_y[i])
#     X.append(train_x[i])
#     Y.append(train_y[i])
#
#
# dic={'data':X,'label':train_y}
# df=pd.DataFrame(data=dic)
# df.to_csv('train_data.csv',index=None,encoding='utf_8_sig')

# from sklearn.model_selection import train_test_split
# df=pd.read_csv('train_data.csv')
# X=df['data'][0:10]
# y=df['label'][0:10]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# xx=[]
# for i in X_train:
#     xx.append([int(i[1]),int(i[3]),int(i[5])])
#
# print(np.array([i for i in y_train]).T)
# print(np.array(y_train[0:6]).T)
# print(y_test)
# print(len())


# """指数加权实验"""
# import numpy as np
# import matplotlib.pyplot as plt
# n=1000
# m=50 # 只取每个指数函数的前50个
# b=0.90
# b2=0.50
# theta_t=np.random.randint(0,100,n)
# beta=[(1-b)*b**(100-t) for t in range(100)]
# beta2=[(1-b2)*b2**(100-t) for t in range(100)]
# a=0
# average=[]
# for i in range(n):
#     average.append((a*i+theta_t[i])/(i+1))
#     a=(a*i+theta_t[i])/(i+1)
# result2=[np.sum(theta_t[i-m:i]*beta2[100-m:100],axis=0) for i in range(m,n)]
# result1=[np.sum(theta_t[i-m:i]*beta[100-m:100],axis=0) for i in range(m,n)]
# plt.plot(theta_t[20:n])
# plt.plot(result1,markersize=100)
# plt.plot(result2,markersize=100)
# # plt.scatter([x for x in range(20,n)],theta_t[20:n])
# # plt.scatter([x for x in range(len(result1))],result1)
# # plt.scatter([x for x in range(len(result2))],result2)
# plt.legend(['origin','beta2=0.9','beta=0.5'])
# plt.show()
import numpy as np

print(np.zeros((1,4))*5)