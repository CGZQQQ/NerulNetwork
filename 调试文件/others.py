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

from sklearn.model_selection import train_test_split
df=pd.read_csv('train_data.csv')
X=df['data'][0:10]
y=df['label'][0:10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xx=[]
for i in X_train:
    xx.append([int(i[1]),int(i[3]),int(i[5])])

print(np.array([i for i in y_train]).T)
print(np.array(y_train[0:6]).T)
print(y_test)
print(len())
