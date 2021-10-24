import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class NN():
    def __init__(self, alpha, struct, BatchSize):
        # Super parameters
        self.struct = struct
        self.L = len(struct)
        self.W = [np.random.randn(struct[i], struct[i - 1]) / np.sqrt(struct[i] + struct[i - 1]) for i in range(1, self.L)]
        self.B = [np.zeros((struct[i], 1)) for i in range(1,self.L)]
        self.BatchSize = BatchSize
        self.alpha = alpha
        self.loss = []
        self.lmda = 10
        # RMSProp Gradient decent parameters
        self.Sdw_beta = 0.99
        self.Sdw = np.array([np.zeros((struct[self.L - i], struct[self.L - i - 1])) for i in range(1, self.L)],   dtype=object)
        self.Sd_BN_beta = np.array([np.zeros((self.struct[self.L - 2 - i], 1)) for i in range(self.L - 2)], dtype=object)
        self.Sd_BN_gama = np.array([np.zeros((self.struct[self.L - 2 - i], 1)) for i in range(self.L - 2)], dtype=object)
        # Momentum Gradient decent parameters
        self.Vdw_beta = 0.9
        self.Vdw = np.array([np.zeros((struct[self.L - i], struct[self.L - i - 1])) for i in range(1, self.L)], dtype=object)
        self.Vd_BN_beta = np.array([np.zeros((self.struct[self.L - 2 - i], 1)) for i in range(self.L - 2)],dtype=object)
        self.Vd_BN_gama = np.array([np.zeros((self.struct[self.L - 2 - i], 1)) for i in range(self.L - 2)], dtype=object)
        # Batch Normalization parameters
        self.BN_gama = np.array([np.random.randn(self.struct[i + 1], 1) for i in range(self.L - 2)], dtype=object)
        self.BN_beta = np.array([np.random.randn(self.struct[i + 1], 1) for i in range(self.L - 2)], dtype=object)
        self.BN_sigma = np.array([np.zeros((self.struct[i + 1], 1)) for i in range(self.L - 2)], dtype=object)
        self.BN_mu = np.array([np.zeros((self.struct[i + 1], 1)) for i in range(self.L - 2)], dtype=object)
        self.I = np.identity(self.BatchSize)

    # 激活函数
    def g(self, z, diff=False, relu=False):
        if relu:
            if diff:
                return np.where(z>0,1,0)
            else:
                return np.maximum(0, z)
        else:
            if diff:
                return self.g(z) * (1 - self.g(z))
            else:
                return 1. / (1 + np.exp(-z))

    def Normalize_Data(self,data):
        average=np.mean(data,axis=1).reshape(3,1)
        variance=np.var(data,axis=1).reshape(3,1)
        normal_data=(data-average)/(variance)
        return normal_data

    def Genernate_Train_Data_batch(self,data_num):
        df = pd.read_csv('train_data.csv')
        X = df['data'][0:data_num]
        Y = df['label'][0:data_num]
        t_s = 100 / data_num
        # 归一化输入
        X = [[int(i[1]), int(i[3]), int(i[5])] for i in X]
        X = self.Normalize_Data(np.array(X).T)
        X = [[X[0][x], X[1][x], X[2][x]] for x in range(data_num)]
        # 前期准备
        train_x, test_x, train_y, test_y = \
            train_test_split(X, Y, test_size=t_s, random_state=42, shuffle=True)
        train_y_list = []
        train_x_list = []
        train_x = np.array(train_x).T
        self.tra_num = self.BatchSize
        for i in range(int((1 - t_s) * data_num / self.BatchSize)):
            train_y_list.append(np.array(train_y[i * self.BatchSize:i * self.BatchSize + self.BatchSize]))
            train_x_list.append(np.array(train_x[:, i * self.BatchSize:i * self.BatchSize + self.BatchSize]))
        # 训练集
        self.train_y = train_y_list
        self.test_y = np.array([n for n in test_y]).T
        self.train_x = train_x_list
        self.test_x = np.array(test_x).T

    def F_P(self,index):
        Z = []
        A = []
        Z.append(self.train_x[index])
        A.append(self.train_x[index])
        for i in range(L - 1):
            Z.append(self.W[i].dot(A[-1])+self.B[i])
            if i < L-2:
                A.append(self.g(Z[-1],relu=True))
            else:
                A.append(self.g(Z[-1]))
        # Loss Function
        y = A[-1]
        J = -(self.train_y[index] * (np.log(y)) + (1 - self.train_y[index]) * np.log(1 - y)).sum() / (self.tra_num )
        print('train_loss is : ', J)
        self.loss.append(J)
        return self.B_P(Z, y, A, index)

    def B_P(self, Z, y, A, index):
        # Back progatation
        dZ = []
        dA = []
        dB = []
        dW = []
        # 初始化一下
        dA.append((y - self.train_y[index]) / (y * (1 - y)))
        dZ.append(y - self.train_y[index])
        # 开始反向传播计算每一层权重的梯度，这里我没有加偏置
        for i in range(L - 1):
            dA.append(self.W[-i - 1].T.dot(dZ[-1]))
            dB.append(dZ[-1].sum(axis=1,keepdims=True))
            dW.append(dZ[-1].dot(A[-2 - i].T))
            dZ.append(dA[-1] * self.g(Z[-i - 2], diff=True, relu=True))
        dZ[-1] = dA[-1]
        # 梯度下降，每个batch结束都会梯度下降一次
        dW = np.array(dW, dtype=object)
        dB = np.array(dB, dtype=object)
        self.gc_dw =dW
        self.gc_db = dB
        # print(dW)
        # self.Adam(dW,index+1)
        self.Gradient_decent(dW,dB)

    # 普通梯度下降
    def Gradient_decent(self,dW,dB):
        for i in range(L-1):
            self.W[i]-=self.alpha*dW[-i-1]/self.BatchSize
            self.B[i]-=self.alpha*dB[-i-1]

    def gc_FP(self, index):
        Z = []
        A = []
        Z.append(self.train_x[index])
        #A.append(self.train_x[index])
        A.append(self.train_x[index])
        for i in range(L - 1):
            Z.append(self.W[i].dot(A[-1]) + self.B[i])
            if i < L - 2:
                A.append(self.g(Z[-1], relu=True))
            else:
                A.append(self.g(Z[-1]))
        # Loss Function
        y = A[-1]
        J = -(self.train_y[index] * (np.log(y)) + (1 - self.train_y[index]) * np.log(1 - y)).sum() / (
                self.tra_num )
        return J

    def gc_W(self,index):
        epslion=1e-7
        for i in range(0,len(self.W)):
            dw=np.zeros(self.W[i].shape)
            for j in range(len(self.W[i])):
                for k in range(len(self.W[i][j])):
                    self.W[i][j][k]-=epslion
                    J_minus=self.gc_FP(index)
                    self.W[i][j][k]+=2*epslion
                    J_plus=self.gc_FP(index)
                    dw[j][k]=(J_plus-J_minus)/(2*epslion)
                    self.W[i][j][k] -= epslion
            print("W[{i}]: ".format(i=i))
            print(1,"神经网络梯度",self.gc_dw[-1-i])
            print(2,"梯度检验",dw)
            print(3,"差",self.gc_dw[-1-i]-dw)
            print(4,'差方',(self.gc_dw[-1 - i] - dw) * (self.gc_dw[-1 - i] - dw) )
            # print(  (self.gc_dw[-1 - i]) * (self.gc_dw[-1 - i]) + (dw) * (dw))
            print(5,(self.gc_dw[-1-i]-dw)*(self.gc_dw[-1-i]-dw)/((self.gc_dw[-1-i])*(self.gc_dw[-1-i])+(dw)*(dw)))
            exit()
            #print((self.gc_dw[-1-i]-dw)**2,np.linalg.norm(self.gc_dw[-1-i],dw))

    def gc_B(self, index):
        epslion = 1e-7
        for i in range(len(self.B)):
            db = np.zeros(self.B[i].shape)
            for j in range(len(self.B[i])):
                for k in range(len(self.B[i][j])):
                    self.B[i][j][k] -= epslion
                    J_minus = self.gc_FP(index)
                    self.B[i][j][k] += 2 * epslion
                    J_plus = self.gc_FP(index)
                    db[j][k] = (J_plus - J_minus) / (2 * epslion)
                    self.B[i][j][k] -= epslion
            print("W[{i}]: ".format(i=i))
            print(1, "神经网络梯度", self.gc_db[-1 - i])
            print(2, "梯度检验", db)
            print(3, "差", self.gc_db[-1 - i] - db)
            print(4, '差方', (self.gc_db[-1 - i] - db) * (self.gc_db[-1 - i] - db))
            # print(  (self.gc_dw[-1 - i]) * (self.gc_dw[-1 - i]) + (dw) * (dw))
            print(5, (self.gc_db[-1 - i] - db) * (self.gc_db[-1 - i] - db) / (
                        (self.gc_db[-1 - i]) * (self.gc_db[-1 - i]) + (db) * (db)))
            exit()
            # print((self.gc_dw[-1-i]-dw)**2,np.linalg.norm(self.gc_dw[-1-i],dw))

    # 正式训练
    def Train(self, train_num):
        self.tra_dim = self.struct[0]
        for iteration in range(train_num):
            # 学习率衰减
            self.alpha *= 0.95 ** iteration
            for batch in range(len(self.train_x)):
                self.F_P(batch)
                self.gc_W(batch)
                #self.gc_B(batch)




if __name__ == '__main__':
    # struct是网络结构，本网络一共5层，每层分别3，3，5，2，1个神经元,
    struct = [3,5,4,2, 1]
    # super parameters
    L = len(struct)
    alpha = 0.001
    # 创建模型
    model = NN(alpha, struct=struct, BatchSize=1)
    model.Genernate_Train_Data_batch(data_num=1000)
    model.Train(train_num=10)