#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import  load_breast_cancer

class NN():

    def __init__(self, alpha, struct, Batchsize):
        # Super parameters
        self.BatchSize=Batchsize
        self.struct=struct
        self.L=len(struct)
        self.alpha=alpha
        self.loss=[]
        self.lmda=10
        self.W = [np.random.randn(struct[i], struct[i - 1]) / np.sqrt(self.struct[i] + self.struct[i - 1]) for i in range(1, self.L)]
        self.B = [np.random.randn(struct[i], 1)/np.sqrt(self.struct[i]+self.struct[i - 1]) for i in range(1, self.L)]
        # RMSprop Gradient decent parameters
        self.Sdw_beta = 0.99
        self.Sdw = np.array([np.zeros((struct[self.L-i],struct[self.L-i-1])) for i in range(1, self.L)],dtype=object)
        # Momentum Gradient decent parameters
        self.Vdw_beta = 0.9
        self.Vdw = np.array([np.zeros((struct[self.L-i],struct[self.L-i-1])) for i in range(1, self.L)],dtype=object)
        self.Vdb = np.array([np.zeros((struct[self.L - i], struct[self.L - i - 1])) for i in range(1, self.L)],dtype=object)

    # 数据归一化
    def Normalize_Data(self,data):
        average=np.mean(data,axis=1).reshape(30,1)
        variance=np.var(data,axis=1).reshape(30,1)
        normal_data=(data-average)/(variance)
        return normal_data

    # mini-batch 梯度下降训练集
    def Genernate_Train_Data_batch(self,data_num):
        X_data, y_data = load_breast_cancer(return_X_y=True)
        train_x, test_x, train_y, test_y = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=28)
        train_y_list = []
        train_x_list = []
        train_x = np.array(train_x).T
        for i in range(int((1 - 0.2) * data_num / 16)):
            train_y_list.append(train_y[i * self.BatchSize:i * self.BatchSize + self.BatchSize])
            train_x_list.append(np.array(train_x[:, i * self.BatchSize:i * self.BatchSize + self.BatchSize]))
        # 训练集
        self.train_y = train_y_list
        self.test_y = np.array([n for n in test_y]).T
        self.train_x = train_x_list
        self.test_x = np.array(test_x).T


    # 每个Epoch训练完之后，check一下在验证集上的正确率与loss
    def check(self):
        Z = []
        A = []
        Z.append(self.test_x)
        A.append(self.test_x)
        for i in range(L - 1):
            Z.append(self.W[i].dot(A[-1]) + self.B[i])
            A.append(self.g(Z[-1]))
        predict = A[-1]
        p_y = np.where(predict >= 0.5, 1, 0).sum(axis=0)
        # print(predict)
        # exit()
        y=predict[-1]
        all = len(self.test_y)
        J = -(self.test_y * (np.log(y)) + (1 - self.test_y) * np.log(1 - y)).sum() / (
                    self.BatchSize)
        t = 0
        for x in range(all):
            if p_y[x] == self.test_y[x]:
                t += 1
        print('val_acc:', t / len(self.test_y),'val_loss:',J,len(self.test_y),t)

    # 激活函数
    def g(self,z,diff=False):
        if diff:
            return self.g(z)*(1-self.g(z))
        else:
            return 1./(1+np.exp(-z)+1e-5)

    # 正则项
    def R(self):
        w2=0
        for w in self.W:
            w2+=np.linalg.norm(w)
        return w2

    # 前向传播
    def F_P(self,index):
        # Forward progatation
        Z=[]
        A=[]
        Z.append(self.train_x[index])
        A.append(self.train_x[index])
        for i in range(L-1):
            Z.append(self.W[i].dot(A[-1])+self.B[i])
            A.append(self.g(Z[-1]))

        # Loss Function
        y=A[-1]
        J=-(self.train_y[index]*(np.log(y))+(1-self.train_y[index])*np.log(1-y)).sum()/(self.BatchSize)
        #print('train_loss is : ',J)
        self.loss.append(J)
        return self.B_P(Z,y,A,index)

    # 反向传播
    def B_P(self,Z,y,A,index):
        # Back progatation
        dZ = []
        dA = []
        dB = []
        dW = []
        # 初始化一下
        dA.append((y-self.train_y[index])/(y*(1-y)))
        dZ.append(y-self.train_y[index])
        # 开始反向传播计算每一层权重的梯度
        for i in range(L-1):
            dA.append(self.W[-i-1].T.dot(dZ[-1]))
            dB.append(dZ[-1].sum(axis=1, keepdims=True))
            dW.append(dZ[-1].dot(A[-2-i].T))
            dZ.append(dA[-1]*self.g(Z[-i-2],diff=True))
        dZ[-1]=dA[-1]
        # 梯度下降，每个batch结束都会梯度下降一次
        dW=np.array(dW,dtype=object)
        dB = np.array(dB, dtype=object)
        #print(dW)
        #self.Adam(dW,index+1)
        self.Gradient_decent(dW,dB)

    # 普通梯度下降
    def Gradient_decent(self,dW,dB):
        for i in range(L-1):
            self.W[i] -= self.alpha*dW[-i-1]/self.BatchSize
            self.B[i] -= self.alpha * dB[-i - 1]/self.BatchSize

    # Momentum梯度下降
    def Momentum_Gradient_decent(self, dW,dB):
        self.Vdw=self.Vdw_beta*self.Vdw+(1-self.Vdw_beta)*dW
        self.Vdb = self.Vdb_beta * self.Vdb + (1 - self.Vdb_beta) * dB
        for i in range(L - 1):
            self.W[i] -= self.alpha * self.Vdw[-i-1]
            self.B[i] -= self.alpha * self.Vdw[-i - 1]

    # RMSprop梯度下降
    def RMSprop_Gradient_decent(self, dW):
        self.Sdw=self.Sdw_beta*self.Sdw+(1-self.Sdw_beta)*(dW*dW)
        for i in range(L - 1):
            self.W[i] -= self.alpha * (dW[-i-1]/(np.sqrt(self.Sdw[-i-1])+1e-8))

    # Adam
    def Adam(self, dW, t):
        self.Vdw = self.Vdw_beta * self.Vdw + (1 - self.Vdw_beta) * dW
        self.Sdw = self.Sdw_beta * self.Sdw + (1 - self.Sdw_beta) * (dW * dW)
        for i in range(L - 1):
            self.W[i] -= self.alpha * (self.Vdw[-i-1])/(np.sqrt(self.Sdw[-i-1])+1e-8)*\
                         ((1-self.Sdw_beta**t)/(1-self.Vdw_beta**t))

    # 正式训练
    def Train(self,train_num):
        self.tra_dim=self.struct[0]
        for iteration in range(train_num):
            # # 学习率衰减
            # self.alpha *= 0.99**np.sqrt(iteration)
            for batch in range(len(self.train_x)):
                self.F_P(batch)
            if iteration%1000==999:
                self.check()

    # 画损失函数值（训练阶段）
    def Plot_Loss(self):
        plt.figure()
        plt.plot(self.loss)
        plt.show()

    # 测试准确率
    def Test(self,test_data_num):
            self.Test_x = np.random.randint(0, 10, (self.tra_dim, test_data_num)).astype('float64')
            self.Test_y = np.where(self.Test_x.sum(axis=0) > 15, 1, 0)
            predict=self.Test_x
            for i in self.W:
                predict=i.dot(predict)
            p_y=np.where(predict>0.5,1,0).sum(axis=0)
            all=len(self.Test_y)
            t=0
            t1=0
            t0=0
            n1=0
            n0=0
            xx=0
            for x in range(all):
                if p_y[x]==self.Test_y[x]:
                    t+=1
                if p_y[x]==1 and self.Test_y[x]==1:
                    t1+=1
                if p_y[x]==0 and self.Test_y[x]==0:
                    t0+=1
                if p_y[x]==1:
                    n1+=1
                if p_y[x]==0:
                    n0+=1
                if self.Test_y[x]==1:
                    xx+=1
            # print(self.Test_x)
            # print('精确率:1,0 is ',t1/n1,t0/n0)
            # print('召回率：1,0 is ',t1/xx,t0/(self.tes_num-xx))
            print('准确率 ：',t/test_data_num)

if __name__ == '__main__':
    # struct是网络结构，本网络一共5层，每层分别3，3，5，2，1个神经元,
    struct =[30,10,5,2,1]
    # super parameters
    L = len(struct)
    alpha = 0.001
    # 创建模型
    model=NN(alpha,struct=struct,Batchsize=8)
    model.Genernate_Train_Data_batch(data_num=569)
    # 训练10次
    model.Train(train_num=30000)
    model.Plot_Loss()
    # # 训练50个Epoch
    # model.Test(test_data_num=50)
