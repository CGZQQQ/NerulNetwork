import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class NN():
    def __init__(self,alpha,struct,BatchSize):
        # Super parameters
        self.BatchSize=BatchSize
        self.struct = struct
        self.L = len(struct)
        self.alpha = alpha
        self.loss = []
        self.lmda = 10
        # RMSProp Gradient decent parameters
        self.Sdw_beta = 0.99
        self.Sdw = np.array([np.zeros((struct[self.L-i], struct[self.L-i-1])) for i in range(1, self.L)], dtype=object)
        self.Sd_BN_beta = np.array([np.zeros((self.struct[self.L-2-i], 1)) for i in range(self.L-2)], dtype=object)
        self.Sd_BN_gama = np.array([np.zeros((self.struct[self.L-2-i], 1)) for i in range(self.L-2)], dtype=object)
        # Momentum Gradient decent parameters
        self.Vdw_beta = 0.9
        self.Vdw = np.array([np.zeros((struct[self.L-i],struct[self.L-i-1])) for i in range(1, self.L)],dtype=object)
        self.Vd_BN_beta = np.array([np.zeros((self.struct[self.L-2-i], 1)) for i in range(self.L - 2)], dtype=object)
        self.Vd_BN_gama = np.array([np.zeros((self.struct[self.L-2-i], 1)) for i in range(self.L - 2)], dtype=object)
        # Batch Normalization parameters
        self.BN_gama=np.array([np.random.randn(self.struct[i+1],1) for i in range(self.L-2)],dtype=object)
        self.BN_beta=np.array([np.random.randn(self.struct[i+1],1) for i in range(self.L-2)],dtype=object)
        self.BN_sigma=[]
        self.BN_mu=[]
        self.I=np.identity(self.BatchSize)
    # 数据归一化
    def Normalize_Data(self,data):
        average=np.mean(data,axis=1).reshape(3,1)
        variance=np.var(data,axis=1).reshape(3,1)
        normal_data=(data-average)/(variance)
        return normal_data

    # mini-batch 梯度下降训练集
    def Genernate_Train_Data_batch(self,data_num=1000):
        df = pd.read_csv('train_data.csv')
        X = df['data'][0:data_num]
        Y = df['label'][0:data_num]
        t_s = 0.2
        # 前期准备
        train_x, test_x, train_y, test_y = \
            train_test_split(X, Y, test_size=t_s, random_state=42)
        train_y_list=[]
        train_x_list=[]
        self.tra_num = self.BatchSize
        train_x1 = np.array([[int(i[1]), int(i[3]), int(i[5])] for i in train_x]).T
        # 归一化输入
        train_x1=self.Normalize_Data(train_x1)
        for i in range(int((1-t_s)*data_num/self.BatchSize)):
            train_y_list.append(np.array([train_y[i*self.BatchSize:i*self.BatchSize+self.BatchSize]]))
            train_x_list.append(np.array(train_x1[:,i*self.BatchSize:i*self.BatchSize+self.BatchSize]))
        # 训练集
        self.train_y = train_y_list
        self.test_y = np.array([n for n in test_y]).T
        self.train_x=train_x_list
        self.test_x=np.array([[int(i[1]),int(i[3]),int(i[5])] for i in test_x]).T

    # 每个Epoch训练完之后，check一下在验证集上的正确率与loss
    def check(self):
        predict = self.test_x
        for i in self.W:
            predict = self.g(i.dot(predict))
        p_y = np.where(predict > 0.5, 1, 0).sum(axis=0)
        y=predict[-1]
        all = len(self.test_y)
        J = -(self.test_y * (np.log(y)) + (1 - self.test_y) * np.log(1 - y)).sum() / (
                    self.tra_num * 2) + self.lmda * self.R()
        t = 0
        for x in range(all):
            if p_y[x] == self.test_y[x]:
                t += 1
        print('val_acc:', t / len(self.test_y),'val_loss:',J,len(self.test_y),t)

    # 激活函数
    def g(self, z, diff=False):
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

    # Batch_Normalization
    def BatchNormalization(self,z,layer):
        if layer<self.L-2:
            average=np.mean(z,axis=1,keepdims=True)
            variance=np.var(z,axis=1,keepdims=True)
            self.BN_mu.append(average)
            self.BN_sigma.append(variance)
            z_normal=(z-average)/(np.sqrt(variance+1e-8))
            return z_normal
        else:
            return z

    # 前向传播
    '''神经网络每一层都会有Z，dZ，A，dA，有L-2层会有BN的scale，shift参数'''
    def F_P(self,index):
        # Forward progatation
        Z=[]
        Z_Batch_Normalization=[]
        Yi=[]
        A=[]
        Z.append(self.train_x[index])
        Z_Batch_Normalization.append(self.train_x[index])
        Yi.append(self.train_x[index])
        A.append(self.train_x[index])
        for i in range(self.L-1):
            # 用BN之后的Z代替Z
            Z.append(self.W[i].dot(A[-1]))
            z_n=self.BatchNormalization(Z[-1],layer=i)
            Z_Batch_Normalization.append(z_n)
            Yi.append(self.BN_gama[i] * z_n + self.BN_beta[i] if i<self.L-2 else z_n)
            # BN之后在进入激活函数
            A.append(self.g(Yi[-1]))
        # Loss Function
        y=A[-1]
        J=-(self.train_y[index]*(np.log(y))+(1-self.train_y[index])*np.log(1-y)).sum()/(self.tra_num*2)+self.lmda*self.R()
        print('train_loss is : ',J)
        self.loss.append(J)
        return self.B_P(Yi,Z,y,index)

    # 计算最难梯度z_norm对z的导数
    def dZN2Z(self,sigma,z,mu,nl):
        temp=np.outer(z-mu,(z-mu).T).reshape(nl,self.BatchSize,self.BatchSize,nl).sum(axis=0) / nl
        d=np.array((1 - 1 / self.BatchSize) / sigma - temp.sum(axis=0).T / (self.BatchSize * sigma ** 3))
        print(d.shape,nl,'duibudui')
        return d

    # 反向传播
    def B_P(self,Yi,Z,y,index):
        # Back progatation
        dZ=[]
        dZ_N=[]
        dA=[]
        dW=[]
        dYi=[]
        d_BN_beta=[]
        d_BN_gama=[]
        # 初始化一下
        dA.append((y-self.train_y[index])/(y*(1-y)))
        dZ.append(y-self.train_y[index])
        # 开始反向传播计算每一层权重的梯度，这里我没有加偏置
        for i in range(self.L-1):
            # 神经网络的梯度
            dA.append(self.W[-i-1].T.dot(dZ[-1]))
            dW.append(dZ[-1].dot(dA[-1].T) / self.BatchSize)
            # 下面其实就是计算BN算法下的dZ
            dYi.append(dA[-1]*self.g(Yi[-2-i],diff=True))
            dZ_N.append(dYi[-1]*self.BN_gama[-1-i])
            dZ.append(dZ_N[-1]*self.dZN2Z(self.BN_sigma[-1-i],Z[-i-2],self.BN_mu[-1-i],self.struct[-2-i]))
            # BN中两个参数的梯度
            if i<self.L-2:
                d_BN_beta.append(np.mean(dZ[-1]*1,axis=1,keepdims=True))
                d_BN_gama.append(np.mean(dZ[-1]*Z[L-2-i],axis=1,keepdims=True))
        # 梯度下降，每个batch结束都会梯度下降一次
        dW = np.array(dW,dtype=object)
        d_BN_gama=np.array(d_BN_gama,dtype=object)
        d_BN_beta=np.array(d_BN_beta,dtype=object)
        # 权重梯度下降
        self.Adam(dW,index+1)
        # BN参数梯度下降
        self.Adam_BN(d_BN_gama,d_BN_beta,index+1)

    # Adam
    def Adam(self, dW, t):
        self.Vdw = self.Vdw_beta * self.Vdw + (1 - self.Vdw_beta) * dW
        self.Sdw = self.Sdw_beta * self.Sdw + (1 - self.Sdw_beta) * (dW * dW)
        for i in range(self.L - 1):
            self.W[i] -= self.alpha * (self.Vdw[-i-1])/(np.sqrt(self.Sdw[-i-1])+1e-8)*\
                         ((1-self.Sdw_beta**t)/(1-self.Vdw_beta**t))

    def Adam_BN(self, d_BN_gama, d_BN_beta, t):
        # 更新gama
        self.Vd_BN_gama = self.Vdw_beta * self.Vd_BN_gama + (1 - self.Vdw_beta) * d_BN_gama
        self.Sd_BN_gama = self.Sdw_beta * self.Sd_BN_gama + (1 - self.Sdw_beta) * (d_BN_gama * d_BN_gama)
        for i in range(self.L - 2):
            self.BN_gama[i] -= self.alpha * (self.Vd_BN_gama[-i-1])/(np.sqrt(self.Sd_BN_gama[-i-1])+1e-8)*\
                         ((1-self.Sdw_beta**t)/(1-self.Vdw_beta**t))
        # 更新beta
        self.Vd_BN_beta = self.Vdw_beta * self.Vd_BN_beta + (1 - self.Vdw_beta) * d_BN_beta
        self.Sd_BN_beta = self.Sdw_beta * self.Sd_BN_beta + (1 - self.Sdw_beta) * (d_BN_beta * d_BN_beta)
        for i in range(L - 2):
            self.BN_beta[i] -= self.alpha * (self.Vd_BN_beta[-i - 1]) / (np.sqrt(self.Sd_BN_beta[-i - 1]) + 1e-8) * \
                         ((1 - self.Sdw_beta ** t) / (1 - self.Vdw_beta ** t))

    # 正式训练
    def Train(self,train_num):
        a = 0
        for i in range(len(self.struct) - 1):
            a += self.struct[i] * self.struct[i + 1]
        self.W = [np.random.randn(struct[i], struct[i - 1]) / a for i in range(1, self.L)]
        self.tra_dim=self.struct[0]
        for iteration in range(train_num):
            # 学习率衰减
            self.alpha *= 0.95**iteration
            # 批量学习
            for batch in range(len(self.train_x)):
                self.F_P(batch)


if __name__ == '__main__':
    # struct是网络结构，本网络一共5层，每层分别3，3，5，2，1个神经元,
    struct = [3, 3, 5, 5, 1]
    # super parameters
    L = len(struct)
    alpha = 0.001
    # 创建模型
    model=NN(alpha,struct=struct,BatchSize=200)
    model.Genernate_Train_Data_batch(data_num=1200)
    # 训练10次
    model.Train(train_num=1)

