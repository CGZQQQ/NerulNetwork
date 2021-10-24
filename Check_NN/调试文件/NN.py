import numpy as np

def g(z,diff=False):
    if diff:
        return g(z)*(1-g(z))
    else:
        return 1./(1+np.exp(-z))

def F_P(W):
    # Forward progatation
    Z=[]
    A=[]
    Z.append(train_x)
    A.append(train_x)
    for i in range(L-1):
        Z.append(W[i].dot(A[-1]))
        A.append(g(Z[-1]))

    # Loss Function
    y=A[-1]
    J=-(train_y*(np.log(y))+(1-train_y)*np.log(1-y)).sum()/tra_num
    print(J)
    return B_P(Z,y,W)

def B_P(Z,y,W):
    # Back progatation
    dZ=[]
    dA=[]
    dW=[]
    dA.append((y-train_y)/(y*(1-y)))
    dZ.append(y-train_y)
    for i in range(L-1):
        dA.append(W[-i-1].T.dot(dZ[-1]))
        dW.append(dZ[-1].dot(dA[-1].T))
        dZ.append(dA[-1]*g(Z[-1],diff=True))

    # gradient decent
    for i in range(L-1):
        W[i]-=alpha*dW[-i-1]
    return W

if __name__ == '__main__':
    tra_dim = 3
    tra_num = 4
    train_x = np.random.randint(0, 10, (tra_dim, tra_num))
    # train_y=np.where(train_x.sum(axis=0)<15,train_x.sum(axis=0),0)
    t_y = train_x.sum(axis=0)
    train_y = np.where(t_y > 15, 1, 0)
    alpha=0.01
    L = 5
    N = [3, 3, 5, 2, 1]
    W_1 = [np.random.randn(N[i], N[i - 1]) for i in range(1, L)]
    for i in range(1000):
        W_1=F_P(W_1)