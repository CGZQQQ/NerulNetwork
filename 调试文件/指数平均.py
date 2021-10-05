import numpy as np
import matplotlib.pyplot as plt
x=np.random.randint(1,20,500)
beta=0.90
beta1=0.5
print()
print(x)
a=0
b=0
c=0
A=[]
B=[]
a_b=[]
av=[]
A.append(0)
B.append(0)
a_b.append(0)
for i in range(len(x)):
    a=a*beta+(1-beta)*x[i]
    b=b*beta1+(1-beta1)*x[i]
    c=(c*i+x[i])/(1+i)
    av.append(c)
    A.append(a)
    B.append(b)
    a_b.append(abs(a-b))
plt.subplot(131)
plt.plot(A)
plt.legend('beta=0.9')
print(A)
plt.ylim((0,20))
# plt.plot(B)
# plt.plot(a_b)
plt.subplot(132)
plt.plot(av)
plt.legend('平均值')
plt.ylim((0,20))
plt.subplot(133)
plt.plot(B)
# plt.legend(['beta=0.9','beta=0.5','a_b'])
plt.ylim((0,20))
plt.legend('beta=0.5')
plt.show()
