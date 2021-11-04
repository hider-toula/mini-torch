import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from nn.module import Linear
from nn.loss import MSELoss

from tools.basic import *


a=50
b=10

x= np.random.uniform(-5,5,50).reshape((-1,1))
y = a * x + b +( np.random.uniform(-100,100,50).reshape((-1,1)) ) 

n = x.shape[1]
d = 1


iteration = 100
gradient_step = 1e-4

loss_mse = MSELoss()
lin_layer = Linear(n, d,type=1)



nb_itr = 10000
loss =[]

for _ in range(nb_itr) :
    
    hidden_layer = lin_layer.forward(x)
    l = loss_mse.forward(y,hidden_layer).mean()
    loss.append(l)
    loss_back = loss_mse.backward(y,hidden_layer)
    delta_linear = lin_layer.backward_delta(x,loss_back)
    lin_layer.backward_update_gradient(x, loss_back)
    lin_layer.update_parameters(gradient_step=gradient_step)
    lin_layer.zero_grad()
    
    
    
pred = lin_layer.forward(x)
    
plt.figure()
plt.scatter(x,y,label="data",color='black')
plt.plot(x,pred,color='red',label='predection')



plt.figure()
plt.scatter(x,y,label="data",color='black')
plt.plot(x,pred,color='red',label='predection')



for i in range(len(x)):
    plt.plot([x[i],x[i]],[y[i], pred[i]], c="blue", linewidth=1)
plt.legend()
plt.xlabel("datax")
plt.ylabel("datay")
plt.title("prediction ligne for ax+b")
plt.show()

plt.figure()
plt.plot(np.arange(nb_itr),loss)
plt.title("loss on each iteration for ax+b")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
plt.show()


batchsize = 1000

datax, datay = gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
d = 1

type=2

iteration = 150000
gradient_step = 1e-5

loss_mse = MSELoss()
lin_layer = Linear(n, d,type=type)

for _ in range(iteration):
    #forward
    
    hidden_l = lin_layer.forward(datax)
    
    
    
    #backward
    loss_back = loss_mse.backward(datay, hidden_l)
    delta_linear = lin_layer.backward_delta(datax, loss_back)

    lin_layer.backward_update_gradient(datax, loss_back)
    
    lin_layer.update_parameters(gradient_step=gradient_step)
    
    lin_layer.zero_grad()


def predict(x):
    hidden_l = lin_layer.forward(x)
    return np.where(hidden_l >= 0.5,1, 0)


acc = np.where(testy == predict(testx),1,0).mean()
print("accuracy : ",acc)
plt.figure()
plot_frontiere(testx, predict, step=100)
plot_data(testx, testy.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()