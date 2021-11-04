import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from nn.module import Linear
from nn.module import TanH
from nn.module import Sigmoide


from nn.loss import MSELoss

from tools.basic import *






nb_itr = 10000
loss =[]






    
    
    
"""pred = lin_layer1.forward(x)
    
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
plt.show()"""


batchsize = 1000

datax, datay = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=1, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=1, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

n = datax.shape[1]
d = 1

type=2

iteration = 1500
gradient_step = 1e-5

loss_mse = MSELoss()
lin_layer1 = Linear(n, 60,type=1)
tanh_layer = TanH()
sig_layer = Sigmoide()
lin_layer2 = Linear(60, d,type=1)

for _ in range(nb_itr) :
    
    
    """FORWARD"""
    h_lin_layer1 = lin_layer1.forward(datax)
    h_tan_layer = tanh_layer.forward(h_lin_layer1)
    h_lin_layer2 = lin_layer2.forward(h_tan_layer)
    h_sig_layer = sig_layer.forward(h_lin_layer2)
    l = loss_mse.forward(datay,h_sig_layer).mean()
    loss.append(l)
    
    
    
    
    """BACKWARD"""
    loss_delta = loss_mse.backward(datay,h_sig_layer)
    sig_delta = sig_layer.backward_delta(h_lin_layer2,loss_delta)
    
    lin2_delta = lin_layer2.backward_delta(h_tan_layer,sig_delta)
    lin_layer2.backward_update_gradient(h_tan_layer,sig_delta)
    lin_layer2.update_parameters(gradient_step=gradient_step)
    
    tan_delta = tanh_layer.backward_delta(h_lin_layer1,lin2_delta)
    
    lin1_delta = lin_layer1.backward_delta(datax,tan_delta)
    lin_layer1.backward_update_gradient(datax,tan_delta)
    lin_layer1.update_parameters(gradient_step=gradient_step)
    
    

    lin_layer1.zero_grad()
    lin_layer2.zero_grad()






def predict(x):
    y_hat = sig_layer.forward(lin_layer2.forward(tanh_layer.forward(lin_layer1.forward(x))))
    
    return np.where(y_hat >= 0.5,1, 0)

acc = np.where(testy == predict(testx),1,0).mean()

print("accuracy : ",acc)


plt.figure()
plot_frontiere(testx, predict, step=100)
plot_data(testx, testy.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()

