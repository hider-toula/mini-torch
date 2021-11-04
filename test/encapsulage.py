
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.module import Linear,TanH,Sigmoide
from nn.loss import MSELoss
from tools.basic import *
from tools.nntools import Sequentiel,Optim
import matplotlib.pyplot as plt



size = 500

datax, datay = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=size, data_type=2, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=size, data_type=2, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
hidden = 80
hidden2 = 60
d = 1


iteration = 500
gradient_step = 1e-3
batchsize=10

def label_func(x):
    return np.where(x >= 0.5,1, 0)
    
loss_mse = MSELoss()
lin_layer = Linear(n, hidden,type=1)
lin_layer2 = Linear(hidden, hidden2,type=1)
lin_layer3 = Linear(hidden2, d,type=1)
lin_layer4 = Linear(40, 30,type=1)
lin_layer5 = Linear(30, d,type=1)
act_sig = Sigmoide()
act_tan = TanH()

net = Sequentiel([lin_layer,act_tan,lin_layer2,act_tan,lin_layer3,act_sig],labels=label_func)


opt = Optim(net,loss_mse,eps=gradient_step)
mean, std = opt.SGD(datax,datay,batchsize,iteration)
plt.figure()
plt.plot(mean)
plt.plot(std)
plt.legend(('Mean', 'std'))
plt.show()


acc = opt.score(datax,datay)
print("accuracy : ",acc)
plt.figure()
plot_frontiere(datax, opt._net.predict, step=100)
plot_data(datax, datay.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()