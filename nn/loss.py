
import numpy as np


class Loss(object):
    
    def forward(self, y, yhat):
        return 

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    
    def forward(self, y, yhat):
        
        assert(y.shape == yhat.shape)
        return np.sum((y - yhat) ** 2,axis = 1)
    
    def backward(self, y, yhat):
        
        assert(y.shape == yhat.shape)
        return -2*(y - yhat) 
    


class CELoss(Loss):
    
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)
        return 1 - np.sum(yhat * y, axis = 1)
    
    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        return yhat-y
    
    
    
class BCELoss(Loss):
    
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        return -(y*np.log(yhat + 1e-100) + (1-y)*np.log(1-yhat+ 1e-100))
    
    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        return ((1-y)/(1-yhat+ 1e-100)) - (y/yhat+ 1e-100)
    
    
class CElogSoftMax(Loss):
    
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return np.log(np.sum(np.exp(yhat), axis=1)+ 1e-100) - np.sum(y * yhat,axis = 1)

    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        expo = np.exp(yhat)
        return expo / np.sum(expo, axis=1).reshape((-1,1)) - y
    
    
    
    
