import numpy as np




class Module(object):
    
    def __init__(self):
        self._params = None
        self._grad = None
        
         
        
    def forward(self,data):
         pass
     
    
    def zero_grad(self):
        pass
    
    
    def backward_update_gradient(self,input,delta):
        pass
    
    
    def backward_delta(self,input,delta):
        pass
    
    def update_parameters(self,gradient_step=1e-3):
        self._params -= gradient_step*self._grad
    
    
    
    

class Linear(Module):
    
    def __init__(self,input,output,bias=True,type=2):
        self.input = input
        self.output = output
        
        self._params = np.random.random((input,output))
        self._grad = np.zeros((input,output)) 
    
        if type == 0:
            self._params = np.random.random((input,output)) - 0.5
        if type == 1:
            self._params = np.random.normal(0, 1,(input,output))
        if type == 2:
            self._params = np.random.normal(0, 1,(input,output))*np.sqrt(2/(input+output))
            
        if bias == True:
            if type == 0:
                self._bias = np.random.random((1,output)) - 0.5
            if type == 1:
                self._bias = np.random.normal(0, 1,(1,output))
            if type == 2:
                self._bias = np.random.normal(0, 1,(1,output))*np.sqrt(2/(input+output))
            self._bias_grad = np.zeros((1, output))
        else:
            self._bias = None
            
            
            
    def forward(self,data):
      
        if not (data.shape[1] == self.input) :
            print(data)
            print(self.input)
            print(data.shape[1])
        
        if self._bias is not None:
            
            return np.dot(data,self._params) + self._bias
        
        return np.dot(data,self._params)
    
    def zero_grad(self):
        self._grad = np.zeros((self.input,self.output))
        
    def backward_delta(self,input,delta):
        return np.dot(delta,self._params.T)
        
    
    def backward_update_gradient(self, input, delta):
        
        self._grad += np.dot(input.T, delta)
        
        if self._bias is not None:
            self._bias_grad += np.sum(delta, axis = 0)
            


class TanH(Module):
    def __init__(self):
        pass
    
    def forward(self,input):
        
        return np.tanh(input) 
    
    def backward_delta(self,input,delta):
        return (1 - np.tanh(input)**2) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass
    
    
    
class Sigmoide(Module):
    
    def __init__(self):
        pass
    
    def forward(self,input):
        return 1 / (1 + np.exp(-input))
    
    
    def backward_delta(self,input,delta):
        d = 1 / (1 + np.exp(-input))
        return  (d*(1-d))*delta
    
    
    def update_parameters(self, gradient_step=1e-3):
        pass
    
    

class Softmax(Module):
    
    def forward(self, X):
        expo = np.exp(X)
        return expo / np.sum(expo, axis=1).reshape((-1,1))

    def backward_delta(self, input, delta):
        
        expo = np.exp(input)
        out = expo / np.sum(expo, axis=1).reshape((-1,1))
        return delta * (out * (1-out))
    
    def update_parameters(self, gradient_step=1e-3):
        pass
    
    