import numpy as np

class Optimizer_Adagrad:
    def __init__(self, lr=1.0, decay=0.0, epsilon=1e-7):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer,"weight_caches"):
            layer.weight_caches = np.zeros_like(layer.weights)
            layer.bias_caches = np.zeros_like(layer.biases)
            
        layer.weight_caches += layer.dweights ** 2
        layer.bias_caches += layer.dbiases ** 2
        
        layer.weight += -self.current_lr *\
            layer.dweights / (np.sqrt(layer.weight_caches) + self.epsilon)
        
        layer.biases += -self.current_lr *\
            layer.dbiases / (np.sqrt(layer.bias_caches) + self.epsilon)
        
    

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, lr=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_caches"):
            layer.weight_caches = np.zeros_like(layer.weights)
            layer.bias_caches = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        layer.weight_momentums = (self.beta1 * layer.weight_momentums +
                                  (1 - self.beta1) * layer.dweights)
        layer.bias_momentums = (self.beta1 * layer.bias_momentums +
                                (1 - self.beta1) * layer.dbiases)
        
        layer.weight_caches = (self.beta2 * layer.weight_caches +
                               (1 - self.beta2) * (layer.dweights ** 2))
        layer.bias_caches = (self.beta2 * layer.bias_caches +
                             (1 - self.beta2) * (layer.dbiases ** 2))
        
        weight_caches_corrected = layer.weight_caches / (1 - self.beta2 ** (self.iterations + 1))
        bias_caches_corrected = layer.bias_caches / (1 - self.beta2 ** (self.iterations + 1))
        
        layer.weights += -self.current_lr * weight_caches_corrected / \
            (np.sqrt(weight_caches_corrected) + self.epsilon)
        layer.biases += -self.current_lr * bias_caches_corrected / \
            (np.sqrt(bias_caches_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Rmsprop:
    def __init__(self, lr=1.0, decay=0.0, epsilon=1e-7 , rho = 0.9):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_caches"):
            layer.weight_caches = np.zeros_like(layer.weights)
            layer.bias_caches = np.zeros_like(layer.biases)
        
        layer.weight_caches = self.rho * layer.weight_caches + (1 - self.rho) * layer.dweights ** 2
        layer.bias_caches = self.rho * layer.bias_caches + (1 - self.rho) * layer.dbiases ** 2
        
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_caches) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_caches) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD:
    def __init__(self , lr=1. , decay=0. , momentum=0.):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_momentums = self.momentum * layer.weight_momentums -\
                self.current_lr * layer.dweights
            layer.bias_momentums = self.momentum * layer.bias_momentums -\
                self.current_lr * layer.dbiases
        else:
            layer.weights -= self.current_lr * layer.dweights
            layer.biases -= self.current_lr * layer.dbiases
        
        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums
    
    def post_update_params(self):
        self.iterations += 1
