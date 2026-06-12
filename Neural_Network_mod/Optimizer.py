from Layer import *
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def set_learning_rate(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def reset(self):
        pass

    def update_params(self, layer):
        raise NotImplementedError("Optimizer Not implemented")
    

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        
    def update_params(self, layer):
        if isinstance(layer, Dense):
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases

class SGDWithDecay(Optimizer):
    def __init__(self, learning_rate=0.01,  decay=1e-7):
        super().__init__()
        self.initial_learning_rate = learning_rate
        self.iterations = 0
        self.decay = decay

    def set_learning_rate(self, learning_rate=0.01):
        self.initial_learning_rate = learning_rate

    def reset(self):
        self.iterations = 0
        
    def update_params(self, layer):
        if isinstance(layer, Dense):
            self.learning_rate = self.initial_learning_rate * (1. / (1.+ self.decay * self.iterations))
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases
            self.iterations += 1