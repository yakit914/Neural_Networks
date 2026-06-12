from Layer import Layer
import numpy as np 


class ReLU(Layer):
    def forward(self,inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self,inputs):
        self.input = inputs
        self.output = np.where(inputs > 0,inputs,self.alpha*inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] *= self.alpha

class Softmax(Layer):
    def forward(self, inputs):
        self.input = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.zeros_like(dvalues)
        s = self.output
        v = np.sum(s * dvalues, axis=1, keepdims=True)   
        self.dinputs = s * (dvalues - v)                 
        
# Test
if __name__ == "__main__":
    np.random.seed(0)


    print("---ReLU Test---")
    relu = ReLU()
    X = np.array([[-1.5, 0.0, 2.0],
                  [3.0, -0.1, 0.0]])
    print("Input:\n", X)

    # Forward
    relu.forwards(X)
    print("\nForward output (ReLU):\n", relu.output)

    # Should output:
    # [[0. 0. 2.]
    #  [3. 0. 0.]]

    dvalues = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])

    # Backward
    relu.backward(dvalues)
    print("\nGradient for the previous layer :\n", relu.dinputs)

    # should output
    # [[0. 0. 3.]
    #  [4. 0. 0.]]
