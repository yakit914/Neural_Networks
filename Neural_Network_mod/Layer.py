import numpy as np 

class Layer:
    def __init__(self):
        self.input = None
        self.output = None



class Dense(Layer):
    def __init__(self,n_input,n_neurons):
        super().__init__()
        self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2 / n_input)
        self.biases =  np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.input = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)



# Test
if __name__ == "__main__":
    learning_rate = 1
    np.random.seed(0)
    dense1 = Dense(3, 3)

    X = np.array([[-1.5, 0.0, 2.0],
                  [3.0, -0.1, 0.0]])
    print("Input:\n", X)

    # Forward
    dense1.forward(X)
    print("\nForward output:\n", dense1.output)

    dvalues = np.array([[1.0, 0.5, -1.0],
                        [2.0, -0.5, 3.0]])

    # Backward
    dense1.backward(dvalues)

    dense1.weights += -learning_rate * dense1.dweights
    dense1.biases  += -learning_rate * dense1.dbiases
    print("\nGradient for the previous layer :\n", dense1.dinputs)
    print("\nUpdated weights:\n", dense1.weights)
    print("\nUpdated biases:\n", dense1.biases)
