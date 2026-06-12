import numpy as np
from Layer import *
from Loss import *
from Activation import *
from Optimizer import *
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers = []):
        self.layers = layers
        self.loss = None
        self.optimizer = None
        
    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_x):
        X = input_x
        for layer in self.layers:
            layer.forward(X)
            X = layer.output
        return X
    
    def compute_loss(self, output, y):
        if self.loss is None:
            raise ValueError("Loss function not defined.")
        return self.loss.calculate(output, y)
    
    def backward(self, output, y):
        last_layer = self.layers[-1]
        if isinstance(last_layer, Softmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            dinputs = output.copy()
            samples = dinputs.shape[0]
            if y.ndim == 1:
                dinputs[range(samples), y] -= 1
            else:
                dinputs -= y
            dinputs = dinputs / samples
        else:
            if hasattr(self.loss, 'backward'):
                dinputs = self.loss.backward(output, y)
            else:
                raise RuntimeError("Loss function backward not defined")

        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def classification(self, output):
        return np.argmax(output, axis=1)
    
    
    
    def predict(self, X):
        output = self.forward(X)
        if isinstance(self.loss, LossMeanSquaredError) or isinstance(self.loss, LossMeanAbsoluteError):
            return output  # For regression, return raw output
        else:
            return self.classification(output)  # For classification, use argmax
    

    def accuracy(self, output, y):
        predictions = self.classification(output)
        return np.mean(predictions == y)
    
    
    def update_params(self, learning_rate=0.01):
        if self.optimizer is not None:
            for layer in self.layers:
                if isinstance(self.optimizer,Optimizer):
                      self.optimizer.update_params(layer)
        else:
            #default when no optimizer specified
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.weights += -learning_rate * layer.dweights
                    layer.biases  += -learning_rate * layer.dbiases

    def train_batch(self, X, y, learning_rate=0.01):
        output = self.forward(X)
        loss = self.compute_loss(output, y)
        acc = self.accuracy(output, y)
        self.backward(output, y)
        self.update_params(learning_rate)
        return loss, acc
    
    def test_batch(self, X, y):
        output = self.forward(X)
        loss = self.compute_loss(output, y)
        acc = self.accuracy(output, y)
        return loss, acc
    
    def train(self, X, y, epochs=1,learning_rate = 0.01, batch_size=None, validation_ratio = 0, shuffle = True, batch_shuffle = True):

        # Ensure X and y must have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        history = [] #To store loss and acc of each epoch
        max_epochs = int(epochs)
    
        # Reset Optimizer before training start
        if self.optimizer is not None:
            self.optimizer.reset()
            self.optimizer.set_learning_rate(learning_rate)

        if not 0 <= validation_ratio <= 1: 
            raise ValueError("validation_ratio must be between 0.0 and 1.0")

        # shuffle data if shuffle is True
        if shuffle:
            num_samples = len(X)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]


        # split into training and validation data set base on validation_ratio
        if validation_ratio > 0:
            split_index = int((1 - validation_ratio) * len(X))
            X_train, y_train = X[:split_index], y[:split_index]
            X_val, y_val = X[split_index:], y[split_index:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Initial Performance
        loss, acc = self.test_batch(X_train, y_train)
        initial_point = {}
        if validation_ratio > 0:
            val_loss, val_acc = self.test_batch(X_val, y_val)
            print(f"Epoch {0:>3}/{max_epochs:<4} | Loss: {loss:.4f}, Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            initial_point["Epoch"] = 0
            initial_point["Loss"] = loss
            initial_point["Accuracy"] = acc
            initial_point["Val_Loss"] = val_loss
            initial_point["Val_Accuracy"] = val_acc
                
        else:
            print(f"Epoch {0:>3}/{max_epochs:<4} | Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            initial_point["Epoch"] = 0
            initial_point["Loss"] = loss
            initial_point["Accuracy"] = acc
        history.append(initial_point)

        # Training Loop
        for epoch in range(max_epochs):
            # Full batch or mini-batch base on batch_size
            if batch_size is None:
                loss, acc = self.train_batch(X_train, y_train, learning_rate)

            else:
                # mini batch
                num_samples = len(X_train)
                if batch_shuffle:
                    indices = np.arange(num_samples)
                    np.random.shuffle(indices)
                    X_train_Epoch = X_train[indices]
                    y_train_Epoch = y_train[indices]
                else:
                    X_train_Epoch = X_train
                    y_train_Epoch = y_train

                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_train_Epoch[start:end]
                    y_batch = y_train_Epoch[start:end]

                    loss, acc = self.train_batch(X_batch, y_batch, learning_rate)
                    batch_progress = (start/num_samples)
                    bar_length = 22
                    print(f"Batch Progress | [{'':{'|'}<{int(batch_progress * bar_length)}}{'':{' '}<{bar_length - int(batch_progress * bar_length)}}]-{batch_progress * 100 :>4.1f}% | Learning Rate: {self.optimizer.learning_rate:.4f}", end='\r', flush=True)

            Point = {}
            if validation_ratio > 0:
                # Compute validation loss and accuracy for current model on the validation set
                val_loss, val_acc = self.test_batch(X_val, y_val)
                print(f"Epoch {epoch+1:>3}/{max_epochs:<4} | Loss: {loss:.4f}, Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                Point["Epoch"] = epoch+1
                Point["Loss"] = loss
                Point["Accuracy"] = acc
                Point["Val_Loss"] = val_loss
                Point["Val_Accuracy"] = val_acc
                
            else:
                print(f"Epoch {epoch+1:>3}/{max_epochs:<4} | Loss: {loss:.4f}, Accuracy: {acc:.4f}")
                Point["Epoch"] = epoch+1
                Point["Loss"] = loss
                Point["Accuracy"] = acc

            history.append(Point)

        print(f"Training completed")
        return history
        
        

        



