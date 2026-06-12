import numpy as np
from Layer import Layer


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        samples = y_pred.shape[0]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

class LossMeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} do not match.")
        return (y_pred - y_true) ** 2

    def backward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} do not match.")
        samples = y_pred.shape[0]
        return (2 / samples) * (y_pred - y_true)


class LossMeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} do not match.")
        return np.abs(y_pred - y_true)

    def backward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} do not match.")
        samples = y_pred.shape[0]
        return np.sign(y_pred - y_true) / samples