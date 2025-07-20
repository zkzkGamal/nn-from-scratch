import numpy as np

class Loss:
    def regularization_loss(self , layer):
        if layer.weight_regularizer_l1 > 0:
            return layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            return layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 > 0:
            return layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            return layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return 0
    
    def calculate(self, output , y):
        sample_lossess = self.forward(output , y)
        data_loss = np.mean(sample_lossess)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
                ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * 
                                         y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples
        return self.dinputs
    
