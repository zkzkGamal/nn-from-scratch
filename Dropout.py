import numpy as np
class Layer_Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = 1 - dropout_rate

    def forward(self, inputs, training=True):
        if not training:
            return inputs
        self.mask = np.random.binomial(1, self.dropout_rate,
                                       size=inputs.shape) / self.dropout_rate
        self.output = inputs * self.mask
        return self.output

    def backward(self, dvalues):
        self.output = dvalues * self.mask
        return self.output