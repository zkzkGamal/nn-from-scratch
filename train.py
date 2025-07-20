import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs

import matplotlib
matplotlib.use("TkAgg")  # or "Agg" if on server (headless, no GUI)
import matplotlib.pyplot as plt

from Optimizer import Optimizer_Adam
nnfs.init()

from Loss import *
from Activation import *
from Dropout import Layer_Dropout
from Dense import Layer_Dense


print("Spiral Data Generation")
# Generate spiral data
X , y = spiral_data(samples=1000, classes=3)

print("Data Shape:", X.shape, y.shape)

print("starting model creation")
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=0.01,
                     bias_regularizer_l2=0.01)

activation_1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(64, 3)
activation_2 = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(lr=0.05, decay=1e-5)

print("Model created, starting training")
epochs = 10000
for epoch in range(epochs + 1):
    # Forward pass
    dense1.forward(X)
    activation_1.forward(dense1.output)
    dropout1.forward(activation_1.output)
    dense2.forward(dropout1.output)

    data_loss = activation_2.forward(dense2.output, y)
    regularization_loss = activation_2.loss.regularization_loss(dense1) + \
                          activation_2.loss.regularization_loss(dense2)
    print(f'Epoch {epoch}, Data Loss: {data_loss:.4f}, Regularization Loss: {regularization_loss:.4f}')
    loss = data_loss + regularization_loss
    
    try:
        predictions = np.argmax(activation_2.output, axis=1)
    except:
        predictions = np.argmax(activation_2.output, axis=-1)
    
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f})'+
              f'lr: {optimizer.current_lr:.5f}')
    # Backward pass
    print("Starting backward pass")
    activation_2.backward(activation_2.output, y)
    dropout1.backward(activation_2.dinputs)
    activation_1.backward(dropout1.dinputs)
    dense2.backward(activation_1.dinputs)
    dense1.backward(dense2.dinputs)
    
    # Update parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# save plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', s=10)
plt.title('Spiral Data Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Class')
plt.savefig('spiral_classification.png')
plt.show()

# validate the model
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation_1.forward(dense1.output)
dense2.forward(activation_1.output)
loss = activation_2.forward(dense2.output, y_test)
predictions = np.argmax(activation_2.output, axis=1)
if len(y_test.shape) == 2:
    y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_true)
print(f'Validation accuracy: {accuracy:.3f}, loss: {loss:.3f}')