# An example of non-linear regression (sin) using Rondo

import numpy as np
import matplotlib.pyplot as plt

import rondo
import rondo.functions as F

# Generate toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# Initialize parameters
I, H, O = 1, 10, 1 # Input, Hidden, Output sizes
W1 = rondo.Variable(0.01 * np.random.randn(I, H))
b1 = rondo.Variable(np.zeros(H))
W2 = rondo.Variable(0.01 * np.random.randn(H, O))
b2 = rondo.Variable(np.zeros(O))

# Prediction using the neural network
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2      # Learning rate
iters = 10000 # Number of iterations

losses = [] # Store loss values for visualization
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)
        losses.append(loss.data)

# Plotting
# Plot the real data as scatter points
plt.figure(figsize=(10, 8))
plt.scatter(x, y, label='Real Data', c='blue', alpha=0.5)

# Plot the neural network predictions as a line
sorted_indices = x.argsort(axis=0).ravel()
plt.plot(x[sorted_indices], y_pred.data[sorted_indices], label='NN Predictions', color='red')

plt.legend()
plt.title('Real Data vs Neural Network Predictions')
plt.show()

# Plot the loss over time
plt.figure(figsize=(10, 8))
plt.plot(losses)
plt.title('Loss over Iterations')
plt.xlabel('Iterations (in thousands)')
plt.ylabel('Loss Value')
plt.show()
