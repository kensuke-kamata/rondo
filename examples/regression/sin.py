# An example of non-linear regression (sin) using Rondo

import numpy as np
import matplotlib.pyplot as plt

import rondo
import rondo.functions as F
import rondo.layers as L

# Generate toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# Define the neural network
l1 = L.Linear(10) # Output size
l2 = L.Linear(1)

# Prediction using the neural network
def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2      # Learning rate
iters = 10000 # Number of iterations

losses = [] # Store loss values for visualization
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

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
