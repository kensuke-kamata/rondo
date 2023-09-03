# An example of non-linear regression (sin) using Rondo

import numpy as np
import matplotlib.pyplot as plt

import rondo.functions as F
import rondo.models as M
import rondo.optimizers as O

# Generate toy dataset
np.random.seed(0)
x = np.random.rand(100, 1).astype(np.float32)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model     = M.MLP([10, 1])
# optimizer = O.SGD(lr=0.2).setup(model)
optimizer = O.MomentumSGD().setup(model)

iters  = 10000 # Number of iterations
losses = []    # Store loss values for visualization
for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    # Update parameters using the optimizer
    optimizer.update()

    if i % 1000 == 0:
        print(loss)
        losses.append(loss.data)

# Plotting
# Plot the dot graph
model.plot(x)

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
