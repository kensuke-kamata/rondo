import numpy as np
import rondo
import rondo.functions as F

# Generate toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)             # Input data (100 samples, 1 feature)
y = 5 + 2 * x + np.random.rand(100, 1) # True output data with a linear relationship and some noise

# Initialize parameters for the linear regression model
W = rondo.Variable(np.zeros((1, 1))) # Weight
b = rondo.Variable(np.zeros(1))      # Bias

# Prediction function using the linear model
def predict(x):
    y = F.matmul(x, W) + b
    return y

# Hyperparameters for gradient descent
lr = 0.1    # Learning rate
iters = 100 # Number of iterations

# Training loop
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)
