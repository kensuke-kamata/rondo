import math
import matplotlib.pyplot as plt
import numpy as np

import rondo.datasets as D
import rondo.functions as F
import rondo.models as M
import rondo.optimizers as O

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# Data
x, t = D.get_spiral(train=True)
data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

# Model/Optimizer
model = M.MLP([hidden_size, 3])
optimizer = O.SGD(lr).setup(model)

# Training
losses = []
for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # Get batch
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # Forward
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)

        # Backward
        model.cleargrads()
        loss.backward()

        # Update parameters
        optimizer.update()

        # Accumulate loss
        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

    losses.append(avg_loss)

# Plotting the decision boundary
h = 0.01  # step size in the mesh
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh using the model.
X_test = np.c_[xx.ravel(), yy.ravel()]
score = model(X_test)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)

# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=t, s=40, edgecolors='k', cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# Plotting
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
