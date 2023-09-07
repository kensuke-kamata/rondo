import math
import matplotlib.pyplot as plt
import numpy as np

import rondo
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
train_set = D.Spiral(train=True)
test_set  = D.Spiral(train=False)
train_loader = rondo.DataLoader(train_set, batch_size, shuffle=True)
test_loader  = rondo.DataLoader(test_set,  batch_size, shuffle=False)

# Model/Optimizer
model = M.MLP([hidden_size, 3])
optimizer = O.SGD(lr).setup(model)

train_losses = []
train_accs   = []
test_losses  = []
test_accs    = []
for epoch in range(max_epoch):
    # Training
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        # Forward
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc  = F.accuracy(y, t)

        # Backward
        model.cleargrads()
        loss.backward()

        # Update parameters
        optimizer.update()

        # Accumulate loss
        sum_loss += float(loss.data) * len(t)
        sum_acc  += float(acc.data)  * len(t)

    # Print loss every epoch
    avg_loss = sum_loss / len(train_set)
    avg_acc  = sum_acc  / len(train_set)
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))

    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

    # Evaluation
    sum_loss, sum_acc = 0, 0
    with rondo.no_grad():
        for x, t in test_loader:
            # Forward
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc  = F.accuracy(y, t)

            # Accumulate loss
            sum_loss += float(loss.data) * len(t)
            sum_acc  += float(acc.data)  * len(t)

    avg_loss = sum_loss / len(test_set)
    avg_acc  = sum_acc  / len(test_set)
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))

    test_losses.append(avg_loss)
    test_accs.append(avg_acc)

# Plotting the decision boundary
h = 0.01  # step size in the mesh
x = np.array([sample[0] for sample in train_set])
t = np.array([sample[1] for sample in train_set])
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

# Plotting losses over training
plt.plot(range(max_epoch), train_losses, label="Training Loss", color='blue')
plt.plot(range(max_epoch), test_losses, label="Test Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plotting accuracies over training
plt.plot(range(max_epoch), train_accs, label="Training Accuracy", color='blue')
plt.plot(range(max_epoch), test_accs, label="Test Accuracy", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
