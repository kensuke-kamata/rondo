import os

import matplotlib.pyplot as plt

import rondo
import rondo.functions as F

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = rondo.datasets.MNIST(train=True)
test_set  = rondo.datasets.MNIST(train=False)

train_loader = rondo.DataLoader(train_set, batch_size)
test_loader = rondo.DataLoader(test_set, batch_size, shuffle=False)

model = rondo.models.MLP([hidden_size, hidden_size, 10], activation=F.relu)
optimizer = rondo.optimizers.Adam().setup(model)

path = '.cache/mnist.npz'
if os.path.exists(path):
    model.load(path)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc  = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with rondo.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc  = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))

model.save(path)
