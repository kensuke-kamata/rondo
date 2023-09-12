import numpy as np
import matplotlib.pyplot as plt

import rondo
from rondo.dataloaders import SeqDataLoader
from rondo.datasets import SinCurve
from rondo.functions import mean_squared_error
from rondo.optimizers import Adam
# from rondo.models import RNN
from rondo.models import LSTM

# Hyperparameters
max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batchsize=batch_size)
seqlen = len(train_set)

# model = RNN(hidden_size, 1)
model = LSTM(hidden_size, 1)
optimizer = Adam().setup(model)

# Training
for epoch in range(max_epoch):
    model.reset()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset()
pred_list = []

with rondo.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
