import numpy as np
from projects.mnist import fetch_mnist

import matplotlib.pyplot as plt

from nn.containers import Sequential
from nn.layers import Linear, LogSoftMax, BatchNormalization, ChannelwiseScaling, Dropout, Conv2d, MaxPool2d, Flatten
from nn.activations import LeakyReLU
from nn.criterions import ClassNLLCriterion
from nn.optimizers import sgd_momentum, adam_optimizer
from nn.utils import train, ohe
from nn.experimental import SequentialWithPassiveLayers, InputSample


# fetching mnist data
X_train, y_train, X_val, y_val, X_test, y_test = fetch_mnist.load_dataset()
y_train_ohe, y_val_ohe, y_test_ohe = ohe(y_train, 10), ohe(y_val, 10), ohe(y_test, 10)

# data for networks without convolutions
X_train_flatten = X_train.reshape((X_train.shape[0], 28 * 28))
X_val_flatten = X_val.reshape((X_val.shape[0], 28 * 28))
X_test_flatten = X_test.reshape((X_test.shape[0], 28 * 28))

# data for networks with convolutions
X_train_conv = X_train.reshape((X_train.shape[0], 1, 28, 28))
X_val_conv = X_val.reshape((X_val.shape[0], 1, 28, 28))
X_test_conv = X_test.reshape((X_test.shape[0], 1, 28, 28))


# add layers here
net = Sequential()
net.add(Linear(28 * 28, 64))
net.add(LeakyReLU())
net.add(Linear(64, 10))
net.add(LogSoftMax())

criterion = ClassNLLCriterion()

# train settings
optimizer = adam_optimizer
optimizer_config = {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
optimizer_state = {}

n_epoch = 20
batch_size = 512

loss_history = train(
    net,
    X_train_flatten,
    y_train_ohe,
    criterion,
    optimizer,
    optimizer_config,
    optimizer_state,
    n_epoch,
    batch_size
)

# digit to visualize by logsoftmax
digit = 0
y_sample = np.zeros((1, 10))
y_sample[0, digit] = 1

# noise as input
X_sample = (np.random.random((1, 28 * 28)) - 0.5) / 20
sample_layer = InputSample(X_sample)

# net adapted for updating only input layer
net2 = SequentialWithPassiveLayers([sample_layer] + net.modules)
net2.train()

n_epoch = 100
batch_size = 1

optimizer = adam_optimizer
optimizer_config = {'learning_rate': 3e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
optimizer_state = {}

loss_history = train(
    net2,
    X_sample,
    y_sample,
    criterion,
    optimizer,
    optimizer_config,
    optimizer_state,
    n_epoch,
    batch_size
)

plt.imshow(sample_layer.input_sample[0].reshape((28, 28)), cmap='gray')
plt.show()
