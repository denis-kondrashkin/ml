import numpy as np
from projects.mnist import fetch_mnist

import matplotlib.pyplot as plt

from nn.containers import Sequential
from nn.layers import Linear, LogSoftMax, BatchNormalization, ChannelwiseScaling, Dropout, Conv2d, MaxPool2d, Flatten
from nn.activations import LeakyReLU
from nn.criterions import ClassNLLCriterion
from nn.optimizers import sgd_momentum, adam_optimizer
from nn.utils import train, ohe


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

criterion = ClassNLLCriterion()

# train settings
optimizer = adam_optimizer
optimizer_config = {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
optimizer_state = {}

n_epoch = 30
batch_size = 512

# train function
loss_history = train(
    net,
    X_train_conv,
    y_train_ohe,
    criterion,
    optimizer,
    optimizer_config,
    optimizer_state,
    n_epoch,
    batch_size
)

net.evaluate()

# test accuracy
predicted = net.forward(X_test_conv)
predicted = np.argmax(predicted, axis=1)
accuracy = 100 * np.mean(predicted == y_test)
print("Test accuracy: ", accuracy)

# train accuracy
predicted = net.forward(X_val_conv)
predicted = np.argmax(predicted, axis=1)
accuracy = 100 * np.mean(predicted == y_val)
print("Train accuracy: ", accuracy)

# logits values distribution
neuron_vals = net.modules[-2].output[:, 0]
plt.hist(neuron_vals, bins=100)

# train errors
plt.figure(figsize=(8, 6))
plt.title("Training loss")
plt.xlabel("#iteration")
plt.ylabel("loss")
plt.plot(loss_history, 'b')
plt.show()
