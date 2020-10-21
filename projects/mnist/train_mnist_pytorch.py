from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
from projects.mnist import fetch_mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(12)
device = torch.device("cuda:0")


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(in_features=12 * 7 * 7, out_features=64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = x.view(-1, 12 * 7 * 7)
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return x


X_train, y_train, X_val, y_val, X_test, y_test = fetch_mnist.load_dataset()
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
X_val = torch.from_numpy(X_val)
X_train = X_train.view(X_train.size()[0], 1, X_train.size()[1], X_train.size()[2])
X_test = X_test.view(X_test.size()[0], 1, X_test.size()[1], X_test.size()[2])
X_val = X_val.view(X_val.size()[0], 1, X_val.size()[1], X_val.size()[2])
y_train = torch.from_numpy(y_train.astype(np.long))
y_test = torch.from_numpy(y_test.astype(np.long))
y_val = torch.from_numpy(y_val.astype(np.long))

net = Model()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

batch_size, n_epochs = 512, 30

# train
net.train()
trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

# accuracy on test data
net.eval()
testset = TensorDataset(X_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
num_correct = 0
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    predicted_labels = torch.argmax(outputs, dim=1)
    num_correct += torch.sum(torch.eq(labels, predicted_labels)).item()
print(num_correct / y_test.size()[0])

# accuracy on eval data
net.eval()
valset = TensorDataset(X_val, y_val)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
num_correct = 0
for i, data in enumerate(valloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    predicted_labels = torch.argmax(outputs, dim=1)
    num_correct += torch.sum(torch.eq(labels, predicted_labels)).item()
print(num_correct / y_val.size()[0])

# save model into file
state_dict = net.state_dict()
torch.save(state_dict, 'model_mnist.pt')

# load model from file
state_dict = torch.load('model_mnist.pt')
net.load_state_dict(state_dict)

# examples of wrong classification
net.eval()
valset = TensorDataset(X_val, y_val)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
predictions = []
for i, data in enumerate(valloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    predicted_labels = torch.argmax(outputs, dim=1)
    predictions.append(predicted_labels)
    num_correct += torch.sum(torch.eq(labels, predicted_labels)).item()
predictions = torch.cat(predictions).to('cpu')

error_idx = ~predictions.eq(y_val)
X_val_error = X_val[error_idx]
y_val_error = y_val[error_idx]

ind = 0
print(y_val_error[ind])
plt.imshow(X_val_error[ind][0], cmap='gray_r')
plt.show()
