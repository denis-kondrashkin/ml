import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader, TensorDataset

from projects.notmnist import fetch_notmnist

torch.set_num_threads(12)

X_train, y_train, X_test, y_test = fetch_notmnist.load_notmnist()

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)


num_filters1 = 1
# filters1 = 0.5 * (torch.rand((num_filters1, 1, 3, 3)) - 0.5)
# filters_bias1 = 0.25 * (torch.rand(num_filters1, 1, 1) - 0.5)
weights1 = ((6 / (num_filters1 * 28 * 28 + 128)) ** 0.5) * (2 * torch.rand((num_filters1 * 28 * 28, 128)) - 1)
bias1 = 0.25 * (torch.rand(128) - 0.5)
weights2 = ((2 * 6 / (128 + 32)) ** 0.5) * (2 * torch.rand((128, 32)) - 0.5)
bias2 = 0.25 * (torch.rand(32) - 0.5)
weights3 = ((2 * 6 / (32 + 10)) ** 0.5) * (2 * torch.rand((32, 10)) - 0.5)
bias3 = 0.25 * (torch.rand(10) - 0.5)

params = [
    # filters1,
    # filters_bias1,
    weights1,
    bias1,
    weights2,
    bias2,
    weights3,
    bias3
]


def model(input_x, training):
    # input_x = F.conv2d(input=input_x, weight=filters1, stride=1, padding=1)
    # input_x += filters_bias1
    input_x = input_x.view(-1, num_filters1 * 28 * 28)
    input_x = torch.matmul(input_x, weights1)
    input_x += bias1
    input_x = F.relu(input_x)
    input_x = F.dropout(input_x, 0.2, training)
    input_x = input_x.view(-1, 128)
    input_x = torch.matmul(input_x, weights2)
    input_x += bias2
    input_x = F.relu(input_x)
    input_x = F.dropout(input_x, 0.2, training)
    input_x = input_x.view(-1, 32)
    input_x = torch.matmul(input_x, weights3)
    input_x += bias3
    return input_x


def init(step_params):
    for p in step_params:
        p.requires_grad = True


def step(step_params):
    for p in step_params:
        p.data -= lr * p.grad.data
    for p in step_params:
        p.grad.data.zero_()


n_epochs = 30
batch_size = 64
lr = 0.06
# optimizer = optim.Adam(params=params, lr=lr, weight_decay=0.0001)

trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
init(params)
train_log = []
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # optimizer.zero_grad()

        outputs = model(inputs, training=True)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        step(params)
        # optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            train_log.append(running_loss)
            running_loss = 0.0

num_correct = 0
with torch.no_grad():
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        outputs = model(inputs, training=False)
        predicted_labels = torch.argmax(outputs, dim=1)
        num_correct += torch.sum(torch.eq(labels, predicted_labels)).item()
print('train accuracy: ', num_correct / y_train.size()[0])

testset = TensorDataset(X_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
num_correct = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs, training=False)
        predicted_labels = torch.argmax(outputs, dim=1)
        num_correct += torch.sum(torch.eq(labels, predicted_labels)).item()
print('test accuracy: ', num_correct / y_test.size()[0])

plt.figure(figsize=(7,7))
plt.plot(train_log)
plt.show()
