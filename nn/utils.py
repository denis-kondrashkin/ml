import numpy as np


def ohe(y, num_classes):
    y_ohe = np.zeros((y.shape[0], num_classes))
    y_ohe[np.arange(y.shape[0]), y] = 1
    return y_ohe


def get_batches(dataset, batch_size):
    data_x, data_y = dataset
    n_samples = data_x.shape[0]

    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield data_x[batch_idx], data_y[batch_idx]


def train(net, X, Y, criterion, optimizer, optimizer_config, optimizer_state, n_epoch, batch_size):
    net.train()
    loss_history = []
    for i in range(n_epoch):
        for x_batch, y_batch in get_batches((X, Y), batch_size):
            net.zeroGradParameters()

            # Forward
            predictions = net.forward(x_batch)
            loss = criterion.forward(predictions, y_batch)

            # Backward
            grad_loss = criterion.backward(predictions, y_batch)
            net.backward(x_batch, grad_loss)

            # Update weights
            optimizer(
                net.getParameters(),
                net.getGradParameters(),
                optimizer_config,
                optimizer_state
            )

            loss_history.append(loss)
        print('Current loss at %d iteration: %f' % (i, loss))
    return loss_history

