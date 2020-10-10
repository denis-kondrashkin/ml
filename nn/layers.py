import numpy as np

from scipy import signal

from nn.module import Module


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super().__init__()

        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = np.dot(input, self.W.T)
        np.add(self.output, self.b, out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        batch_grads = np.multiply(gradOutput[:, :, None], input[:, None, :])
        np.mean(batch_grads, axis=0, out=self.gradW)
        np.mean(gradOutput, axis=0, out=self.gradb)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        # normalization for numerical stability
        self.output = np.subtract(input, np.max(input, axis=1, keepdims=True))
        np.exp(self.output, out=self.output)
        np.divide(self.output, self.output.sum(axis=1, keepdims=True), out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput - np.sum(np.multiply(self.output, gradOutput), axis=1, keepdims=True)
        np.multiply(self.output, self.gradInput, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        # normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        log_sum = np.log(np.sum(np.exp(self.output), axis=1, keepdims=True))
        np.subtract(self.output, log_sum, out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.subtract(input, input.max(axis=1, keepdims=True))
        self.gradInput = np.exp(self.gradInput, out=self.gradInput)
        exp_sum = self.gradInput.sum(axis=1, keepdims=True)
        np.divide(self.gradInput, exp_sum, out=self.gradInput)
        np.multiply(self.gradInput, np.sum(gradOutput, axis=1, keepdims=True), out=self.gradInput)
        np.subtract(gradOutput, self.gradInput, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        if self.moving_mean is None:
            self.moving_mean = np.zeros((1, input.shape[1]))
            self.moving_variance = np.zeros((1, input.shape[1]))

        if self.training:
            batch_mean = np.mean(input, axis=0, keepdims=True)
            batch_variance = np.var(input, axis=0, keepdims=True)

            self.moving_mean = np.subtract(self.moving_mean, batch_mean)
            np.multiply(self.alpha, self.moving_mean, out=self.moving_mean)
            np.add(self.moving_mean, batch_mean, out=self.moving_mean)

            self.moving_variance = np.subtract(self.moving_variance, batch_variance)
            np.multiply(self.alpha, self.moving_variance, out=self.moving_variance)
            np.add(self.moving_variance, batch_variance, out=self.moving_variance)

            mean = batch_mean
            var = batch_variance
        else:
            mean = self.moving_mean
            var = self.moving_variance

        self.output = input - mean
        np.divide(self.output, np.sqrt(var + self.EPS), out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_mean = np.mean(input, axis=0, keepdims=True)
        batch_variance_eps = np.var(input, axis=0, keepdims=True)
        np.add(batch_variance_eps, self.EPS, out=batch_variance_eps)

        grad_var = np.sum(np.multiply(gradOutput, input - batch_mean), axis=0, keepdims=True)
        np.multiply(-0.5, grad_var, out=grad_var)
        np.divide(grad_var, np.power(batch_variance_eps, 1.5), out=grad_var)

        grad_mean = np.sum(gradOutput, axis=0, keepdims=True)
        np.multiply(-1, grad_mean, out=grad_mean)
        np.divide(grad_mean, np.power(batch_variance_eps, 0.5), out=grad_mean)

        self.gradInput = np.subtract(input, batch_mean)
        np.multiply(2, self.gradInput, out=self.gradInput)
        np.multiply(grad_var, self.gradInput, out=self.gradInput)
        np.add(grad_mean, self.gradInput, out=self.gradInput)
        np.divide(self.gradInput, input.shape[0], out=self.gradInput)
        np.add(self.gradInput, np.divide(gradOutput, np.power(batch_variance_eps, 0.5)), out=self.gradInput)

        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super().__init__()

        stdv = 1. / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training:
            self.mask = np.random.choice([0, 1], input.shape, p=[1 - self.p, self.p])
            self.output = np.multiply(input, self.mask)
            np.divide(self.output, self.p, out=self.output)
        else:
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, self.mask)
        np.divide(self.gradInput, self.p, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1

        stdv = 1. / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        pad_input = np.pad(
            input,
            pad_width=[(0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)]
        )
        self.output = []
        for s in range(self.out_channels):
            out_channel = signal.correlate(pad_input, self.W[s:s+1], mode='valid')
            np.add(out_channel, self.b[s], out=out_channel)
            self.output.append(out_channel)
        self.output = np.stack(self.output, axis=1).squeeze()
        return self.output

    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        pad_grad = np.pad(
            gradOutput,
            pad_width=[(0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)]
        )
        self.gradInput = []
        for k in range(self.in_channels):
            in_grad = signal.correlate(pad_grad, self.W[:, k][None, :, ::-1, ::-1], mode='valid')
            self.gradInput.append(in_grad)
        self.gradInput = np.stack(self.gradInput, axis=1).squeeze()
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        pad_input = np.pad(
            input,
            pad_width=[(0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)]
        )
        for s in range(self.out_channels):
            w_grad = signal.correlate(pad_input, gradOutput[:, s:s+1], mode='valid')
            self.gradW[s] = np.mean(w_grad, axis=0)
            self.gradb[s] = np.mean(gradOutput[:, s].sum(axis=(1, 2)))

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' % (s[1], s[0])
        return q


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        assert input_h % self.kernel_size == 0
        assert input_w % self.kernel_size == 0

        reshaped_input = input.reshape(
            input.shape[0],
            input.shape[1],
            input_h // self.kernel_size,
            self.kernel_size,
            input_w // self.kernel_size,
            self.kernel_size
        )
        reshaped_input = np.swapaxes(reshaped_input, 3, 4)
        self.output = np.max(reshaped_input, axis=(4, 5))
        self.max_indices = self.output[:, :, :, :, None, None] == reshaped_input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = self.max_indices * gradOutput[:, :, :, :, None, None]
        self.gradInput = np.swapaxes(self.gradInput, 3, 4)
        self.gradInput = self.gradInput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' % (self.kernel_size, self.kernel_size)
        return q


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"

