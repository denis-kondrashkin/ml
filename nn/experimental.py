import numpy as np

from nn.containers import Sequential
from nn.module import Module


class SequentialWithPassiveLayers(Sequential):

    def __init__(self, modules, active_layers=(0,)):
        super().__init__()
        self.modules = modules
        self.active_layers = active_layers

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [m.getParameters() for i, m in enumerate(self.modules) if i in self.active_layers]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [m.getGradParameters() for i, m in enumerate(self.modules) if i in self.active_layers]


class InputSample(Module):
    """
    A module which is used for feature visualization by optimization:
    it stores a single sample and updates it according to a gradient from backpropagation.

    The module should work with 2D input of shape (1, n_feature).
    """

    def __init__(self, input_sample):
        super().__init__()

        assert input_sample.shape[0] == 1 and len(input_sample.shape) == 2

        self.input_sample = np.copy(input_sample)
        self.input_sample_grad = np.zeros_like(input_sample)

    def updateOutput(self, input):
        self.output = np.copy(self.input_sample)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        np.copyto(self.input_sample_grad, gradOutput)

    def zeroGradParameters(self):
        self.input_sample_grad.fill(0)

    def getParameters(self):
        return [self.input_sample]

    def getGradParameters(self):
        return [self.input_sample_grad]

    def __repr__(self):
        q = 'InputSample'
        return q
