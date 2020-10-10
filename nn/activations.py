import numpy as np

from nn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope

    def updateOutput(self, input):
        self.output = np.multiply(self.slope, input)
        self.output = np.where(input > 0, input, self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, 1, self.slope)
        np.multiply(gradOutput, self.gradInput, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def updateOutput(self, input):
        self.output = np.exp(input)
        np.subtract(self.output, 1, out=self.output)
        np.multiply(self.alpha, self.output, out=self.output)
        self.output = np.where(input > 0, input, self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.exp(input)
        np.multiply(self.alpha, self.gradInput, out=self.gradInput)
        self.gradInput = np.where(input > 0, 1, self.gradInput)
        np.multiply(gradOutput, self.gradInput, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        self.output = np.exp(input)
        np.add(1, self.output, out=self.output)
        np.log(self.output, out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.exp(input)
        np.add(1, self.gradInput, out=self.gradInput)
        np.divide(1, self.gradInput, out=self.gradInput)
        np.subtract(1, self.gradInput, out=self.gradInput)
        np.multiply(gradOutput, self.gradInput, out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"
