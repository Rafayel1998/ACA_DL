from .module import Module
import numpy as np


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, inpt):
        self.output = np.log(np.exp(inpt) + 1)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = np.multiply(gradOutput, np.exp(inpt) / (np.exp(inpt) + 1))
        return self.gradInput

    def __repr__(self):
        return 'SoftPlus'