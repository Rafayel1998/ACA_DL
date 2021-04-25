from .module import Module
import numpy as np


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, inpt):
        # so that we don't get NaN as result of overflow, caused bby exponentiation
        exp_inpt = np.exp(inpt - np.max(inpt, axis=-1)[:, np.newaxis])
        self.output = exp_inpt / np.sum(exp_inpt, axis=-1)[:, np.newaxis]
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * self.output * (1 - self.output)
        return self.gradInput

    def __repr__(self):
        return 'SoftMax'