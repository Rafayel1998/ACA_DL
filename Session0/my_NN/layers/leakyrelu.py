from .module import Module
import numpy as np


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, inpt):
        self.output = np.maximum(inpt, self.slope * inpt)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        n_index = inpt < 0

        dp = np.ones_like(inpt, dtype='float64')
        dp[n_index] = -self.slope

        self.gradInput = np.multiply(gradOutput, dp)

        return self.gradInput

    def __repr__(self):
        return 'LeakyReLU'