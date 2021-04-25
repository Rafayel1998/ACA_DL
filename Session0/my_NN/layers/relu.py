from .module import Module
import numpy as np


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, inpt):
        self.output = np.maximum(inpt, 0)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        p_index = inpt > 0

        dp = np.zeros_like(inpt)
        dp[p_index] = 1

        self.gradInput = np.multiply(gradOutput, dp)

        return self.gradInput

    def __repr__(self):
        return 'ReLU'