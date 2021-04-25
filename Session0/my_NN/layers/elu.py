from .module import Module
import numpy as np


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, inpt):
        mask = inpt < 0
        self.output = np.maximum(self.alpha * (np.exp(inpt * mask) - 1), inpt)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        p_index = inpt > 0

        grad = self.output + self.alpha
        grad[p_index] = 1

        self.gradInput = grad * gradOutput

        return self.gradInput

    def __repr__(self):
        return 'ELU'