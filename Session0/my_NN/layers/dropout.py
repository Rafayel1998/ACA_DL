from scipy.stats import bernoulli
from .module import Module
import numpy as np


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, inpt):
        if self.training:
            self.mask = np.tile(bernoulli.rvs(self.p, size=inpt.shape[-1]),
                                (inpt.shape[0], 1))
        else:
            self.mask = np.ones(inpt.shape)

        self.output = inpt * self.mask * \
                      inpt.shape[-1] / (np.maximum(self.mask.sum(), 1) / inpt.shape[0])
        # the last line is the equivalent of multiplying with 1/p
        # we do this because the mask doesn't strictly dropout with p probability

        return self.output

    def updateGradInput(self, inpt, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask * \
                             inpt.shape[-1] / (np.maximum(self.mask.sum(), 1) / inpt.shape[0])
        else:
            self.gradInput = gradOutput

        return self.gradInput

    def __repr__(self):
        return 'Dropout'