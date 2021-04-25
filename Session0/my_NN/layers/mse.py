from .criterion import Criterion
import numpy as np


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, inpt, target):
        self.output = np.sqrt(np.mean((target - inpt) ** 2))
        return self.output

    def updateGradInput(self, inpt, target):
        self.gradInput = (inpt - target) / self.output
        return self.gradInput

    def __repr__(self):
        return 'MSECriterion'