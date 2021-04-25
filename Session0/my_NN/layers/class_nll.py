from .criterion import Criterion
import numpy as np


class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))

        self.output = -np.multiply(target, np.log(input_clamp)).sum() / inpt.shape[0]

        return self.output

    def updateGradInput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        self.gradInput = - target / input_clamp / inpt.shape[0]

        return self.gradInput

    def __repr__(self):
        return 'ClassNLLCriterion'