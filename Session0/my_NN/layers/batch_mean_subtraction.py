from .module import Module
import numpy as np


class BatchMeanSubtraction(Module):
    def __init__(self, alpha=0.95):
        super(BatchMeanSubtraction, self).__init__()

        self.alpha = alpha
        self.old_mean = None

    def updateOutput(self, inpt):
        batch_mean = np.mean(inpt, axis=0)
        batch_size = inpt.shape[0]
        self.output = inpt.astype('float64')
        # we don't want to do batch norm during evaluation
        alpha = self.alpha if self.training else 1
        # when batch size is one result would be 0
        # so we don't do batch norm in that case
        if batch_size > 1 or not self.training:
            mean_to_subtract = batch_mean
            if self.old_mean is not None:
                mean_to_subtract = self.old_mean * alpha + batch_mean * (1 - alpha)
            self.old_mean = mean_to_subtract
            self.output -= mean_to_subtract
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        batch_size = inpt.shape[0]
        self.gradInput = gradOutput.astype('float64')
        if batch_size > 1:
            self.gradInput *= (batch_size - 1 + self.alpha) / batch_size
        return self.gradInput

    def __repr__(self):
        return 'BatchMeanNormalization'