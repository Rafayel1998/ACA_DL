from .module import Module
import numpy as np


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nicer (Glorot uniform) initialization
        stdv = 2. / (n_in + n_out)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.ones(shape=(n_out, )) * 0.01

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, inpt):
        self.output = inpt.dot(self.W.T) + self.b

        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput.dot(self.W)

        return self.gradInput

    def accGradParameters(self, inpt, gradOutput):
        self.gradW = gradOutput.T.dot(inpt)
        self.gradb = gradOutput.sum(axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q