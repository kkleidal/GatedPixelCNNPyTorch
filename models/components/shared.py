import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from collections import deque, OrderedDict

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available()

class EMA:
    def __init__(self, _sentinal=None, alpha=None, decay=None):
        assert _sentinal is None, "Use kwargs only."
        if alpha is None and decay is None:
            raise RuntimeError("Must provide alpha or decay.")
        if decay is not None:
            assert alpha is None, "Only provide one: alpha or decay."
            self.alpha = 1 - decay
        else:
            self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value

    def get(self):
        return self.value

    def __call__(self):
        return self.get()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def __call__(self, inp):
        return inp.view(inp.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def __call__(self, inp):
        return inp.view(*([inp.size(0)] + self.shape))
