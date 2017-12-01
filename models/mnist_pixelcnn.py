import torch
import torch.nn as nn
from torch.autograd import Variable
from .components.pixelcnn import *

class MNIST_PixelCNNNew(PixelCNN):
    def __init__(self, layers=5, levels=8, conditional=False, **kwargs):
        super().__init__()
        self.conditional = conditional
        if conditional:
            self.embed = nn.Embedding(10, 32)
        layer_objs = [PixelCNNGatedLayer.primary(1, 16, 7, conditional_features=10 if conditional else None, **kwargs)]
        layer_objs = layer_objs + [
                PixelCNNGatedLayer.secondary(16, 16, 7, conditional_features=10 if conditional else None, **kwargs)
                for _ in range(1, layers)]
        self.stack = PixelCNNGatedStack(*layer_objs)
        self.out = nn.Conv2d(16, levels, 1)

    def _adapt_kwargs(self, kwargs):
        if "labels" in kwargs:
            if self.conditional:
                labels = kwargs["labels"]

                y_onehot = torch.FloatTensor(labels.size(0), 10)
                y_onehot.zero_()
                if next(self.parameters()).data.is_cuda:
                    y_onehot = y_onehot.cuda()
                y_onehot.scatter_(1, labels.data.unsqueeze(1), 1)
                y_onehot = Variable(y_onehot, requires_grad=False)

                # label_vector = self.embed(labels)
                kwargs["conditional_vector"] = y_onehot # label_vector
            del kwargs["labels"]
        return kwargs

    def __call__(self, inp, *args, **kwargs):
        _, h, _ = self.stack(inp, inp, **self._adapt_kwargs(kwargs))
        h = self.out(h)
        h = h.unsqueeze(2)
        return h
