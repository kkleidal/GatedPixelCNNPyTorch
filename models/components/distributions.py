import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Multinoulli:
    def __init__(self, logits, dim=1):
        self.dim = dim
        self.logits = logits
        self.log_probs = F.log_softmax(logits, dim=self.dim)
        self.probs = torch.exp(self.log_probs)
        self.D = self.probs.size(dim)
        S = len(logits.size())
        self.permutation = [(i if i < dim else i + 1) if i != S - 1 else dim for i in range(S)]
        self.permutation_r = [(i if i < dim else i - 1) if i != dim else S - 1 for i in range(S)]
        self.flat_probs = self.probs.permute(*self.permutation).contiguous().view(-1, self.D)
        self.flat_log_probs = self.log_probs.permute(*self.permutation).contiguous().view(-1, self.D)

    def _sample(self, flat_probs):
        N, level_count = flat_probs.size()
        val = torch.rand(N, 1)
        if flat_probs.is_cuda:
            val = val.cuda()
        cutoffs = torch.cumsum(flat_probs, dim=1)
        _, idx = torch.max(cutoffs > val, dim=1)
        size_goal = [-1 if i == 0 else dim for i, dim in enumerate(self.probs.size()) if i != self.dim]
        idx = idx.view(*(size_goal + [1]))
        idx = idx.permute(*self.permutation_r).contiguous()
        idx = idx.squeeze(self.dim)
        return Variable(idx, requires_grad=False)

    @property
    def MAP(self):
        _, idces = self.logits.max(dim=self.dim, keepdim=False)
        return idces

    def sample(self):
        sample = self._sample(self.flat_probs.data)
        return sample

    def sample_n(self, n):
        N, D = self.flat_probs.size()
        flat_probs = self.flat_probs.unsqueeze(0).expand(n, -1, -1).contiguous().view([n * N, D])
        out = self._sample(flat_probs.data)
        N = self.probs.size(0)
        out = out.unsqueeze(0).view(*([n, N] + list(out.size())[1:]))
        return out

    def log_prob(self, value):
        value = value.unsqueeze(self.dim).permute(self.permutation).contiguous().view(-1, 1).squeeze(1)
        N = self.flat_log_probs.size(0)
        assert value.size(0) == N
        idces = torch.arange(0, N).type(torch.LongTensor)
        if value.data.cuda:
            idces = idces.cuda()
        individual = self.flat_log_probs[idces, value]
        return individual.sum()

if __name__ == "__main__":
    logits = torch.zeros((5, 2, 1, 3, 4))
    logits.fill_(-10000)
    logits[:, 0, :, :, :] = 0
    logits[:, 1, :, 1, :] = 10000
    print(logits)
    logits = Variable(logits, requires_grad=False)
    dist = Multinoulli(logits)
    x = dist.sample()
    print(x)

