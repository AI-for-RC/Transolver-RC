import torch
import torch.nn as nn
import torch.nn.functional as F


class TestLoss(object):
    def __init__(self, p=2, epsilon=1e-6):
        super(TestLoss, self).__init__()
        self.p = p
        self.epsilon = epsilon

    def rel(self, out, y):
        num_examples = out.size()[0]
        diff_norms = torch.norm(out.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        return torch.mean(diff_norms / y_norms)


    def __call__(self, out, y):
        return self.rel(out, y)
