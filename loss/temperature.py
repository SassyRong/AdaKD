import torch
import torch.nn as nn
from torch.autograd import Function
import math

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_scale):
        ctx.lambda_scale = lambda_scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output * ctx.lambda_scale, None
        return grad_output.neg() * ctx.lambda_scale, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_scale):
        return GradientReversalFunction.apply(x, lambda_scale)

class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True) 
        self.grl = GradientReversalLayer()

    def forward(self, lambda_scale):
        return self.grl(self.global_T, lambda_scale)

# Cosine scheduler for temperature
class CosineScheduler(object):
    def __init__(self, start_value, end_value, total_steps):
        self.start = start_value
        self.end = end_value
        self.total_steps = total_steps

    def get_value(self, step):
        if step < 0:
            step = 0
        if step >= self.total_steps:
            step = self.total_steps
        cos = math.cos(math.pi * step / self.total_steps)
        value = (cos + 1.0) * 0.5
        return self.end + value * (self.start - self.end)


