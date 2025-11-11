import torch
import torch.nn as nn
import torch.nn.functional as F

class NO(nn.Module):
    def __init__(self):
        super(NO, self).__init__()

    def forward(self, X, alpha):
        return torch.sigmoid(alpha * X)