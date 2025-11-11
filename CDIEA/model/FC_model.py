import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC(nn.Module):
    def __init__(self, in_c, in_h, in_w, out_c, out_h, out_w):
        super(FC, self).__init__()
        self.out_c = out_c
        self.out_h = out_h
        self.out_w = out_w
        self.conv = nn.Conv2d(in_c, out_c*out_h*out_w, kernel_size=(in_h,in_w), stride=1, padding=0)

    def forward(self, X, alpha):
        X = self.conv(X)
        X = X.view(-1, self.out_c, self.out_h, self.out_w)
        return torch.sigmoid(alpha * X)

if __name__ == "__main__":
    x = np.random.randn(1,3,32,32)
    x = torch.Tensor(x)
    y = FC(3,32,32,1,4,4)(x, 0.4)
    print(y.shape)