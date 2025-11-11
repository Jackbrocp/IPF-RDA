import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# 用于MNIST数据集的CNN模型
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            )
        self.fc = nn.Sequential(
            nn.Linear(4*4*64, 200),
            nn.ReLU(),
            nn.Linear(200,  200),
            nn.ReLU(),
            nn.Linear(200,  10),
            )
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*64)
        x = self.fc(x)
        return x
