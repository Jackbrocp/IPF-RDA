""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from tensorboardX import SummaryWriter


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GroupOP(nn.Module):
    def __init__(self, in_c, in_h, in_w, out_c, out_h, out_w):
        super(GroupOP, self).__init__()
        self.out_c = out_c
        self.out_h = out_h
        self.out_w = out_w
        self.max_pool = nn.MaxPool2d(20)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=15, stride=15)
        self.fc = nn.Conv2d(in_c, out_c*out_h*out_w, kernel_size=(in_h,in_w), stride=1, padding=0)
    
    def forward(self, x):
        x = self.max_pool(x)
        x = self.fc(x)
        x = x.view(-1, self.out_c, self.out_h, self.out_w)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=[], is_group = "0"):
        super(UNet, self).__init__()
        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        self.display_names = ['loss_stack', 'matrix_iou_stack']
        self.gpu_ids = gpu_ids
        self.bce_loss = nn.BCELoss()
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device(
        #     'cpu')
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        # print(list(self.down1.parameters()))
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = Up(1024, 512, False)
        self.up2 = Up(512, 256, False)
        self.up3 = Up(256, 128, False)
        self.up4 = Up(128, 64, False)
        self.outc = OutConv(64, out_ch)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        # 如果要做组稀疏，则需要在后面加一个模块
        self.is_group = is_group
        self.group = GroupOP(1, 15, 15, 1, 15, 15)

    def forward(self, x, alpha):
        x1 = self.inc(x)
        # print("x1: ", x1.shape)
        x2 = self.down1(x1)
        # print("x2: ", x2.shape)
        x3 = self.down2(x2)
        # print("x3: ", x3.shape)
        x4 = self.down3(x3)
        # print("x4: ", x4.shape)
        x5 = self.down4(x4)
        # print("x5: ", x5.shape)
        x = self.up1(x5, x4)
        # print("x: ", x.shape)
        x = self.up2(x, x3)
        # print("x: ", x.shape)
        x = self.up3(x, x2)
        # print("x: ", x.shape)
        x = self.up4(x, x1)
        # print("x: ", x.shape)
        logits = self.outc(x)
        if self.is_group == "1":
            logits = self.group(logits) # 如果是组稀疏，则需要再做一下处理
        # print("logits: ", logits.shape)
        return torch.sigmoid(alpha * logits)

if __name__ == "__main__":
    
    model = UNet(3,1, is_group="1")
    model.train()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())))
    x = torch.Tensor(np.random.randn(1,3,300,300))
    y = model(x, 0.5)
    print(y.shape)
    
    model = models.resnet50(pretrained=True)
    model.train()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())))

    

