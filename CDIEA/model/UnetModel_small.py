""" Full assembly of the parts to form the complete network """
from warnings import filters
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class UNet_small(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=[], is_group = "0"):
        super(UNet_small, self).__init__()
        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        self.display_names = ['loss_stack', 'matrix_iou_stack']
        self.gpu_ids = gpu_ids
        self.bce_loss = nn.BCELoss()
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device(
        #     'cpu')

        # filters = [64, 128, 256, 512, 1024]
        filters = [16, 32, 64, 128, 256]

        self.inc = DoubleConv(in_ch, filters[0])
        self.down1 = Down(filters[0], filters[1])
        # print(list(self.down1.parameters()))
        self.down2 = Down(filters[1], filters[2])
        # self.down3 = Down(filters[2], filters[3])
        self.drop3 = nn.Dropout2d(0.5)
        # self.down4 = Down(filters[3], filters[4])
        self.drop4 = nn.Dropout2d(0.5)
        # self.up1 = Up(filters[4], filters[3], False)
        # self.up2 = Up(filters[3], filters[2], False)
        self.up3 = Up(filters[2], filters[1], False)
        self.up4 = Up(filters[1], filters[0], False)
        self.outc = OutConv(filters[0], out_ch)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        # 如果要做组稀疏，则需要在后面加一个模块
        self.is_group = is_group
        self.group = GroupOP(1, 15, 15, 1, 15, 15)

    def forward(self, x, alpha):
        '''
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
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.is_group == "1":
            logits = self.group(logits) # 如果是组稀疏，则需要再做一下处理
        # print("logits: ", logits.shape)
        return torch.sigmoid(alpha * logits)

if __name__ == "__main__":
    model = UNet_small(3,1, is_group="1")
    model.train()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())))
    x = torch.Tensor(np.random.randn(1,3,300,300))
    y = model(x, 0.5)
    print(y.shape)
    

