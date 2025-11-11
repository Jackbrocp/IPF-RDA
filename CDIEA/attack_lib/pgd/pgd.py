import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from data_process import normalization


# PGD攻击
def attack_PGD(model, images, labels, conf, target=False, iterations=100, alpha=2/255, loss_fun="ce", confidence=0.0):
    criterion = nn.CrossEntropyLoss(reduction="none")  # 不要对每个维度的损失求平均
    #noises = Variable(torch.randn(images.shape), requires_grad=True)
    noises = Variable(torch.Tensor(
        np.random.uniform(-conf.eps, conf.eps, images.shape)), requires_grad=True)
    labels_onehot = torch.zeros(images.shape[0], conf.num_classes).scatter(
        1, labels.view(-1, 1), 1).to(conf.device)
    for i in range(iterations):
        inputs = (images + noises).to(conf.device)
        labels = labels.to(conf.device)
        outputs = model(normalization(inputs, conf.normalized_ops))
        # 计算损失
        if loss_fun == "ce":
            loss = criterion(outputs, labels)
        elif loss_fun == "cw":
            print("cw loss not completed...")
            exit(0)
            real = torch.sum(outputs * labels_onehot, dim=1)
            # torch.max返回，0是具体的值，1是索引
            other = torch.max(outputs * (1-labels_onehot) -
                              labels_onehot*10000, dim=1)[0]
            if target:
                loss = torch.clamp(other - real + confidence, min=0.0)
            else:
                loss = torch.clamp(real - other + confidence, min=0.0)
        loss.backward(Variable(torch.ones(loss.shape)).to(
            conf.device), retain_graph=True)  # 这里的loss没有对batch内求平均，反向的时候也是单独对每个输入算梯度
        grad_now = noises.grad
        grad_now = grad_now / \
            torch.sum(torch.abs(grad_now), dim=(1, 2, 3), keepdim=True)
        if i > 0:
            grad = grad * 0.9 + grad_now
        else:
            grad = grad_now
        with torch.no_grad():
            if target:
                noises[:] = noises - alpha * torch.sign(grad)   # 有目标就直接减去
            else:
                noises[:] = noises + alpha * torch.sign(grad)
            noises[:] = (noises + images).clamp_(conf.pixel_min,
                                                 conf.pixel_max) - images
            noises[:] = torch.clamp(noises, -conf.eps, conf.eps)
        noises.grad.data.zero_()
    return noises.detach(), noises.detach() + images
